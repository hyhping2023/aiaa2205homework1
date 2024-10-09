import torch
from torch import nn
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
import sys
from collections import OrderedDict
import argparse

class ResnetBlock():
    def init_weights(self, m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
    def get_conv_block(self, in_channels, out_channels, conv_layers = 5, index = 0):
        def get_conv_dict(in_channels, out_channels, num:list, index):
            assert len(num) == 2
            conv_layers = OrderedDict()
            for _ in range(num[0], num[1]):
                conv_layers.update(
                    {
                f'conv{index}_{_}': nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                f'bn{index}_{_}': nn.BatchNorm2d(out_channels),
                f'relu{index}_{_}': nn.ReLU(),
            }
                )
            return conv_layers
        conv_dict = OrderedDict()
        if in_channels == out_channels:
            conv_dict.update(get_conv_dict(in_channels, out_channels, [0,conv_layers], index))
            return conv_dict
        else:
            conv_dict.update(get_conv_dict(in_channels, out_channels, [0,1], index))
            conv_dict.update(get_conv_dict(out_channels, out_channels, [1,conv_layers], index))
            return conv_dict
    def __init__(self, in_channels, out_channels, conv_layers:list, layers = 5, device = 'cuda'):
        assert layers == len(conv_layers)
        self.layers = layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut = None
        self.blocks = []
        for _ in range(layers):
            if in_channels != out_channels and _ == 0:
                self.blocks.append(nn.Sequential(self.get_conv_block(in_channels, out_channels, conv_layers[_], _)))
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0),
                )
            else:
                self.blocks.append(nn.Sequential(self.get_conv_block(out_channels, out_channels, conv_layers[_], _)))
        for item in self.blocks:
            item.apply(self.init_weights)
            item = item.to(device)
        for item in self.shortcut:
            item.apply(self.init_weights)
            item = item.to(device)
        self.net = [block for block in self.blocks] + [self.shortcut]
    def forward(self, x):
        for inx, block in enumerate(self.blocks):
            x_temp = block(x)
            if inx == 0:
                x = x_temp + self.shortcut(x)
            else:
                x = x + x_temp
        return x
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        params = []
        for block in self.blocks:
            params.extend(block.parameters())
        return (*params, *self.shortcut.parameters())

    def changeMode(self, mode):
        if mode == 'train':
            for block in self.blocks:
                block.train()
            self.shortcut.train()
        elif mode == 'eval':
            for block in self.blocks:
                block.eval()
            self.shortcut.eval()
        

class MyResnetCNN():
    def __init__(self, device='cuda'):
        self.device = device
        self.net0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(7, 7), stride=2, padding=3),
            nn.BatchNorm2d(8),nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        ).to(device)
        self.net1 = ResnetBlock(8, 16, [2, 2, 2, 2, 2], layers=5, device = device)
        self.net2 = ResnetBlock(16, 32, [2, 2, 2, 2, 2, 2, 2], layers=7, device = device)
        self.net3 = ResnetBlock(32, 64, [2, 2, 2, 2, 2, 2, 2, 2, 2], layers=9, device = device)
        self.net4 = ResnetBlock(64, 128, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], layers=11, device = device)
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)
        self.init_weights(self.net0)
        self.init_weights(self.net4)
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) == nn.BatchNorm2d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        
    def forward(self, x):
        self.changeMode('train')
        x0 = self.net0.forward(x)
        x1 = self.net1.forward(x0)
        x2 = self.net2.forward(x1)
        x3 = self.net3.forward(x2)
        x4 = self.net4.forward(x3)
        result = self.output(x4)
        return result
    def changeMode(self, mode):
        if mode == 'train':
            self.net0.train()
            self.net1.changeMode('train')
            self.net2.changeMode('train')
            self.net3.changeMode('train')
            self.net4.changeMode('train')
            self.output.train()
        elif mode == 'eval':
            self.net0.eval()
            self.net1.changeMode('eval')
            self.net2.changeMode('eval')
            self.net3.changeMode('eval')
            self.net4.changeMode('eval')
            self.output.eval()
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return (*self.net0.parameters() ,*self.net1(), *self.net2(), *self.net3(), *self.net4(), *self.output.parameters())
    def predict(self, x):
        self.changeMode('eval')
        with torch.no_grad():
            return self.forward(x) 
        
        
def dataLoader(list_videos, mfcc_path, batch_size, shrink, testmode = False, ratio = 0.2) -> list:
    '''
    The return values are X, x_t, y, y_t
    X stands for train X
    x_t stands for test X
    y stands for train y
    y_t stands for test y
    They are all list.
    testmode is to output all data for training
    '''
    fread = open(list_videos, "r")
    label_list = []
    df_videos_label = {}
    for line in open(list_videos).readlines()[1:]:
        video_id, category = line.strip().split(",")
        df_videos_label[video_id] = category

    raw_data = []
    for line in tqdm(fread.readlines()[1:shrink]):
        path = os.path.join(mfcc_path, line.strip().split(",")[0] + ".mfcc.csv") 
        if not os.path.exists(path):
            continue
        label_list.append(int(df_videos_label[line.strip().split(",")[0]])) 
        array = np.genfromtxt(path, delimiter=";")
        array = torch.from_numpy(array).float()
        raw_data.append(array)
    padding_data = pad_sequence(raw_data, batch_first=True)

    Y = torch.from_numpy(np.array(label_list)).long()
    print(padding_data.shape, Y.shape)

    X, x_t, y, y_t = train_test_split(padding_data, Y, test_size=ratio, random_state=18877)
    if testmode:
        X = torch.cat([X, x_t])
        y = torch.cat([y, y_t])
    print(X.shape, y.shape, x_t.shape, y_t.shape)

    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    x_t = x_t.reshape(x_t.shape[0], 1, x_t.shape[1], x_t.shape[2])

    x_loader = []
    x_t_loader = []
    y_loader = []
    y_t_loader = []
    items = [[x_loader, y_loader], [x_t_loader, y_t_loader]]
    for inx, item in enumerate([[X, y], [x_t, y_t]]): 
        n = 0
        while n < item[0].shape[0]:
            end = min(X.shape[0], n + batch_size)
            items[inx][0].append(item[0][n:end])
            items[inx][1].append(item[1][n:end])
            n += batch_size
    return x_loader, x_t_loader, y_loader, y_t_loader

def loadModel(load:bool, output_file, device):
    if load:
        with open(output_file, 'rb') as f:
            checkpoint = pickle.load(f)
            model = checkpoint['CNN']
            optimizer = checkpoint['optimizer']
            return model, optimizer
    else:
        model = MyResnetCNN(device=device)
        return model, torch.optim.Adam(model(), lr=0.01, weight_decay=1e-4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cont', action='store_true', default=False)
    parser.add_argument('--final', action='store_true', default=False)
    args = parser.parse_args()

    list_videos = '../labels/trainval.csv'
    feat_appendix = '.csv'
    mfcc_path = '../mfcc/'
    output_file = '../models/mfcc-100.resnet.model'
    device = 'cuda'
    batch_size = 512

    X, x_t, y, y_t = dataLoader(list_videos, mfcc_path, batch_size, shrink = 10000, testmode = args.final, ratio = 0.2)
    CNN, optimizer = loadModel(args.cont, output_file, device)

    loss = nn.CrossEntropyLoss()
    print(CNN.net1.net, '\n', CNN.net2.net,'\n', CNN.net3.net,'\n',CNN.net4.net, '\n', CNN.output)
    
    epoches = []
    scores = []
    losses = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(200):
        for inx, (x_iter, y_iter) in enumerate(zip(X, y)):
            optimizer.zero_grad()
            x_iter = x_iter.cuda()
            y_iter = y_iter.cuda()
            y_hat = CNN.forward(x_iter)
            l = loss(y_hat, y_iter)
            l.backward()
            optimizer.step()
            x_iter = x_iter.cpu()
            y_iter = y_iter.cpu()
            print(f'Train Epoch {epoch + 1} Batch {inx + 1} Finished', end='\r')
        print(f'Train Epoch {epoch + 1} Finished             ', end='\r')
        if epoch % 10 == 9:
            epoches.append(epoch)
            scores.append(l.item())
            test_y_hat = []
            for x_t_iter, y_t_iter in zip(x_t, y_t):
                x_t_iter = x_t_iter.cuda()
                test_y_hat.append(CNN.predict(x_t_iter).cpu())
                x_t_iter = x_t_iter.cpu()
            test_y = torch.cat(y_t, dim=0)
            test_y_hat = torch.cat(test_y_hat, dim=0)
            test_y_hat = test_y_hat.cpu().detach().numpy()
            test_y = test_y.cpu().detach().numpy()
            test_y_hat = test_y_hat.argmax(axis=1)
            accuracy = (test_y == test_y_hat).mean()
            losses.append(accuracy)
            print(f'epoch:{epoch + 1} Training Loss: {l.item()} Val Accuracy: {accuracy}')

    with open(output_file, 'wb') as f:
        pickle.dump({'CNN': CNN, 'optimizer': optimizer}, f)
    # plot and add labels
    plt.plot(epoches, np.array(scores)/max(scores), label='loss')
    plt.plot(epoches, losses, label='accuracy')
    plt.savefig(f'{output_file}.jpg')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    index = 0
    while os.path.exists('{output_file}_{index}.png'.format(output_file=output_file, index=index)):
        index += 1
    if not args.final:
        plt.savefig(f'{output_file.replace("models", "testresults")}_{index}.png')
    plt.legend()
    plt.show()
