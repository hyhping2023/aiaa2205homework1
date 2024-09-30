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
    def get_conv_block(self, in_channels, out_channels, conv_layers = 5):
        def get_conv_dict(in_channels, out_channels, num:list):
            assert len(num) == 2
            conv_layers = OrderedDict()
            for _ in range(num[0], num[1]):
                conv_layers.update(
                    {
                f'conv{_}': nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                f'bn{_}': nn.BatchNorm2d(out_channels),
                f'relu{_}': nn.ReLU(),
            }
                )
            return conv_layers
        conv_dict = OrderedDict()
        if in_channels == out_channels:
            conv_dict.update(get_conv_dict(in_channels, out_channels, [0,conv_layers]))
            return conv_dict
        else:
            conv_dict.update(get_conv_dict(in_channels, out_channels, [0,1]))
            conv_dict.update(get_conv_dict(out_channels, out_channels, [1,conv_layers]))
            return conv_dict
    def __init__(self, in_channels, out_channels, conv_layers:list, layers = 5, device = 'cuda'):
        assert layers == len(conv_layers)
        self.layers = layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        blocks = OrderedDict()
        for _ in range(layers):
            blocks.update(self.get_conv_block(in_channels, out_channels, conv_layers[_]))
        self.net = nn.Sequential(blocks)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.net.apply(self.init_weights)
        self.shortcut.apply(self.init_weights)
        self.device = device
        self.net = self.net.to(device)
        self.shortcut = self.shortcut.to(device)
    def forward(self, x):
        x_d = self.net(x)
        x_s = self.shortcut(x)
        x_f = x_d + x_s
        return x_f
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return (*self.net.parameters(), *self.shortcut.parameters())
    def changeMode(self, mode):
        if mode == 'train':
            self.net.train()
            self.shortcut.train()
        elif mode == 'eval':
            self.net.eval()
            self.shortcut.eval()
        

class MyResnetCNN():
    def __init__(self, device='cuda'):
        self.device = device
        self.net1 = ResnetBlock(1, 16, [2, 2, 2, 2, 2], layers=5, device = device)
        self.net2 = ResnetBlock(16, 32, [2, 2, 2, 2, 2, 2, 2], layers=7, device = device)
        self.net3 = ResnetBlock(32, 64, [2, 2, 2, 2, 2, 2, 2, 2, 2], layers=9, device = device)
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ).to(device)

        
    def forward(self, x):
        x1 = self.net1.forward(x)
        x2 = self.net2.forward(x1)
        x3 = self.net3.forward(x2)
        result = self.output(x3)
        return result
    def changeMode(self, mode):
        if mode == 'train':
            self.net1.changeMode('train')
            self.net2.changeMode('train')
            self.net3.changeMode('train')
            self.output.train()
        elif mode == 'eval':
            self.net1.changeMode('eval')
            self.net2.changeMode('eval')
            self.net3.changeMode('eval')
            self.output.eval()
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return (*self.net1(), *self.net2(), *self.net3(), *self.output.parameters())
    def predict(self, x):
        self.changeMode('eval')
        with torch.no_grad():
            return self.forward(x) 
        self.changeMode('train')
        
def dataLoader(list_videos, mfcc_path, batch_size, shrink) -> list:
    '''
    The return values are X, x_t, y, y_t
    X stands for train X
    x_t stands for test X
    y stands for train y
    y_t stands for test y
    They are all list.
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

    Y = np.array(label_list)
    print(padding_data.shape, Y.shape)

    X, x_t, y, y_t = train_test_split(padding_data, Y, test_size=0.2, random_state=18877)
    if args.final:
        X = padding_data
        y = Y
    print(X.shape, y.shape, x_t.shape, y_t.shape)

    y = torch.from_numpy(y).long()
    y_t = torch.from_numpy(y_t).long()

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
        net1 = checkpoint['net1']
        net2 = checkpoint['net2']
        return net1, net2
    else:
        return MyResnetCNN(device=device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cont', action='store_true', default=False)
    parser.add_argument('--final', action='store_true', default=False)
    args = parser.parse_args()

    list_videos = '../labels/trainval.csv'
    feat_appendix = '.csv'
    mfcc_path = '../mfcc/'
    output_file = '../models/mfcc-100.conv_2.model'
    device = 'cuda'
    batch_size = 256

    X, x_t, y, y_t = dataLoader(list_videos, mfcc_path, batch_size, shrink = 10000)
    CNN = loadModel(args.cont, output_file, device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNN(), lr=0.01, weight_decay=1e-4)
    
    epoches = []
    scores = []
    losses = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(200):
        for x_iter, y_iter in zip(X, y):
            x_iter = x_iter.cuda()
            y_iter = y_iter.cuda()
            y_hat = CNN.forward(x_iter)
            l = loss(y_hat, y_iter)
            l.backward()
            optimizer.step()
            x_iter = x_iter.cpu()
            y_iter = y_iter.cpu()
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

    # plot and add labels
    plt.plot(epoches, np.array(scores)/max(scores), label='loss')
    plt.plot(epoches, losses, label='accuracy')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    index = 0
    while os.path.exists('{output_file}_{index}.png'.format(output_file=output_file, index=index)):
        index += 1
    if not args.final:
        plt.savefig(f'{output_file.replace("models", "testresults")}_{index}.png')
    plt.legend()
    plt.show()
