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
import argparse

class MyCNN():
    def __init__(self, device='cuda'):
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.net1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.net3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(8),
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
        )
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        if device == 'cuda' and torch.cuda.is_available():
            self.net1 = self.net1.cuda()
            self.net2 = self.net2.cuda()
            self.net3 = self.net3.cuda()
            self.shortcut1 = self.shortcut1.cuda()
            self.shortcut2 = self.shortcut2.cuda()
            self.shortcut3 = self.shortcut3.cuda()
            self.output = self.output.cuda()
        self.net1.apply(init_weights)
        self.net2.apply(init_weights)
        self.net3.apply(init_weights)
        self.output.apply(init_weights)
        self.shortcut1.apply(init_weights)
        self.shortcut2.apply(init_weights)
        self.shortcut3.apply(init_weights)
    def forward(self, x):
        x1 = self.net1(x)
        x1_1 = x1 + self.shortcut1(x)
        x2 = self.net2(x1_1)
        x2_1 = x2 + self.shortcut2(x1_1)
        x3 = self.net3(x2_1)
        x3_1 = x3 + self.shortcut3(x2_1)
        result = self.output(x3_1)
        return result
    def predict(self, x):
        self.net1.eval()
        self.net2.eval()
        self.net3.eval()
        self.output.eval()
        self.shortcut1.eval()
        self.shortcut2.eval()
        self.shortcut3.eval()
        with torch.no_grad():
            x = self.forward(x)
        self.net1.train()
        self.net2.train()
        self.net3.train()
        self.output.train()
        self.shortcut1.train()
        self.shortcut2.train()
        self.shortcut3.train()
        return x
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cont', action='store_true', default=False)
    parser.add_argument('--final', action='store_true', default=False)
    args = parser.parse_args()

    feat_dir = '../mybof100/'
    feat_dim = 100
    list_videos = '../labels/trainval.csv'
    feat_appendix = '.csv'
    mfcc_path = '../mfcc/'

    # 1. read all features in one array.
    fread = open(list_videos, "r")
    # labels are [0-9]
    label_list = []
    # load video names and events in dict
    df_videos_label = {}
    for line in open(list_videos).readlines()[1:]:
        video_id, category = line.strip().split(",")
        df_videos_label[video_id] = category

    raw_data = []
    for line in tqdm(fread.readlines()[1:10000]):
        path = os.path.join(mfcc_path, line.strip().split(",")[0] + ".mfcc.csv") 
        if not os.path.exists(path):
            continue
        # print(mfcc_path) 
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

    # Bring numpy to torch tensor
    # X = torch.from_numpy(X).float()
    # x_t = torch.from_numpy(x_t).float()
    y = torch.from_numpy(y).long()
    y_t = torch.from_numpy(y_t).long()

    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    x_t = x_t.reshape(x_t.shape[0], 1, x_t.shape[1], x_t.shape[2])

    # Train
    # if torch.cuda.is_available():
    #     X = X.cuda()
    #     x_t = x_t.cuda()
    #     y = y.cuda()
    #     y_t = y_t.cuda()

    output_file = '../models/mfcc-100.conv_2.model'
    # output_file = 'models/temp.model'
    CNN = MyCNN()

    if args.cont:
        with open(output_file, 'rb') as f:
            checkpoint = pickle.load(f)
        net1 = checkpoint['net1']
        net2 = checkpoint['net2']

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((*CNN.net1.parameters(), *CNN.net2.parameters(), *CNN.net3.parameters(), *CNN.output.parameters(), *CNN.shortcut1.parameters(), *CNN.shortcut2.parameters(), *CNN.shortcut3.parameters()), 
                                 lr=0.01, weight_decay=1e-4)
    batch_size = 128
    epoches = []
    scores = []
    losses = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(200):
        n = 0
        while n < X.shape[0]:
            end = min(X.shape[0], n + batch_size)
            x_iter = X[n:end]
            y_iter = y[n:end]
            n += batch_size
            # print(x_iter.shape, y_iter.shape)
            # print(x_iter, y_iter)
            x_iter = x_iter.cuda()
            y_iter = y_iter.cuda()

            y_hat = CNN.forward(x_iter)
            # y_hat = torch.nn.functional.softmax(y_hat, dim=0)
            # print(y_hat, y_iter)
            l = loss(y_hat, y_iter)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            x_iter = x_iter.cpu()
            y_iter = y_iter.cpu()
        print('Train finished')
        if epoch % 10 == 9:
            epoches.append(epoch)
            scores.append(l.item())
            test_y_hat = []
            n_t = 0
            while n_t < x_t.shape[0]:
                end = min(x_t.shape[0], n + batch_size)
                x_t_iter = x_t[n_t:end]
                x_t_iter = x_t_iter.cuda()
                n_t += batch_size
                test_y_hat.append(CNN.predict(x_t_iter).cpu())
                x_t_iter = x_t_iter.cpu()

            test_y = y_t
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
