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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cont', action='store_true', default=False)
    parser.add_argument('--final', action='store_true', default=False)
    args = parser.parse_args()

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

    begin_dim = X.shape[1]
    # Train
    net1 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
        # nn.ReLU(),
        nn.BatchNorm2d(8, track_running_stats=True),
        nn.AvgPool2d(kernel_size=5, stride=3, padding=0, ceil_mode=False),
        nn.ReLU(),

        nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        # nn.ReLU(),
        nn.BatchNorm2d(16, track_running_stats=True),
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),
        nn.ReLU(),

        nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        # nn.ReLU(),
        nn.BatchNorm2d(32, track_running_stats=True),
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d((1, 1)),
    )
    net2 = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        # nn.Dropout(0.99),
        nn.Linear(64, 10),
    )

    output_file = '../models/mfcc-100.conv_2.model'
    # output_file = 'models/temp.model'

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    net1.apply(init_weights)
    net2.apply(init_weights)

    if args.cont:
        with open(output_file, 'rb') as f:
            checkpoint = pickle.load(f)
        net1 = checkpoint['net1']
        net2 = checkpoint['net2']

    if torch.cuda.is_available():
        print('Using Cuda')
        net1 = net1.cuda()
        net2 = net2.cuda()
        X = X.cuda()
        x_t = x_t.cuda()
        y = y.cuda()
        y_t = y_t.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((*net1.parameters(), *net2.parameters()), lr=0.01, weight_decay=1e-4)
    batch_size = 4096
    epoches = []
    scores = []
    losses = []
    for epoch in range(200):
        n = 0
        while n < X.shape[0]:
            optimizer.zero_grad()
            end = min(X.shape[0], n + batch_size)
            x_iter = X[n:end]
            y_iter = y[n:end]
            n += batch_size
            # print(x_iter.shape, y_iter.shape)
            # print(x_iter, y_iter)

            x_temp = net1(x_iter)
            x_temp = x_temp.reshape(x_temp.shape[0], -1)
            y_hat = net2(x_temp)
            # y_hat = torch.nn.functional.softmax(y_hat, dim=0)
            # print(y_hat, y_iter)
            l = loss(y_hat, y_iter)
            l.backward()
            optimizer.step()
        if epoch % 10 == 9:
            net1.eval()
            net2.eval()
            with torch.no_grad():
                epoches.append(epoch)
                scores.append(l.item())
                test_y_hat = net1(x_t)
                test_y_hat = net2(test_y_hat.reshape(test_y_hat.shape[0], -1))
                test_y = y_t
                test_y_hat = test_y_hat.cpu().detach().numpy()
                test_y = test_y.cpu().detach().numpy()
                test_y_hat = test_y_hat.argmax(axis=1)
                accuracy = (test_y == test_y_hat).mean()
            losses.append(accuracy)
            print(f'epoch:{epoch + 1} Training Loss: {l.item()} Val Accuracy: {accuracy}')
            with open(output_file, 'wb') as f:
                pickle.dump({'net1': net1, 'net2': net2}, f)
            net1.train()
            net2.train()

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


    # Predict on test set
    net1.eval()
    net2.eval()
    with torch.no_grad():
        test_y_hat = net1(x_t)
        test_y_hat = net2(test_y_hat.reshape(test_y_hat.shape[0], -1))
        test_y = y_t

        # to cpu
        test_y_hat = test_y_hat.cpu()
        test_y = test_y.cpu()

        # to numpy
        test_y_hat = test_y_hat.detach().numpy()
        test_y = test_y.detach().numpy()

        print(classification_report(test_y, test_y_hat.argmax(axis=1), target_names=[str(i) for i in range(10)]))
        print(max(losses))

    with open(output_file, 'wb') as f:
        pickle.dump({'net1': net1, 'net2': net2}, f)