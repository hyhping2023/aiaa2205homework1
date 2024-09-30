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

    # os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

    # feat_dir = 'mybof100/'
    # feat_dim = 100
    list_videos = '../labels/trainval.csv'
    feat_appendix = '.csv'
    mfcc_path = '../mfcc/'
    output_file = '../models/mfcc-100.rnn.model'
    # output_file = 'models/temp.model'

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
    for line in tqdm(fread.readlines()[1:20000]):
        path = os.path.join(mfcc_path, line.strip().split(",")[0] + ".mfcc.csv") 
        if not os.path.exists(path):
            continue
        # print(mfcc_path) 
        label_list.append(int(df_videos_label[line.strip().split(",")[0]])) 
        array = np.genfromtxt(path, delimiter=";")
        array = torch.from_numpy(array).float()
        raw_data.append(array)
    padding_data = pad_sequence(raw_data, batch_first=True)
    # shapes = [i.shape for i in raw_data]
    # import collections
    # shapes = collections.Counter(shapes)
    # print(shapes)
    # shapes = [i.shape for i in padding_data]
    # shapes = collections.Counter(shapes)
    # print(shapes)
    # print(padding_data.shape)
    # print(padding_data[0])
    # print(label_list[0])


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

    # X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    # x_t = x_t.reshape(x_t.shape[0], 1, x_t.shape[1], x_t.shape[2])

    begin_dim = X.shape[1]
    # Train
    net0 = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.MaxPool2d(kernel_size=5, stride=1, padding=1, dilation=1, ceil_mode=False),

        nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),

        nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),

        nn.AdaptiveAvgPool2d((1, 1)),
    )
    net1 = nn.Sequential(
        nn.LSTM(1, 64, 2,batch_first=True),
        # nn.ReLU(),
        # nn.MultiheadAttention(128, 8),
        
    )
    # multi = nn.MultiheadAttention(128, 8, dropout=0.1, device='cuda' if torch.cuda.is_available() else 'cpu')
    net2 = nn.Sequential(
        nn.Tanh(),
        nn.Linear(64, 10),
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    net1.apply(init_weights)
    net2.apply(init_weights)

    # if args.cont:
    #     with open(output_file, 'rb') as f:
    #         checkpoint = pickle.load(f)
    #     net1 = checkpoint['net1']
    #     net2 = checkpoint['net2']

    if torch.cuda.is_available():
        print('Using Cuda')
        net0 = net0.cuda()
        net1 = net1.cuda()
        net2 = net2.cuda()
        # multi = multi.cuda()
        X = X.cuda()
        x_t = x_t.cuda()
        y = y.cuda()
        y_t = y_t.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((*net1.parameters(), *net2.parameters()), lr=0.01, weight_decay=1e-4)
    batch_size = 256
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

            x_iter = x_iter.reshape(x_iter.shape[0], 1, x_iter.shape[1], x_iter.shape[2])

            x_temp = net0(x_iter)
            x_temp = x_temp.reshape(x_temp.shape[0], -1, 1)
            y_hat, _ = net1(x_temp)

            y_hat = net2(y_hat[:, -1, :])
            l = loss(y_hat, y_iter)
            
            l.backward()
            optimizer.step()
        if epoch % 10 == 9:
            epoches.append(epoch)
            scores.append(l.item())
            test_n = 0
            raw_data = []
            torch.cuda.empty_cache()
            net0.eval()
            net1.eval()
            net2.eval()
            with torch.no_grad():
                while test_n < x_t.shape[0]:
                    test_end = min(x_t.shape[0], test_n + batch_size)
                    test_x = x_t[test_n:test_end]
                    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2])

                    test_x = net0(test_x)
                    test_x = test_x.reshape(test_x.shape[0], -1, 1)
                    test_y_hat, _ = net1(test_x)
                    test_y_hat = net2(test_y_hat[:, -1, :])
                    
                    raw_data.append(test_y_hat)
                    test_n += batch_size
                test_y = y_t
                test_y_hat = torch.cat(raw_data, dim=0)
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
            net0.train()

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

    print(classification_report(test_y, test_y_hat, target_names=[str(i) for i in range(10)]))
    print('The best accuracy is',max(losses))

    with open(output_file, 'wb') as f:
        pickle.dump({'net1': net1, 'net2': net2}, f)