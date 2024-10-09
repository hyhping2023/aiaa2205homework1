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
import math

class PostionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device = 'cuda'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout).to(device)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MyTransformer(nn.Module):
    def __init__(self, d_model = 39, hid_dim = 128, device = 'cuda'):
        super(MyTransformer, self).__init__()
        self.pre_encoder = nn.Sequential(
            nn.Linear(d_model, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        ).to(device)
        self.encoder = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=8, dim_feedforward=1024, 
                                                   device=device, batch_first=True)
        self.pos_encoder = PostionalEncoding(d_model=hid_dim, device=device)
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 10)
        ).to(device)
        self.init_weights()
        self.d_model = d_model
        self.hid_dim = hid_dim

    def init_weights(self):
        for layer in self.pre_encoder:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
        for layer in self.decoder:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.zero_()
    def changemode(self, mode):
        if mode == 'train':
            self.pre_encoder.train()
            self.encoder.train()
            self.decoder.train()
        elif mode == 'eval':
            self.pre_encoder.eval()
            self.encoder.eval()
            self.decoder.eval()
        else:
            raise Exception('mode should be train or eval')
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return *self.pre_encoder.parameters(), *self.encoder.parameters(), *self.decoder.parameters()
    def forward(self, x_o):
        self.changemode('train')
        # print(x_o.shape)
        X = self.pre_encoder(x_o)
        X = self.pos_encoder(X)
        X = self.encoder(X)
        X = self.decoder(X)
        # print(X.shape, X[-1, :].shape)
        return X[-1, -1, :]
    def predict(self, x):
        self.changemode('eval')
        with torch.no_grad():
            return self.forward(x)



        
def dataLoader(list_videos, mfcc_path, shrink, testmode = False, ratio = 0.2) -> list:
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
    # padding_data = pad_sequence(raw_data, batch_first=True)

    Y = torch.from_numpy(np.array(label_list)).long()
    # print(Y.shape)

    X, x_t, y, y_t = train_test_split(raw_data, Y, test_size=ratio, random_state=18877)
    if testmode:
        X = torch.cat([X, x_t])
        y = torch.cat([y, y_t])
    
    return X, x_t, y, y_t

def loadModel(load:bool, output_file, device):
    if load:
        with open(output_file, 'rb') as f:
            checkpoint = pickle.load(f)
            model = checkpoint['CNN']
            optimizer = checkpoint['optimizer']
            return model, optimizer
    else:
        model = MyTransformer(d_model=39, device=device)
        return model, torch.optim.Adam(model(), lr=0.01, weight_decay=1e-4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cont', action='store_true', default=False)
    parser.add_argument('--final', action='store_true', default=False)
    args = parser.parse_args()

    list_videos = '../labels/trainval.csv'
    feat_appendix = '.csv'
    mfcc_path = '../mfcc/'
    output_file = '../models/mfcc-100.transformer.model'
    device = 'cuda'
    batch_size = 1024

    X, x_t, y, y_t = dataLoader(list_videos, mfcc_path, shrink = 10000, testmode = args.final, ratio = 0.2)
    Transformer, optimizer = loadModel(args.cont, output_file, device)

    loss = nn.CrossEntropyLoss()
    print(Transformer.pre_encoder, Transformer.encoder, Transformer.decoder)
    
    epoches = []
    scores = []
    losses = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(200):
        for inx, (x_iter, y_iter) in enumerate(zip(X, y)):
            optimizer.zero_grad()
            x_iter = x_iter.reshape(1, x_iter.shape[0], x_iter.shape[1]).cuda()
            y_iter = y_iter.cuda()
            y_hat = Transformer.forward(x_iter)
            # print(y_hat.shape, y_iter.shape)
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
                x_t_iter = x_t_iter.reshape(1, x_t_iter.shape[0], x_t_iter.shape[1]).cuda()
                test_y_hat.append(Transformer.predict(x_t_iter).cpu())
                x_t_iter = x_t_iter.cpu()
            test_y = y_t
            test_y_hat = torch.cat([_.argmax(axis=0).reshape(1) for _ in test_y_hat], dim=0)
            test_y_hat = test_y_hat.cpu().detach().numpy()
            test_y = test_y.cpu().detach().numpy()
            # test_y_hat = test_y_hat.argmax(axis=1)
            accuracy = (test_y == test_y_hat).mean()
            losses.append(accuracy)
            print(f'epoch:{epoch + 1} Training Loss: {l.item()} Val Accuracy: {accuracy}')

    with open(output_file, 'wb') as f:
        pickle.dump({'Transformer': Transformer, 'optimizer': optimizer}, f)
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
