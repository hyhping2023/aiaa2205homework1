import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from train_resnet import MyResnetCNN, ResnetBlock

def dataLoader(list_videos, mfcc_path, batch_size, feat_appendix) -> list:
    fread = open(list_videos, "r")
    feat_list = []
    video_ids = []
    for line in tqdm(fread.readlines()[:]):
        # HW00006228
        video_id = os.path.splitext(line.strip())[0]
        video_ids.append(video_id)
        mfcc_filepath = os.path.join(mfcc_path, '{}.mfcc{}'.format(video_id, feat_appendix))
        if not os.path.exists(mfcc_filepath):
            feat_list.append(torch.zeros((1009, 39)))
        else:
            feat_list.append(torch.from_numpy(np.genfromtxt(
                mfcc_filepath, delimiter=";", dtype="float")).float())

    padding_data = torch.nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    X = torch.from_numpy(padding_data.numpy()).float()
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    x_loader = []
    n = 0
    while n < X.shape[0]:
        end = min(X.shape[0], n + batch_size)
        x_loader.append(X[n:end])
        n += batch_size
    return x_loader, video_ids


def predict(model, x_iter):
    x_pred = []
    for inx, x in enumerate(x_iter):
        x = x.cuda()
        x_pred.append(model.predict(x).cpu())
        x = x.cpu()
    y_pred = torch.cat(x_pred).argmax(dim=1)
    return y_pred

def loadModel(output_file):
    model = MyResnetCNN(device='cuda')
    checkpoint = torch.load(output_file)
    model.load_state_dict(checkpoint['model'])
    optimizer = checkpoint['optimizer']
    return model, optimizer

if __name__ == '__main__':
    
    model = '../models/mfcc-100.resnet.pth'
    output_file = '../testresults/mfcc-100.resnet.csv'
    list_videos = '../labels/test_for_student.label'
    feat_appendix = '.csv'
    mfcc_path = '../mfcc/'

    # 1. read all features in one array.
    batch_size = 512
    x_loader, video_ids = dataLoader(list_videos, mfcc_path, batch_size, feat_appendix)

    # 2. load model
    CNN, optimizer = loadModel(model)

    # 3. predict
    pred_classes = predict(CNN, x_loader)
    pred_classes = pred_classes.cpu().detach().numpy()

    # 4. save for submission
    with open(output_file, "w") as f:
        f.writelines("Id,Category\n")
        for i, pred_class in enumerate(pred_classes):
            f.writelines("%s,%d\n" % (video_ids[i], pred_class))

    