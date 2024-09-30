import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def predict(net1, net2, X):
    with torch.no_grad():
        x_temp = net1(X)
        x_tmep = x_temp.reshape(x_temp.shape[0], -1)
        y_hat = net2(x_tmep)
        prediction = y_hat.argmax(axis=1) 
    return prediction
if __name__ == '__main__':
    
    model = '../models/mfcc-100.conv_1_final.model'
    output_file = '../mfcc-100.pytorch_try.csv'
    list_videos = '../labels/test_for_student.label'
    feat_appendix = '.csv'
    mfcc_path = '../mfcc/'

    with open(model, 'rb') as f:
        checkpoint = pickle.load(f)
    net1 = checkpoint['net1']
    net2 = checkpoint['net2']

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

    if torch.cuda.is_available():
        X = X.cuda()
        net1 = net1.cuda()
        net2 = net2.cuda()

    net1.eval()
    net2.eval()

    # 3. predict
    batch_size = 1024
    pred_classes = []
    start = 0
    while start < X.shape[0]:
        end = min(X.shape[0], start + batch_size)
        x_iter = X[start:end]

        print(x_iter.shape)
        
        pred_classes.extend(predict(net1, net2, x_iter))
        start += batch_size
    # print(torch.cat((X[:1140], X[1140:]), dim=0) == X)
    pred_classes = torch.tensor(pred_classes)

    pred_classes = pred_classes.cpu().detach().numpy()

    # 4. save for submission
    with open(output_file, "w") as f:
        f.writelines("Id,Category\n")
        for i, pred_class in enumerate(pred_classes):
            f.writelines("%s,%d\n" % (video_ids[i], pred_class))

    for i, layer in enumerate(net1):
        print(i, layer)

    for i, layer in enumerate(net2):
        print(i, layer)


    