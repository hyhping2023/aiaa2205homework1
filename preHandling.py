import glob
import os
import numpy as np

def readCsv(path, new_path):
    files = glob.glob(path)
    for file in files:
        array = np.genfromtxt(file, delimiter=";")
        # 对array进行softmax操作
        array = np.exp(array) / np.sum(np.exp(array)) * 1e75
        print(os.path.join(new_path, os.path.basename(file)))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        np.savetxt(os.path.join(new_path, os.path.basename(file)), array, delimiter=";")



if __name__ == '__main__':
    readCsv('/home/benchmark0/hyh/pytorch/mfcc/*', '/home/benchmark0/hyh/pytorch/mymfcc2')