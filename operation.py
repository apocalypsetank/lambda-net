import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
import os
import math
import h5py
from operator import itemgetter



def load_hscnn():
    imgs = []
    for i in range(150):
        print i
        path = 'training_data/%d.mat' % i
        img = sio.loadmat(path)['data']
        imgs.append(img)
    return imgs

def normalize_0_to_1(mat):
    for i in range(mat.shape[0]):

        mat[i]=(mat[i]-mat[i].min())/(mat[i].max()-mat[i].min())
    return mat


def shuffle_crop(original_data,batch_size):

    index=np.random.choice(range(150),batch_size)

    new_data=[]

    for i in range(batch_size):
        h=original_data[index[i]].shape[0]
        w=original_data[index[i]].shape[1]
        x_index = np.random.randint(0, h - 256)
        y_index = np.random.randint(0, w - 256)
        img=original_data[index[i]][x_index:x_index + 256, y_index:y_index + 256, :]
        new_data.append(img)
    new_data=np.array(new_data)#(30,100,240,31)

    return new_data



