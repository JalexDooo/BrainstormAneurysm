import os
import sys
import glob
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.misc

from .datautils import *

module = ['flair', 't1', 't1ce', 't2']


class AneuMulti(Dataset):
    def __init__(self, train_path, is_train=True, val_path=''):
        self.train_path = train_path
        self.is_train = is_train
        self.val_path = val_path
        if self.is_train:
            self.path_list = load_aneu_image_path(self.train_path)
        else:
            self.path_list = load_aneu_image_path(self.val_path)
        self.index_box = [128, 448, 448]
        self.data_dim = 64

    def __len__(self):
        return len(self.path_list)

    def second_pre(self, image, label):
        """
            随机切片。
            output:[9, 4, 16, 192, 192]
        """
        times = int(image.shape[1] / self.data_dim)

        image_volumn = []
        label_volumn = []

        for i in range(times):
            if self.is_train:
                st = np.random.randint(0, image.shape[1] - self.data_dim + 1)
            else:
                st = i * self.data_dim

            image_volumn.append(image[:, st:st + self.data_dim, :, :])
            if self.is_train:
                label_volumn.append(label[st:st + self.data_dim, :, :])

        image_volumn = np.asarray(image_volumn)
        label_volumn = np.asarray(label_volumn)

        return image_volumn, label_volumn

    def __getitem__(self, item):
        # print('len: ', len(self.path_list))
        # print('item: ', item)
        # print(self.path_list[item])

        path = glob.glob(self.path_list[item] + '/*')
        image = None
        label = None
        # print(path)
        if 'Untitled.nii.gz' in path[0]:
            image_path = path[1]
            label_path = path[0]
        else:
            image_path = path[0]
            label_path = path[1]

        # print('path: ', image_path)
        image = load_nii_to_array(image_path)
        if self.is_train:
            label = load_nii_to_array(label_path)

        head_image = nib.load(image_path)
        affine = head_image.affine

        img_shape = image.shape
        # print(image_path, ' --> ', img_shape)

        index_min, index_max = get_box(image, margin=0)

        index_min, index_max = make_box(image, index_min, index_max, self.index_box)

        image = crop_with_box(image, index_min, index_max)
        if self.is_train:
            label = crop_with_box(label, index_min, index_max)

        image = normalization(image)

        img = []
        img.append(image)
        image = np.asarray(img)
        if self.is_train:
            label = get_NCR_NET_label(label)
            label = np.asarray(label)
        # print('image.shape: ', image.shape)

        image, label = self.second_pre(image, label)
        # print('imagere.shape: ', image.shape)
        name_list = image_path.split('/')
        name = name_list[-3] + '_' + name_list[-2]
        if self.is_train:
            return torch.from_numpy(image).float(), torch.from_numpy(label).float(), name
        else:
            return torch.from_numpy(image).float(), torch.from_numpy(
                label).float(), img_shape, index_min, index_max, affine, name

