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
from config import config

from .datautils import *

module = ['flair', 't1', 't1ce', 't2']


class BraTS_Random(Dataset):
    """
    Brats2019数据集。
    """
    def __init__(self, train_root_path, val_root_path, is_train=True, task='WT', predict=False):
        self.train_root_path = train_root_path
        self.val_root_path = val_root_path
        self.is_train = is_train
        self.task = task
        self.predict = predict
        self.data_box = [144, 192, 192]# 240, 240, 155
        self.data_dim = 16
        self.random_width = 64

        self.path_list = load_hgg_lgg_files(self.train_root_path)
        # print(self.path_list)
        if not self.is_train:
            self.path_list = load_val_file(self.val_root_path)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, item):

        path = self.path_list[item]
        if self.predict:
            print(path)
        if self.is_train:
            image, label, box_min, box_max = self.first_pre(path)
            # image, label = self.second_pre(image, label) # 切片
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()

            return image, label
        else:
            image, label, box_min, box_max = self.first_pre(path)
            image = torch.from_numpy(image).float()
            name = path.split('/')[-1]

            return image, name, box_min, box_max

    def first_pre(self, path):
        """
        从路径加载，第一步处理。
        :param path: 单个大脑路径。
        :return: 图像，标签。
        """
        image = []
        label = []
        image_t, label_t = make_image_label(path)
        # print(image_t[0].shape)
        flair, t1, t1ce, t2 = image_t
        seg = label_t

        # 按照flair确定裁剪区域
        box_min, box_max = get_box(flair, 0)
        index_min, index_max = make_box(flair, box_min, box_max, self.data_box)

        # 裁剪
        flair = crop_with_box(flair, index_min, index_max)
        t1 = crop_with_box(t1, index_min, index_max)
        t1ce = crop_with_box(t1ce, index_min, index_max)
        t2 = crop_with_box(t2, index_min, index_max)
        if self.is_train:
            seg = crop_with_box(seg, index_min, index_max)

        # 标准化
        flair = normalization(flair)
        t1 = normalization(t1)
        t1ce = normalization(t1ce)
        t2 = normalization(t2)

        # label1 = get_ncr_labels(seg)
        # label2 = get_ed_labels(seg)
        # label3 = get_ot_labels(seg)
        # label4 = get_tumor_core_labels(seg)
        if self.task == 'WT' and seg:
            label = get_WT_labels(seg)
        elif self.task == 'TC' and seg:
            label = get_TC_labels(seg)
        elif self.task == 'ET' and seg:
            label = get_ET_labels(seg)
        elif self.task == 'NCR' and seg:
            label = get_NCR_NET_label(seg)
        elif self.is_train:
            label = seg * 1.0

        # 想法：阅读别人的程序，发现也可以多方向扫描MRI。
        # ...

        image.append(flair)
        image.append(t1)
        image.append(t1ce)
        image.append(t2)

        # label.append(label1)
        # label.append(label2)
        # label.append(label3)
        # label.append(label4)
        image = np.asarray(image)
        # img = rotate_image(image)
        label = np.asarray(label)
        image, label = self.random_slice(image, label)

        return image, label, index_min, index_max

    def random_slice(self, image, label):
        image_volumn = []
        label_volumn = []

        if self.is_train:
            for _ in range(9):
                idy = np.random.randint(0, image.shape[2] - self.random_width + 1)
                idz = np.random.randint(0, image.shape[3] - self.random_width + 1)
                image_volumn.append(image[:, :, idy:idy+self.random_width, idz:idz+self.random_width])
                label_volumn.append(label[:, idy:idy+self.random_width, idz:idz+self.random_width])
        else:
            for i in range(3):
                for j in range(3):
                    idy = i * self.random_width
                    idz = j * self.random_width
                    image_volumn.append(image[:, :, idy:idy + self.random_width, idz:idz + self.random_width])

        return np.asarray(image_volumn), np.asarray(label_volumn)

    def second_pre(self, image, label):
        """
            随机切片。
            output:[9, 4, 16, 192, 192]
        """
        times = int(image.shape[1] / self.data_dim) # 12 * 2

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


class BraTS2020(Dataset):
    """Redesigned class of BraTS2020 dataset processing. Aim: .nii.gz -> pytorch tensor.

    Data Augumentation Methods:
        1. Random bias.
        2. Random reverse.
        3. Random slicing. Batchsize sets 1. Using group normalization in model.

    """
    def __init__(self, config):
        self.train_path = config.dataset_train_path
        self.val_path = config.dataset_val_path
        self.is_train = config.is_train
        self.image_box = config.model_input_shape
        self.random_width = config.dataset_random_width
        self.times = self.image_box[-1] // self.random_width

        if self.is_train:
            self.path_list = load_hgg_lgg_files(self.train_path)
        else:
            self.path_list = load_hgg_lgg_files(self.val_path)

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, idx):
        path = self.path_list[idx]
        # print('path: {}'.format(path))

        image, label, onehot_label, box_min, box_max = self._read_image(path)
        if self.is_train:
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()
            onehot_label = torch.from_numpy(onehot_label).float()
            return image, label, onehot_label
        else:
            image = torch.from_numpy(image).float()
            name = path.split('/')[-1]
            return image, name, box_min, box_max
    
    def _random_slice(self, image, label, onehot_label):
        image_volumn = []
        label_volumn = []
        onehot_label_volumn = []

        if self.is_train:
            for _ in range(self.times*2):
                idy = np.random.randint(0, image.shape[2] - self.random_width + 1)
                idz = np.random.randint(0, image.shape[3] - self.random_width + 1)
                image_volumn.append(image[:, :, idy:idy+self.random_width, idz:idz+self.random_width])
                label_volumn.append(label[:, :, idy:idy+self.random_width, idz:idz+self.random_width])
                onehot_label_volumn.append(onehot_label[:, idy:idy+self.random_width, idz:idz+self.random_width])
        else:
            for i in range(self.times):
                for j in range(self.times):
                    idy = i * self.random_width
                    idz = j * self.random_width
                    image_volumn.append(image[:, :, idy:idy + self.random_width, idz:idz + self.random_width])

        return np.asarray(image_volumn), np.asarray(label_volumn), np.asarray(onehot_label_volumn)

    def _read_image(self, path):
        image = []
        label = []
        onehot_label = []
        image_t, label_t = make_image_label(path)
        flair, t1, t1ce, t2 = image_t
        seg = label_t
        onehot_label = label_t

        # 按照flair确定裁剪区域
        box_min, box_max = get_box(flair, 0)
        index_min, index_max = make_box(flair, box_min, box_max, self.image_box)

        # 裁剪
        flair = crop_with_box(flair, index_min, index_max)
        t1 = crop_with_box(t1, index_min, index_max)
        t1ce = crop_with_box(t1ce, index_min, index_max)
        t2 = crop_with_box(t2, index_min, index_max)
        if self.is_train:
            seg = crop_with_box(seg, index_min, index_max)
            onehot_label = crop_with_box(onehot_label, index_min, index_max)
            # 随机强度偏移
            flair = random_bias(flair)
            t1 = random_bias(t1)
            t1c1 = random_bias(t1ce)
            t2 = random_bias(t2)

            # 随机镜像反转
            d1 = random.choice([True, False])
            d2 = random.choice([True, False])
            d3 = random.choice([True, False])
            flair = random_reverse(flair, d1, d2, d3)
            t1 = random_reverse(t1, d1, d2, d3)
            t1ce = random_reverse(t1ce, d1, d2, d3)
            t2 = random_reverse(t2, d1, d2, d3)
            seg = random_reverse(seg, d1, d2, d3)
            onehot_label = random_reverse(onehot_label, d1, d2, d3)

            label = label_processing(seg)

            # plt.imshow(flair[70, :, :])
            # plt.show()
            # raise Exception("my break")

        # 标准化
        flair = normalization(flair)
        t1 = normalization(t1)
        t1ce = normalization(t1ce)
        t2 = normalization(t2)

        image.append(flair)
        image.append(t1)
        image.append(t1ce)
        image.append(t2)

        image = np.asarray(image)
        label = np.asarray(label)
        onehot_label = np.asarray(onehot_label)
        onehot_label = get__labels(onehot_label)
        # print('dataset-> image.shape: {}, label.shape: {}'.format(image.shape, label.shape))
        image, label, onehot_label = self._random_slice(image, label, onehot_label)
        # print('dataset random slice-> image.shape: {}, label.shape: {}'.format(image.shape, label.shape))

        return image, label, onehot_label, index_min, index_max
