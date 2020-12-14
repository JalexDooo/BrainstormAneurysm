import os
import sys
import glob
import random
import numpy as np
import nibabel as nib

import torch as t
import torch.nn as nn
from torch.utils.data import Dataset


def load_files(path):
    return glob.glob(path+'/*')

def load_nii_to_array(path):
    image = nib.load(path)
    affine = image.affine
    image = image.get_data()
    return image, affine

def make_image_label(path):
    pathes = glob.glob(path + '/*.nii.gz')
    image = []
    seg = None

    for p in pathes:
        if 'flair.nii' in p:
            flair, aff = load_nii_to_array(p)
        elif 't2.nii' in p:
            t2, aff = load_nii_to_array(p)
        elif 't1.nii' in p:
            t1, aff = load_nii_to_array(p)
        elif 't1ce.nii' in p:
            t1ce, aff = load_nii_to_array(p)
        else:
            seg, aff = load_nii_to_array(p)
    image.append(flair)
    image.append(t1)
    image.append(t1ce)
    image.append(t2)

    label = seg
    return np.asarray(image), np.asarray(label), aff

def get_box(image, margin):
    shape = image.shape
    nonindex = np.nonzero(image)  # 返回的是3个数组，分别对应三个维度的下标。

    margin = [margin] * len(shape)

    index_min = []
    index_max = []

    for i in range(len(shape)):
        index_min.append(nonindex[i].min())
        index_max.append(nonindex[i].max())

    # 扩大margin个区域
    for i in range(len(shape)):
        index_min[i] = max(index_min[i] - margin[i], 0)
        index_max[i] = min(index_max[i] + margin[i], shape[i] - 1)

    return index_min, index_max

def make_box(image, index_min, index_max, data_box):
    shape = image.shape

    for i in range(len(shape)):
        mid = (index_min[i] + index_max[i]) / 2
        index_min[i] = mid - data_box[i] / 2
        index_max[i] = mid + data_box[i] / 2

        flag = index_max[i] - shape[i]
        if flag > 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag

        flag = index_min[i]
        if flag < 0:
            index_max[i] = index_max[i] - flag
            index_min[i] = index_min[i] - flag

        if index_max[i] - index_min[i] != data_box[i]:
            index_max[i] = index_min[i] + data_box[i]

        index_max[i] = int(index_max[i])
        index_min[i] = int(index_min[i])

    return index_min, index_max

def crop_with_box(image, index_min, index_max):
    x = index_max[0] - index_min[0] - image.shape[0]
    y = index_max[1] - index_min[1] - image.shape[1]
    z = index_max[2] - index_min[2] - image.shape[2]
    img = image
    img1 = image
    img2 = image

    if x > 0:
        img = np.zeros((image.shape[0] + x, image.shape[1], image.shape[2]))
        img[x // 2:image.shape[0] + x // 2, :, :] = image[:, :, :]
        img1 = img

    if y > 0:
        img = np.zeros((img1.shape[0], img1.shape[1] + y, img1.shape[2]))
        img[:, y // 2:image.shape[1] + y // 2, :] = img1[:, :, :]
        img2 = img

    if z > 0:
        img = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2] + z))
        img[:, :, z // 2:image.shape[2] + z // 2] = img2[:, :, :]

    return img[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]

def normalization(image):
    img = image[image > 0]
    image = (image - img.mean()) / img.std()
    return image

def label_processing(label):
    lbl = (label==1)*1.0 + (label==2)*2.0 + (label==4)*3.0
    return np.array(lbl)

def random_slice(image, label, times, random_width, use=False, is_train=False):
    image_volumn = []
    label_volumn = []

    if use:
        for _ in range(times*times):
            idy = np.random.randint(0, image.shape[1] - random_width + 1)
            idz = np.random.randint(0, image.shape[2] - random_width + 1)
            image_volumn.append(image[:, idy:idy+random_width, idz:idz+random_width, :])
            label_volumn.append(label[idy:idy+random_width, idz:idz+random_width, :])
    else:
        for i in range(times):
            for j in range(times):
                idy = i * random_width
                idz = j * random_width
                image_volumn.append(image[:, idy:idy + random_width, idz:idz + random_width, :])
                if is_train:
                    label_volumn.append(label[idy:idy+random_width, idz:idz+random_width, :])


    return np.asarray(image_volumn), np.asarray(label_volumn)

def random_bias(image):
    std = image.std()
    rand_bias = random.uniform(-0.1*std/2, 0.1*std/2)
    image = (image > 0)*image + int(rand_bias)
    return image

def random_reverse(image, d1, d2, d3):
    if d1:
        image = image[::-1, :, :]
    if d2:
        image = image[:, ::-1, :]
    if d3:
        image = image[:, :, ::-1]
    
    return image


class BraTS2020(nn.Module):
    def __init__(self, config):
        self.train_path = config.train_path
        self.val_path = config.val_path
        self.is_train = config.is_train
        self.image_box = config.image_box
        self.random_width = config.random_width
        self.times = self.image_box[0] // self.random_width
        
        if self.is_train:
            self.path_list = load_files(self.train_path)
        else:
            self.path_list = load_files(self.val_path)
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, idx):
        path = self.path_list[idx]
        name = path.split('/')[-1]
        images, labels, affine = make_image_label(path)
        flair, t1, t1ce, t2 = images
        # image: (4, 240, 240, 155), label: (240, 240, 155), affine: (4, 4)

        box_min, box_max = get_box(flair, 0)
        index_min, index_max = make_box(flair, box_min, box_max, self.image_box)
        
        flair = crop_with_box(flair, index_min, index_max)
        t1 = crop_with_box(t1, index_min, index_max)
        t1ce = crop_with_box(t1ce, index_min, index_max)
        t2 = crop_with_box(t2, index_min, index_max)

        if self.is_train:
            labels = crop_with_box(labels, index_min, index_max)
            ###############
            # 随机强度偏移
            flair = random_bias(flair)
            t1 = random_bias(t1)
            t1ce = random_bias(t1ce)
            t2 = random_bias(t2)

            # 随机镜像反转
            d1 = random.choice([True, False])
            d2 = random.choice([True, False])
            d3 = random.choice([True, False])
            flair = random_reverse(flair, d1, d2, d3)
            t1 = random_reverse(t1, d1, d2, d3)
            t1ce = random_reverse(t1ce, d1, d2, d3)
            t2 = random_reverse(t2, d1, d2, d3)

            labels = random_reverse(labels, d1, d2, d3)
            ###############
            # Other approaches
            labels = label_processing(labels)

        flair = normalization(flair)
        t1 = normalization(t1)
        t1ce = normalization(t1ce)
        t2 = normalization(t2)

        images = []
        images.append(flair)
        images.append(t1)
        images.append(t1ce)
        images.append(t2)
        images = np.asarray(images)

        ###############
        # images, labels = random_slice(images, labels, self.times, self.random_width, use=False, is_train=self.is_train)
        images, labels = random_slice(images, labels, self.times, self.random_width, use=True, is_train=self.is_train)
        ###############


        images = t.from_numpy(images).float()
        labels = t.from_numpy(labels).float()

        if self.is_train:
            return images, labels
        else:
            return name, images, labels, affine, index_min, index_max

