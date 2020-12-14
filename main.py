# coding=utf-8
import os
import gzip
import glob
import math
import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

from data.BraTS2020 import BraTS2020

import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm

import scipy.misc

import warnings
warnings.filterwarnings('ignore')

import models
import loss_function
from utils.utils import *

from config import config


def train(**kwargs):
	config._parse(kwargs)

	model = getattr(models, config.model)(in_data=4, out_data=4)
	criterion = getattr(loss_function, config.loss_function)()

	if t.cuda.is_available() is False:
		raise Exception("cuda is not working.")
	
	gpu_devices = [i for i in range(config.use_gpu_num)]
	model = nn.DataParallel(model)
	model = model.cuda(device=gpu_devices[0])
	criterion = criterion.cuda()

	ckpt_path = 'ckpt/' + config.model + config.description + '/'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	if config.load_model:
		model.load_state_dict(t.load(ckpt_path+config.load_model))
	model.train()

	optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
	if config.lr_decay != 1.0:
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_decay, patience=10, verbose=True)

	figure_losses = []
	figure_dices = []
	for epoch in range(config.max_epoch):
		train_dataset = BraTS2020(config)
		train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

		losses = []
		dices = []
		for images, labels in train_dataloader:
			times = images.shape[1]
			for i in range(times):
				_image = images[:, i, ...].cuda()
				_label = labels[:, i, ...].cuda()
				optimizer.zero_grad()
				predict = model(_image)
				loss = criterion(predict, _label)
				_, predict = t.max(predict, dim=1)
				loss.backward()
				optimizer.step()

				dice1 = dice(predict, _label, val=1.0)
				dice2 = dice(predict, _label, val=2.0)
				dice3 = dice(predict, _label, val=3.0)
				dice_avg = (dice1+dice2+dice3)/3
				losses.append(loss)
				dices.append(dice_avg)
		losses = sum(losses) / len(losses)
		dices = sum(dices) / len(dices)
		figure_losses.append(float(losses))
		figure_dices.append(dices)
		if config.lr_decay != 1.0:
			scheduler.step(losses)

		print('epoch {}/{}: losses: {}, dices: {}'.format(epoch, config.max_epoch, losses, dices))
	
	print('figure_losses:')
	print(figure_losses)
	print('figure_dices:')
	print(figure_dices)
	torch.save(model.state_dict(), os.path.join(ckpt_path, config.model + '.pth'))






if __name__ == '__main__':
	import fire
	fire.Fire()