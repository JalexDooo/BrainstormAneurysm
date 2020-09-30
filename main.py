# coding=utf-8
import os
import gzip
import glob
import math
import numpy as np
import pandas as pd
import nibabel as nib

import cv2
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

from data.dataset import *

import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

import scipy.misc

import warnings
warnings.filterwarnings('ignore')

import models
from utils.utils import *

from config import opt


def data_test():
	# data = BraTS_Random('/Volumes/309实验室专属/孙近东/数据集/MICCAI_BraTS2020_TrainingData', opt.val_root_path, is_train=True, task=opt.task)
	# image, label = data[0]
	lr = opt.lr
	for _ in range(100):
		lr *= opt.lr_decay
		print('%.6f' % lr)


def multi_train_random(**kwargs):

	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return
	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))
	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	lr = opt.lr
	print('criterion and optimizer is finished.')
	fig_loss = []
	fig_dice = []

	for kkepoch in range(opt.random_epoch):
		print('----------------------kkepoch %d--------------------' % (kkepoch))

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
		lr *= opt.lr_decay

		# pytorch数据处理
		train_data = BraTS_Random(opt.train_root_path, opt.val_root_path, is_train=True, task=opt.task)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
		print('train and test dataloader load finished.')

		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label) in enumerate(train_dataloader):
			# img = image.cuda()
			# lbl = label.cuda()
			for i in range(9):
				img_ = image[:, i, :, :, :, :]
				lbl_ = label[:, i, :, :, :]
				img_ = img_.cuda()
				lbl_ = lbl_.cuda()

				optimizer.zero_grad()
				predicts = model(img_)
				# print('predicts.shape(): ', predicts.shape)
				# print('lbl.shape(): ', lbl_.shape)
				loss = criterion(predicts, lbl_.long())
				train_loss.append(float(loss))
				loss.backward()
				optimizer.step()

				value, tmp = t.max(predicts, dim=1)

				d = dice(tmp, lbl_.long())
				# print('d: ', d)
				train_dice.append(d)

		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		fig_loss.append(sum(train_loss) / (len(train_loss) * 1.0))
		fig_dice.append(sum(train_dice) / (len(train_dice) * 1.0))
		torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, kkepoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))
	print(fig_loss)
	print(fig_dice)


def multi_val_random(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)()
	print('model is : ', opt.model)
	save_dir = 'ckpt_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')
	# output path
	path = opt.predict_nibable_path
	if not os.path.exists(path):
		os.mkdir(path)

	# pytorch数据处理
	val_data = BraTS_Random(opt.train_root_path, opt.val_root_path, is_train=False, task=opt.task)
	# test_data = BraTS2017(opt.train_root_path, opt.val_root_path, is_train=False)
	print('train_data and test_data load finished.')

	val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
	# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
	print('train and test dataloader load finished.')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
	best_dice = -1
	best_epoch = -1
	print('criterion and optimizer is finished.')
	print(model.eval())

	predictss = []
	predicts_names = []

	save_nii_head_path = opt.val_root_path + '/BraTS20_Testing_001/BraTS20_Testing_001_flair.nii.gz'
	# save_nii_head_path = opt.val_root_path + '/Brats18_CBICA_AAM_1/Brats18_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))
		train_loss = []
		train_dice = []
		for image, name, box_min, box_max in val_dataloader:
			img = image.cuda()
			out_predict = []

			for k in range(9):
				img_ = img[:, k, :, :, :]
				predicts = model(img_)
				value, tmp = t.max(predicts, dim=1)
				out_predict.append(tmp.int())
			predict0 = t.cat((out_predict[0], out_predict[1]), dim=-1)
			predict0 = t.cat((predict0, out_predict[2]), dim=-1)

			predict1 = t.cat((out_predict[3], out_predict[4]), dim=-1)
			predict1 = t.cat((predict1, out_predict[5]), dim=-1)

			predict2 = t.cat((out_predict[6], out_predict[7]), dim=-1)
			predict2 = t.cat((predict2, out_predict[8]), dim=-1)

			predict = t.cat((predict0, predict1), dim=-2)
			predict = t.cat((predict, predict2), dim=-2)

			predict = predict.data.cpu().numpy()

			pp = np.zeros((155, 240, 240))
			x, y, z = box_min
			pp[x:x + 144, y:y + 192, z:z + 192] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predict = out_precessing(predict)
			predictss.append(predict)
			predicts_names.append(name[0])

	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)
	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		nib.save(output, path + predicts_names[i] + '.nii.gz')


# def tem():
# 	#定义两个数组
# 	Loss_list = []
# 	Accuracy_list = []
#
# 	Loss_list.append(train_loss / (len(train_dataset)))
# 	Accuracy_list.append(100 * train_acc / (len(train_dataset)))
#
# 	#我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
# 	x1 = range(0, 200)
# 	x2 = range(0, 200)
# 	y1 = Accuracy_list
# 	y2 = Loss_list
# 	plt.subplot(2, 1, 1)
# 	plt.plot(x1, y1, 'o-')
# 	plt.title('Test accuracy vs. epoches')
# 	plt.ylabel('Test accuracy')
# 	plt.subplot(2, 1, 2)
# 	plt.plot(x2, y2, '.-')
# 	plt.xlabel('Test loss vs. epoches')
# 	plt.ylabel('Test loss')
# 	plt.show()
# 	plt.savefig("accuracy_loss.jpg")

def test():
	lr = opt.lr
	for _ in range(100):
		print(lr)
		lr *= opt.lr_decay


def test_():
	print('hello world')

def gan_test():
	from torchvision import datasets
	from torchvision import transforms
	import models.gan as gan

	img_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])
	mnist = datasets.MNIST(
		root='./MNIST/', train=True, transform=img_transform, download=False
	)
	
	dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=128, shuffle=True)

	print('mnist[0].shape: ', mnist)
	discriminator = gan.DiscriminatorGAN()
	generator = gan.GenerativeGAN()

	criterion = nn.BCELoss()
	d_optim = optim.Adam(discriminator.parameters(), lr=0.0003)
	g_optim = optim.Adam(generator.parameters(), lr=0.0003)
	num_epoch = 10
	for epoch in range(num_epoch):
		for i, (img, _) in enumerate(dataloader):
			print('img.shape: ', img.shape)
			num_img = img.size(0)
			print('num_img: ', num_img)
			# train discriminator
			img = img.view(num_img, -1)
			real_img = Variable(img)
			real_label = Variable(t.ones(num_img))
			fake_label = Variable(t.zeros(num_img))

			real_out = discriminator(real_img)
			d_loss_real = criterion(real_out, real_label)
			real_scores = real_out

			z = Variable(t.randn(num_img, 100))
			fake_img = generator(z)
			fake_out = discriminator(fake_img)
			d_loss_fake = criterion(fake_out, fake_label)
			fake_scores = fake_out

			d_loss = d_loss_real + d_loss_fake
			d_optim.zero_grad()
			d_loss.backward()
			d_optim.step()

			# train denerator
			z = Variable(t.randn(num_img, 100))
			fake_img = generator(z)
			output = discriminator(fake_img)
			g_loss = criterion(output, real_label)
			g_optim.zero_grad()
			g_loss.backward()
			g_optim.step()

			if (i + 1) % 10 == 0:
				print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} D real: {:.6f}, D fake: {:.6f}'.format(epoch, num_epoch, d_loss.data[0], g_loss.data[0], real_scores.data.mean(), fake_scores.data.mean() ))
			







if __name__ == '__main__':
	import fire
	fire.Fire()