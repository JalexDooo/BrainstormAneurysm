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


def Aneu_train(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0, 1, 2, 3]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)(1, 2)
	print('model is : ', opt.model)
	save_dir = 'ckpt_aneu_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	fig_loss = []
	fig_dice = []
	lr = opt.lr
	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------' % (epoch))

		# pytorch数据处理
		train_data = AneuMulti(opt.aneu_path)
		# test_data = AneuMulti(opt.aneu_val_path)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		# test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
		print('train and test dataloader load finished.')

		criterion = nn.CrossEntropyLoss()
		print('lr: ', opt.lr)
		optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
		lr *= opt.lr_decay

		print('criterion and optimizer is finished.')
		# print(model.eval())

		train_loss = []
		train_dice = []
		model.train()
		for ii, (image, label, name) in enumerate(train_dataloader):
			for k in range(2):
				img = image[:, k, :, :, :, :]
				lbl = label[:, k, :, :, :]

				for fck in range(7):
					img_ = img[:, :, :, :, 64*fck:64*(fck+1)]
					lbl_ = lbl[:, :, :, 64*fck:64*(fck+1)]
					img_ = img_.cuda()
					lbl_ = lbl_.cuda()

					optimizer.zero_grad()
					predicts = model(img_)
					loss = criterion(predicts, lbl_.long())
					train_loss.append(float(loss))
					loss.backward()
					optimizer.step()

					value, tmp = t.max(predicts, dim=1)
					d = dice(tmp, lbl_.long())
					train_dice.append(d)
		print('train_loss : ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
		print('train_dice : ' + str(sum(train_dice) / (len(train_dice) * 1.0)))
		fig_loss.append(sum(train_loss) / (len(train_loss) * 1.0))
		fig_dice.append(sum(train_dice) / (len(train_dice) * 1.0))

		torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__epoch_%s.pth'%(opt.task, epoch)))

	torch.save(model.state_dict(), os.path.join(save_dir, 'task_%s__final_epoch.pth' % (opt.task)))
	print('--------------fig loss-------------')
	print(fig_loss)
	print('--------------fig dice-------------')
	print(fig_dice)


def Aneu_predict(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)(in_data=1, out_data=2)
	print('model is : ', opt.model)
	save_dir = 'ckpt_aneu_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	path = opt.aneu_output_path
	path += '_' + opt.model + '_' + opt.load_model_path[:-4]

	if not os.path.exists(path):
		os.mkdir(path)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))

		# pytorch数据处理
		train_data = AneuMulti(opt.aneu_path, is_train=False, val_path=opt.aneu_val_path)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

		train_loss = []
		train_dice = []
		for ii, (image, label, img_shape, index_min, index_max, affine, name) in enumerate(train_dataloader):
			predict = []
			iimage = []
			for k in range(2):
				img = image[:, k, :, :, :, :]
				img = img.cuda()

				for fck in range(7):
					img_ = img[:, :, :, :, 64*fck:64*fck+64]

					predicts = model(img_)

					value, tmp = t.max(predicts, dim=1)
					predict.append(tmp.int().data.cpu())
					iimage.append(img_.data.cpu())

			predict1 = t.cat((predict[0], predict[1]), dim=3)
			for i in range(2, 7):
				predict1 = t.cat((predict1, predict[i]), dim=3)
			predict2 = t.cat((predict[7], predict[8]), dim=3)
			for i in range(9, 14):
				predict2 = t.cat((predict2, predict[i]), dim=3)
			out_predict = t.cat((predict1, predict2), dim=1).numpy()

			image1 = t.cat((iimage[0], iimage[1]), dim=4)
			for i in range(2, 7):
				image1 = t.cat((image1, iimage[i]), dim=4)
			image2 = t.cat((iimage[7], iimage[8]), dim=4)
			for i in range(9, 14):
				image2 = t.cat((image2, iimage[i]), dim=4)
			out_image = t.cat((image1, image2), dim=2).numpy()
			out_image = out_image[:, 0, :, :, :]

			# [128, 448, 448]
			# print('shape: ', out_image[0].shape)
			out_image = crop_with_box(out_image[0], np.array([0, 0, 0]), img_shape)
			out_predict = crop_with_box(out_predict[0], np.array([0, 0, 0]), img_shape)
			# print('shape_: ', out_image.shape)

			out_image = np.transpose(out_image, [2, 1, 0])
			out_predict = np.transpose(out_predict, [2, 1, 0])

			# affine [0]: batch size
			output = nib.Nifti1Image(out_predict, affine[0])
			nib.save(output, path + '/' + name[0] + '_predict.nii.gz')
			im = nib.Nifti1Image(out_image, affine[0])
			nib.save(im, path + '/' + name[0] + '_image.nii.gz')


def Aneu_test(**kwargs):
	if not t.cuda.is_available():
		print('无法使用CUDA，所以无法训练')
		return

	device_ids = [0]
	opt._parse(kwargs)
	print('Task %s' % (opt.task))

	# 配置模型
	model = getattr(models, opt.model)(in_data=1, out_data=2)
	print('model is : ', opt.model)
	save_dir = 'ckpt_aneu_' + opt.model + '/'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	print('Start load DataParallels')
	model = nn.DataParallel(model)
	model = model.cuda(device=device_ids[0])
	print('Loading DataParallel finished.')
	if opt.load_model_path:
		model.load_state_dict(t.load('./' + save_dir + opt.load_model_path))
		print('load model')

	score = []
	score.append(['Name', 'Dice', 'Sensitivity', 'Specificity'])

	for epoch in range(opt.max_epoch):
		print('----------------------epoch %d--------------------'%(epoch))

		# pytorch数据处理
		train_data = AneuMulti(opt.aneu_path, is_train=True, val_path=opt.aneu_val_path)
		print('train_data and test_data load finished.')

		train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

		for ii, (image, label, name) in enumerate(train_dataloader):
			test_dice = []
			test_sensitivity = []
			test_specificity = []
			predict = []
			iimage = []
			for k in range(2):
				img = image[:, k, :, :, :, :]
				lbl = label[:, k, :, :, :]
				img = img.cuda()
				lbl = lbl.cuda()

				for fck in range(7):
					img_ = img[:, :, :, :, 64*fck:64*fck+64]
					lbl_ = lbl[:, :, :, 64*fck:64*fck+64]
					predicts = model(img_)

					value, tmp = t.max(predicts, dim=1)
					d = dice(tmp, lbl_.long())
					se = sensitivity(tmp, lbl_.long())
					sp = specificity(tmp, lbl_.long())
					predict.append(tmp.int().data.cpu())
					iimage.append(img_.data.cpu())
					test_dice.append(d)
					test_sensitivity.append(se)
					test_specificity.append(sp)
			lbl_dice = sum(test_dice)/len(test_dice)
			lbl_se = sum(test_sensitivity)/len(test_sensitivity)
			lbl_sp = sum(test_specificity)/len(test_specificity)
			score.append([name, str(lbl_dice), str(lbl_se), str(lbl_sp)])
			# if lbl_dice < 0.7:
			print('test dice: {} -->  {}'.format(name, lbl_dice))
			print('test sensitivity: {} -->  {}'.format(name, lbl_se))
			print('test specificity: {} -->  {}'.format(name, lbl_sp))
			# return
	save = pd.DataFrame(score, columns=['Name', 'Dice', 'Sensitivity', 'Specificity'])
	save.to_csv('./' + opt.model + '_' + opt.load_model_path[:-4] + '_multi_score.csv', index=False, header=False)


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
	lr_decay = 0.99
	for i in range(50):
		print(i, ' --> ', lr)
		lr *= lr_decay


if __name__ == '__main__':
	import fire
	fire.Fire()