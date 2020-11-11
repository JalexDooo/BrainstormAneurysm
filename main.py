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

from data.dataset import *
from loss_function.nvnet_loss import *

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
import loss_function
from utils.utils import *

from config import opt, config


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


def tem(**kwargs):
	config._parse(kwargs)

	Loss_list = [0.9060841432418102, 0.4646485366156197, 0.2517089592579127, 0.14397606339930924, 0.09500968314478302, 0.06515735441083463, 0.05543600145207793, 0.04790010517485105, 0.04810570260338115, 0.043587040414826736, 0.040156588354643356, 0.040909058705025106, 0.03915702878777158, 0.036831541169086914, 0.03660739647845427, 0.03548712919171169, 0.03355729073697776, 0.03598278287707896, 0.03422316055068962, 0.03272630181947908, 0.03205239042448734, 0.033578918301984756, 0.03182102599286917, 0.030297669266811825, 0.03111161453644998, 0.03018755124082093, 0.028785930016798012, 0.03044026321800311, 0.03021719267640822, 0.029493754626973918, 0.029242040296349383, 0.029770615753656072, 0.027573490765657998, 0.028343504088205324, 0.029215381318652655, 0.02742839896824127, 0.02829551860240418, 0.02799897024272578, 0.027780495514519662, 0.0269147235280658, 0.02698005991258568, 0.02745457336743611, 0.025576961574080813, 0.027843704864924365, 0.026197851334156877, 0.025821218312631008, 0.025703038604727484, 0.026291221169420204, 0.02656419850393511, 0.026105564168129276, 0.024779427844764616, 0.025622959757866624, 0.024363408081278085, 0.024735295036796646, 0.024969355671349516, 0.024282555526078966, 0.02457734217771481, 0.023479386933762184, 0.026785566575134978, 0.024280116176695055, 0.02334197691371213, 0.02418036306187198, 0.024370898584142795, 0.024270859750479802, 0.023216738704584836, 0.022933391788115737, 0.022814854776476295, 0.022869899604760165, 0.02226361752223239, 0.02310443899353986, 0.022464410631684036, 0.023261236678588756, 0.022566200436484267, 0.022900485741051604, 0.021728818667224162, 0.022447481246631543, 0.02256532964698726, 0.02080414041240499, 0.02244440519563847, 0.022235420593553185, 0.022607580943024293, 0.02213991472070176, 0.02321787059493418, 0.021562875786465405, 0.021731322774282122, 0.020941530431311986, 0.021479776322670462, 0.021958093219034892, 0.020991694779714888, 0.021385667930854193, 0.020069255085921257, 0.021089446603763082, 0.020998854668707483, 0.02034049168146264, 0.020204229206320153, 0.020786246782566156, 0.021765326829470286, 0.02055176001874098, 0.02214988723792737, 0.02107954270401243]
	Accuracy_list = [0.6028009947130609, 1.2159051688008586, 1.6437557679274182, 1.905297062278931, 2.040689949839584, 2.1493000689818897, 2.1262837576299956, 2.1596856835462983, 2.1956527176096867, 2.126188115070546, 2.210862534584935, 2.189274929881179, 2.1641230837423966, 2.2070860249224635, 2.237178263522053, 2.1850273036602608, 2.2359623338218384, 2.168383261772752, 2.219611302115002, 2.2604573835124455, 2.226269274305557, 2.2131835424891784, 2.181370196646559, 2.2310139402590425, 2.3019985097523654, 2.2578711569389633, 2.2781290010776303, 2.2462227489293296, 2.2736096483607926, 2.2882377725640826, 2.2537473141265587, 2.2891227043359703, 2.2568977458859374, 2.2537765844088034, 2.2788890633771492, 2.2794709004270066, 2.2800443267333073, 2.2911095402648947, 2.2923114297059306, 2.305772697074342, 2.285622888757818, 2.294476475252834, 2.2609234574029413, 2.2844253083115063, 2.2722711938642677, 2.297830852159806, 2.3086472177009796, 2.297996816842449, 2.28171236762056, 2.294585606444584, 2.273699553521505, 2.3293824691775797, 2.31212514300463, 2.2431135888857825, 2.327142351176931, 2.2899876598338236, 2.305820981812783, 2.3148039597742947, 2.2958236630784645, 2.279493278122524, 2.2811730248958124, 2.3394307753365244, 2.294470631279533, 2.3084583659043485, 2.315076891310145, 2.2929538906860003, 2.2919521656403266, 2.3599402885220813, 2.323793202973872, 2.295104679970041, 2.3171633802010945, 2.2865117740202003, 2.336709048990695, 2.337398983558175, 2.312693477820084, 2.3230380691810284, 2.3039304722866487, 2.3294730496793496, 2.32675697521915, 2.322423419151203, 2.323859450777491, 2.301934106779944, 2.342640603000194, 2.323430221574147, 2.3061896659435983, 2.3024567355828944, 2.301847109975955, 2.3146363268895445, 2.3130681930059835, 2.296297948479815, 2.349626747664289, 2.327917199409569, 2.3347469441683413, 2.3132383944674055, 2.343317463155724, 2.303038463645329, 2.31063364870399, 2.3125370820420557, 2.3428050971278958, 2.3214468860845097]
	c = []
	c.append(Loss_list)
	c.append(Accuracy_list)

	figure_multi_array(c, labels=['loss', 'acc'], save_path='loss.jpg', xlabel='xlabel', ylabel='ylabel', title='title')
	

def test(**kwargs):
	config._parse(kwargs)
	# if config.training_use_gpu and not t.cuda.is_available():
	# 	print('Using GPU, but cuda is not available!!')
	# 	return
	
	model = getattr(models, config.model)()
	if config.training_use_gpu:
		gpu_devices = [i for i in range(config.training_use_gpu_num)]
		print(gpu_devices)


def train(**kwargs):
	# config modified.
	config._parse(kwargs)
	if config.training_use_gpu and t.cuda.is_available() == False:
		raise BaseException('Using GPU, but cuda is not available!!')
	
	# get model from models.__init__.py and model for gpu parallel.
	if config.model == 'NvNet':
		model = getattr(models, config.model)(vae_flag=config.model_vae_flag, shape=config.model_input_shape)
	else:
		model = getattr(models, config.model)()
	# if config.model_vae_flag:
	# 	criterion = CombinedLoss(0.1, 0.1)
	# else:
	# 	criterion = MultiClassDiceLoss()
	
	criterion = getattr(loss_function, config.training_criterion)()
	
	if config.training_use_gpu:
		gpu_devices = [i for i in range(config.training_use_gpu_num)]
		model = nn.DataParallel(model)
		model = model.cuda(device=gpu_devices[0])
		criterion = criterion.cuda()

	# model save dir.
	model_save_dir = 'ckpt_' + config.model + '_' + config.description + '/'
	if not os.path.exists(model_save_dir):
		os.mkdir(model_save_dir)

	# loading pretrained model.
	if config.training_load_model:
		if not os.path.exists(model_save_dir+config.training_load_model):
			raise BaseException('The loaded model is not exists.')
		model.load_state_dict(t.load('./' + model_save_dir + config.training_load_model))
		print('load model -> {}'.format(config.training_load_model))

	optimizer = optim.Adam(params=model.parameters(), lr=config.training_lr)
	if config.training_lr_decay != 1.0:
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1-config.training_lr_decay, patience=3000, verbose=True)

	model.train()
	train_loss_epoch = []
	train_dice_epoch = []
	for epoch in range(config.training_max_epoch):
		train_dataset = BraTS2020(config)
		train_dataloader = DataLoader(train_dataset, batch_size=config.training_batch_size, shuffle=True, num_workers=config.training_num_workers)

		for ii, (image, label, onehot_label) in enumerate(train_dataloader):
			# print("label.shape: {}".format(label.shape))
			# print("label0: {}, label1: {}, label2: {}, label3: {}".format((label[:, 0, 0, ...]==1).sum(), (label[:, 0, 1, ...]==1).sum(), (label[:, 0, 2, ...]==1).sum(), (label[:, 0, 3, ...]==1).sum()))
			# raise Exception('debug testing.')
			train_loss = []
			train_dice = []
			for kk in range((config.model_input_shape[-1] // config.dataset_random_width) * 2):
				# print('kkkkkkk: {}'.format(kk))
				# print('image.shape: {}, label.shape: {}'.format(image.shape, label.shape))
				if config.training_use_gpu:
					_image = image[:, kk, :, :, :, :].cuda()
					_label = label[:, kk, :, :, :, :].cuda()
					_onehot_label = onehot_label[:, kk, :, :, :].cuda()
				else:
					_image = image[:, kk, :, :, :, :]
					_label = label[:, kk, :, :, :, :]

				optimizer.zero_grad()
				if config.model_vae_flag:
					predict, distr = model(_image)
					# predict = (predict > 0.5).float()
					losses = criterion(predict, _label, distr)
					print('distr.shape: {}'.format(distr.shape))
				else:
					predict = model(_image)
					# print('predict.shape: {}, _label.shape: {}'.format(predict.shape, _label.shape))
					# predict = (predict > 0.5).float()
					losses = criterion(predict, _label, _onehot_label)
				# print("predict.shape: {}".format(predict.shape))
				value, predict = t.max(predict, dim=1)
				# predict = sigmoid_deal(predict)

				# print('predict {}, {}, {}, {}'.format((predict==0).sum(), (predict==1).int().sum(), (predict==2).sum(), (predict==3).sum()))

				train_dice.append(dice(predict, _onehot_label.long())/4)
				train_loss.append(float(losses))
				losses.backward()
				optimizer.step()
				if config.training_lr_decay != 1.0:
					scheduler.step(losses)

		loss_debug = sum(train_loss) / len(train_loss)
		dice_debug = sum(train_dice) / len(train_dice)

		print('training: {}/{} th, lr: {:.10f}, losses: {:0.6f}, dice: {}'.format(epoch, config.training_max_epoch, optimizer.param_groups[0]['lr'], loss_debug, dice_debug))
		
		train_loss_epoch.append(loss_debug)
		train_dice_epoch.append(dice_debug)

			# raise BaseException('one epoch one case interruption.')
		if (epoch+1) % 5 == 0:
			torch.save(model.state_dict(), os.path.join(model_save_dir, 'epoch_{}.pth'.format(epoch+1)))
	print('train_loss: {}'.format(train_loss_epoch))
	print('train_dice: {}'.format(train_dice_epoch))


def val(**kwargs):
	# config modified.
	config._parse(kwargs)
	config.is_train = False
	if config.val_use_gpu and t.cuda.is_available() == False:
		raise BaseException('Using GPU, but cuda is not available!!')
	
	# get model from models.__init__.py and model for gpu parallel.
	if config.model == 'NvNet':
		model = getattr(models, config.model)(vae_flag=config.model_vae_flag, shape=config.model_input_shape)
	else:
		model = getattr(models, config.model)()
	if config.model_vae_flag:
		criterion = CombinedLoss(0.1, 0.1)
	else:
		criterion = SoftDiceLoss()
	
	if config.val_use_gpu:
		gpu_devices = [i for i in range(config.val_use_gpu_num)]
		model = nn.DataParallel(model)
		model = model.cuda(device=gpu_devices[0])
		criterion = criterion.cuda()

	# model save dir.
	model_save_dir = 'ckpt_' + config.model + '_' + config.description + '/'
	if not os.path.exists(model_save_dir):
		os.mkdir(model_save_dir)

	# loading pretrained model.
	if config.val_load_model:
		if not os.path.exists(model_save_dir+config.val_load_model):
			raise BaseException('The loaded model is not exists.')
		model.load_state_dict(t.load('./' + model_save_dir + config.val_load_model))
		print('load model -> {}'.format(config.val_load_model))

	size_axis = config.model_input_shape[-1] // config.dataset_random_width

	model.eval()
	for epoch in range(config.val_max_epoch):
		val_dataset = BraTS2020(config)
		val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.val_num_workers)
		predictss = []
		predicts_names = []

		for ii, (image, name, box_min, box_max) in enumerate(val_dataloader):
			out_predicts = []
			for kk in range(size_axis * 2):
				if config.val_use_gpu:
					_image = image[:, kk, :, :, :, :].cuda()
				else:
					_image = image[:, kk, :, :, :, :].cuda()

				with t.no_grad():
					if config.model_vae_flag:
						predict, distr = model(_image)

						print('distr.shape: {}'.format(distr.shape))
					else:
						predict = model(_image)

				print("predicy.shape: {}".format(predict.shape))
				value, predict = t.max(predict, dim=1)
				print("predict0123: 0: {}, 1: {}, 2: {}, 3: {}".format((predict==0).sum(), (predict==1).sum(), (predict==2).sum(), (predict==3).sum()))
				# predict = sigmoid_deal(predict)
				# raise Exception('debug exception for label out processing.')
				out_predicts.append(predict.int())

			out_z_axis = t.cat((out_predicts[0], out_predicts[1]), dim=-1)
			for kk in range(size_axis-2):
				out_z_axis = t.cat((out_z_axis, out_predicts[kk+2]), dim=-1)
			
			out_y_axis = t.cat((out_predicts[0+size_axis], out_predicts[1+size_axis]), dim=-1)
			for kk in range(size_axis-2):
				out_y_axis = t.cat((out_y_axis, out_predicts[size_axis+kk+2]), dim=-1)

			predict = t.cat((out_z_axis, out_y_axis), dim=-2)
			predict = predict.data.cpu().numpy()

			pp = np.zeros((155, 240, 240), dtype=np.int32)
			x, y, z = box_min
			x_, y_, z_ = config.model_input_shape
			pp[x:x + x_, y:y + y_, z:z + z_] = predict[0]
			predict = np.transpose(pp, [2, 1, 0])
			predict = out_precessing(predict)
			predictss.append(predict)
			predicts_names.append(name[0])
			break


	predictss = np.array(predictss)
	predicts_names = np.array(predicts_names)

	# output path
	predict_path = config.predict_path
	if not os.path.exists(predict_path):
		os.mkdir(predict_path)
	
	# loading original image header.
	save_nii_head_path = config.dataset_val_path + '/BraTS20_Validation_001/BraTS20_Validation_001_flair.nii.gz'
	# '/BraTS20_Testing_001/BraTS20_Testing_001_flair.nii.gz'
	# save_nii_head_path = opt.val_root_path + '/Brats18_CBICA_AAM_1/Brats18_CBICA_AAM_1_flair.nii.gz'
	head_image = nib.load(save_nii_head_path)
	affine = head_image.affine

	for i in range(len(predictss)):
		output = nib.Nifti1Image(predictss[i], affine)
		print('save nii -> {}'.format(predicts_names[i]))
		nib.save(output, predict_path + predicts_names[i] + '.nii.gz')


def gan_test():
	from torchvision import datasets
	from torchvision import transforms
	import models.gan as gan

	img_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=0.5, std=0.5)
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

			# train generator
			z = Variable(t.randn(num_img, 100))
			fake_img = generator(z)
			output = discriminator(fake_img)
			g_loss = criterion(output, real_label)
			g_optim.zero_grad()
			g_loss.backward()
			g_optim.step()

			if (i + 1) % 10 == 0:
				print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} D real: {:.6f}, D fake: {:.6f}'.format(epoch, num_epoch, d_loss.data[0], g_loss.data[0], real_scores.data.mean(), fake_scores.data.mean() ))


def ttt():
	pi = np.pi
	print("pi is {:.6f}".format(pi))


if __name__ == '__main__':
	import fire
	fire.Fire()