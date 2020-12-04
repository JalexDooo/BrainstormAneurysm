import torch as t
import warnings


class DefaultConfig(object):
    env = 'default'
    vis_port = 8097
    model = 'UNet3D'

    train_root_path = '/home/sunjindong/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    val_root_path = '/home/sunjindong/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    # '/home/sunjindong/dataset/BraTS_TestData_JSun_paper31/MICCAI_BraTS2020_TestingData'

    local_root_path = '/Users/juntysun/Downloads/数据集/MICCAI_BraTS_2019_Data_Training'

    test_img_path = ''
    test_images = './test_images/'

    task = 'Random'

    predict_nibable_path = './predict_nibable'

    load_model_path = None
    batch_size = 4
    use_gpu = True
    num_workers = 0
    print_freq = 20

    max_epoch = 2
    random_epoch = 4
    lr = 0.001
    lr_decay = 0.99
    weight_decay = 1e-4

    def _parse(self, kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

    # print('user config:')
    # for k, v in self.__class__.__dict__.items():
    # 	if not k.startswith('_'):
    # 		print(k, getattr(self, k))


opt = DefaultConfig()


class BraTS2020Config(object):
    model = 'NormalResNet'

    is_train = True
    predict_path = './predict_nibable'
    predict_figure = './figure'
    description = ''

    dataset_train_path = '/home/sunjindong/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    dataset_val_path = '/home/sunjindong/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData' 
    dataset_random_width = 64

    training_use_gpu = True
    training_use_gpu_num = 4
    training_num_workers = 0
    training_load_model = None

    training_criterion = 'SingleCrossEntropyDiceLoss'
    training_batch_size = 2
    training_max_epoch = 100
    training_lr = 0.001
    training_lr_decay = 1.0

    val_use_gpu = True
    val_use_gpu_num = 1
    val_num_workers = 0
    val_load_model = None

    val_batch_size = 1
    val_max_epoch = 1

    model_vae_flag = True
    model_input_shape = [144, 192, 192]

    def _parse(self, kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

    # print('user config:')
    # for k, v in self.__class__.__dict__.items():
    # 	if not k.startswith('_'):
    # 		print(k, getattr(self, k))


config = BraTS2020Config()
