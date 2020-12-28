import torch as t
import warnings


class BraTS2020Config(object):
    model = 'BaseLineModel'

    is_train = True
    predict_path = './predict_nibable'
    predict_figure = './figure'
    description = ''

    train_path = '/home/sunjindong/dataset/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    val_path = '/home/sunjindong/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData' 
    random_width = 64
    image_box = [192, 192, 144]

    use_gpu = True
    use_gpu_num = 4
    load_model = None

    loss_function = 'CrossEntropyLoss'
    batch_size = 8
    max_epoch = 200
    lr = 0.001
    lr_decay = 0.2

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
