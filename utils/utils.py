import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt


def netSize(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


def dice(predict, target):
    """

    :param predict: 4D Long Tensor Batch_Size * 16(volume_size) * height * weight
    :param target:  4D Long Tensor Batch_Size * 16(volume_size) * height * weight
    :return:
    """
    smooth = 0.00000001
    batch_num = target.shape[0]
    target = target.view(batch_num, -1)
    predict = predict.view(batch_num, -1)
    intersection = float((target * predict).sum())

    return (2.0 * intersection + smooth) / (float(predict.sum()) + float(target.sum()) + smooth)


def sensitivity(predict, target):
    """

    :param predict: 4D Long Tensor Batch_Size * 16(volume_size) * height * weight
    :param target:  4D Long Tensor Batch_Size * 16(volume_size) * height * weight
    :return:
    """
    smooth = 0.00000001
    batch_num = target.shape[0]
    target = target.view(batch_num, -1)
    predict = predict.view(batch_num, -1)
    intersection = float((target * predict).sum())

    return (intersection + smooth) / (float(target.sum()) + smooth)


def specificity(predict, target):
    """

    :param predict: 4D Long Tensor Batch_Size * 16(volume_size) * height * weight
    :param target:  4D Long Tensor Batch_Size * 16(volume_size) * height * weight
    :return:
    """
    smooth = 0.00000001
    batch_num = target.shape[0]
    target = target.view(batch_num, -1)
    predict = predict.view(batch_num, -1)
    target = (target == 0)
    predict = (predict == 0)
    tn = float((target * predict).sum())
    predict = (predict == 0)
    fp = float((target * predict).sum())

    return (tn + smooth) / (tn + fp + smooth)


def save_array_as_nifty_volume(data, filename, reference_name = None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


def make_box(image, index_min, index_max, data_box):
    """
        抠图，使用get_box()获得的下标。
    """
    shape = image.shape

    for i in range(len(shape)):

        # print('before index[%s]: '%i, index_min[i], index_max[i])

        # 按照data_box扩大或缩小box。
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

        # print('index[%s]: '%i, index_min[i], index_max[i])

        if index_max[i] - index_min[i] != data_box[i]:
            index_max[i] = index_min[i] + data_box[i]

        index_max[i] = int(index_max[i])
        index_min[i] = int(index_min[i])

        # print('after index[%s]: '%i, index_min[i], index_max[i])
    return index_min, index_max


def crop_with_box(image, index_min, index_max):
    """
        按照box分割图像。
    """
    # return image[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]
    x = index_max[0] - index_min[0] - image.shape[0]
    y = index_max[1] - index_min[1] - image.shape[1]
    z = index_max[2] - index_min[2] - image.shape[2]
    img = image
    img1 = image
    img2 = image

    if x > 0:
        img = np.zeros((image.shape[0]+x, image.shape[1], image.shape[2]))
        img[x//2:image.shape[0]+x//2, :, :] = image[:, :, :]
        img1 = img

    if y > 0:
        img = np.zeros((img1.shape[0], img1.shape[1]+y, img1.shape[2]))
        img[:, y//2:image.shape[1]+y//2, :] = img1[:, :, :]
        img2 = img

    if z > 0:
        img = np.zeros((img2.shape[0], img2.shape[1], img2.shape[2]+z))
        img[:, :, z//2:image.shape[2]+z//2] = img2[:, :, :]

    return img[np.ix_(range(index_min[0], index_max[0]), range(index_min[1], index_max[1]), range(index_min[2], index_max[2]))]


def out_precessing(label):
    tmp = np.asarray(label)

    print('before out_processing: label is 0: {}, label is 1: {}, label is 2: {}, label is 3: {}, label is 4: {}'.format((label==0).sum(), (label==1).sum(), (label==2).sum(), (label==3).sum(), (label==4).sum()))

    if (tmp==3).sum() <= 500:
        tmp = (tmp == 3)*1 + (tmp == 1)*1 + (tmp==2)*2
    tmp = (tmp == 3)*4 + (tmp == 1)*1 + (tmp==2)*2
    print('after out_processing: label is 0: {}, label is 1: {}, label is 2: {}, label is 3: {}, label is 4: {}'.format((tmp==0).sum(), (tmp==1).sum(), (tmp==2).sum(), (tmp==3).sum(), (tmp==4).sum()))
    c = (tmp!=0)*(tmp!=1)*(tmp!=2)*(tmp!=3)*(tmp!=4)
    print('c.sum(): {}'.format(c.sum()))
    return tmp


def load_nii_to_array(path):
    """
        将.nii格式加载成图像数据。
    """
    image = nib.load(path)
    image = image.get_data()

    image = np.transpose(image, [2, 1, 0])
    # image = np.array(image) ###
    return image


def rotation(image, theta, c=np.array([])):
    theta = -np.pi * theta / 180
    if c.size == 0:
        c = np.array([(image.shape[0]-1)//2, (image.shape[1]-1)//2, (image.shape[2]-1)//2])

    mean = np.mean(image)
    new_image = np.zeros(image.shape)
    print(new_image.shape)

    # x rotation


def figure_multi_array(arrays, labels, save_path, xlabel=None, ylabel=None, title=None):
    """ figure of array.

    ``array: [n, [arr]]`` and arr has the same size.

    n max is 8

    Or color=(0.3,0.3,0.4) ...

    b               蓝色  
    g               绿色
    r               红色  
    y               黄色
    c               青色
    k               黑色   
    m               洋红色 
    w               白色
    """
    arr_idx = len(arrays)

    if arr_idx > 8:
        raise Exception('Figures number is too large. It should be equal or less than 8.')

    colors = ['b', 'g', 'r', 'y', 'c', 'k', 'm', 'w']
    plt.figure()
    for idx in range(arr_idx):
        array = arrays[idx]
        color = colors[idx]
        x = range(0, len(array))
        y = array
        print(labels[idx])
        plt.plot(x, y, '-', color=color, label=labels[idx])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path)


def sigmoid_deal(image):
    """ sigmoid processing for onehot decoding.

    Args:
        image.shape: [bz, 4, 144, 96, 96]
    
    return:
        image with shape: [bz, 144, 96, 96]

    """
    print('image.shape: {}'.format(image.shape))
    image = image > 0.5
    # print('>0.5 ---: 0: {}, 1: {}, 2: {}, 4: {}'.format((image[:, 0, ...]==1).sum(), (image[:, 1, ...]==1).sum(), (image[:, 2, ...]==1).sum(), (image[:, 3, ...]==1).sum()))
    # if intersactive
    label0 = ((image[:, :, 0, ...]==1)).int()*0
    label1 = ((image[:, :, 3, ...]==0)*(image[:, :, 2, ...]==0)*(image[:, :, 1, ...]==1)).int()*1
    label2 = ((image[:, :, 3, ...]==0)*(image[:, :, 2, ...]==1)).int()*2 
    label4 = ((image[:, :, 3, ...]==1)).int()*4.0

    # print('label0: {}, label1: {}, label2: {}, label4: {}'.format(label0.sum(), label1.sum(), label2.sum(), label4.sum()))

    _image = label0 + label1 + label2 + label4
    # print('_image: 1: {}, 2: {}, 4: {}'.format((_image==1).sum(), (_image==2).sum(), (_image==4).sum()))
    image = _image.int()
    # print('<function> sigmoid_deal: \nimage.shape: {}, image_debug: {}'.format(image.shape, image[0, :, :, :]))
    # print('image: 1: {}, 2: {}, 4: {}'.format((image==1).sum(), (image==2).sum(), (image==4).sum()))

    return image
