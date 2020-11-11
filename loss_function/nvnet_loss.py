import torch as t
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division, 
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()
    
    def forward(self, y_pred, y_true, eps=1e-8):
        # print('y_pred, y_true: {}, {}'.format(y_pred.shape, y_true.shape))
        intersection = t.sum(t.mul(y_pred, y_true))
        union = t.sum(t.mul(y_pred, y_pred)) + t.sum(t.mul(y_true, y_true)) + eps
        dice = 2 * intersection / union
        dice_loss = 1-dice

        return dice_loss


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()
    
    def forward(self, mean, std):
        return t.mean(t.mul(mean, mean)) + t.mean(t.mul(std, std)) - t.mean(t.log(t.mul(std, std))) - 1


class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''
    def __init__(self, k1=0.1, k2=0.1):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()
    
    def forward(self, y_pred, y_true, y_mid):
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred, seg_truth =  (y_pred[:,0,:,:,:], y_true[:,0,:,:,:])
        vae_pred, vae_truth = (y_pred[:,1:,:,:,:], y_true[:,1:,:,:,:])
        dice_loss = self.dice_loss(seg_pred, seg_truth)
        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        #print("dice_loss:%.4f, L2_loss:%.4f, KL_div:%.4f, combined_loss:%.4f"%(dice_loss,l2_loss,kl_div,combined_loss))
        
        return combined_loss


class DiceLoss(_Loss):
    """Dice loss.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, predict, target, eps=1e-6):
        batch_size = predict.size(0)

        predict_flat = predict.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        intersection = predict_flat * target_flat

        dice = 2 * (intersection.sum(1)) / (predict_flat.sum(1) + target_flat.sum(1) + eps)
        loss = 1 - dice.sum() / batch_size
        # print("<DiceLoss> -> intersection.sum(1): {}, predict_flat.sum(1): {}, target_flat.sum(1): {}, dice.sum(): {}, loss: {}".format(intersection.sum(1), predict_flat.sum(1), target_flat.sum(1), dice.sum(), loss))
        return loss


class MultiClassDiceLoss(_Loss):
    """Multi Class Dice Loss.
    """
    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()
    
    def forward(self, predict, target, weights=None):
        dice = DiceLoss()
        _class = target.shape[1]

        # if weights is None:
        #     weights = t.ones(_class)
        
        total_loss = 0.0

        for i in range(_class):
            dice_loss = dice(predict[:, i, ...], target[:, i, ...])
            if weights is not None:
                dice_loss *= weights[i]
            total_loss += dice_loss
        
        return total_loss


class CrossEntropyDiceLoss(_Loss):
    def __init__(self):
        super(CrossEntropyDiceLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_loss = SoftDiceLoss()
    
    def forward(self, predict, target, onehot_target, weights=[0.95, 0.05]):
        loss1 = self.cross_entropy_loss(predict, onehot_target.long())
        loss2 = self.dice_loss(predict, target)

        total_loss = weights[0] * loss1 + weights[1] * loss2
        return total_loss


class SingleCrossEntropyDiceLoss(_Loss):
    def __init__(self):
        super(SingleCrossEntropyDiceLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, predict, target, onehot_target):
        return self.loss(predict, onehot_target.long())
