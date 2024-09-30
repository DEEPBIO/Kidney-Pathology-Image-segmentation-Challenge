import torch
import pdb
import torch.nn.functional as F 
import torch.nn as nn 
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np 
import torchvision
import PIL

import numpy as np
from sklearn.metrics import confusion_matrix

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def calculate_cf_matrix(outputs, targets):
    # 1차원 벡터로 펼치기
    outputs = outputs.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    binary_outputs = (outputs >= 0.).astype(int) # before sigmoid
    binary_targets = (targets >= 0.5).astype(int)
    
    # 혼동 행렬 계산
    cf_matrix = confusion_matrix(binary_targets, binary_outputs, labels=(0,1))
    return cf_matrix


def calculate_metrics(cf_matrix):
    tn, fp, fn, tp = cf_matrix.ravel()
    
    # Dice coefficient 계산
    dice_coefficient = (2 * tp) / (2 * tp + fp + fn)
    accuracy = (tp+tn)/(tn + fp + fn + tp)
    iou = tp / (tp + fn + fp)

    
    return dice_coefficient, accuracy, iou



class my_loss_bce(torch.nn.Module):

    def __init__(self, cancer_weight) : 
        super().__init__()
        self.cancer_weight = cancer_weight

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        results = targets*torch.log(inputs+1e-4) + (1-targets)*torch.log(1-inputs+1e-4)
        results = results * -1 #torch.sum(-1*results, dim=1)
        weights_for_batch = targets*(self.cancer_weight-1)+1 # .squeeze(1)
        loss = (results * weights_for_batch).sum() / weights_for_batch.sum()
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs) # sigmoid를 통과한 출력이면 주석처리
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice 
        
# from: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = round(W * cut_rat)
    cut_h = round(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)//4*4
    bby1 = np.clip(cy - cut_h // 2, 0, H)//4*4
    bbx2 = np.clip(cx + cut_w // 2, 0, W)//4*4
    bby2 = np.clip(cy + cut_h // 2, 0, H)//4*4

    return bbx1, bby1, bbx2, bby2

def cutmix(input, target, beta=1.0) : 
    B,C,H,W = input.shape 
    B1,C1,H1,W1 = target.shape 
    ratio = H/H1 
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(input.size()[0])
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    target[:, :, round(bbx1/ratio):round(bbx2/ratio), round(bby1/ratio):round(bby2/ratio)] = \
        target[rand_index, :, round(bbx1/ratio):round(bbx2/ratio), round(bby1/ratio):round(bby2/ratio)]
    # adjust lambda to exactly match pixel ratio
    #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, target 

# modified from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup(x, y1, y2=None, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y1 = lam * y1 + (1 - lam) * y1[index, :]
    if y2: 
        mixed_y2 = lam * y2 + (1 - lam) * y2[index, :]
        return mixed_x, mixed_y1, mixed_y2
    else: 
        return mixed_x, mixed_y1
