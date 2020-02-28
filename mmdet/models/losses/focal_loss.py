import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def softmax_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=None,
                          reduction='mean',
                          avg_factor=None):
    # in py_softmax_focal_loss
    # alpha is None or [weight_0, weight_1, ...],example alpha = [0.25, 0.75]
    # "weighted_loss" is not applicable
    N = pred.size(0)
    C = pred.size(1)
    pred_softmax = F.softmax(pred)
    
    class_mask = pred.data.new(N, C).fill_(0)
    class_mask = torch.tensor(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    class_mask = class_mask.type_as(pred)
    
    
    probs = (pred_softmax*class_mask).sum(1).view(-1,1)
    log_p = probs.log()
    
    if alpha is None:
        class_alpha = torch.tensor(1.0).type_as(pred)
    else:
        alpha = torch.tensor(alpha).view(-1,1)
        class_alpha = alpha[ids.data.view(-1)]
        class_alpha = torch.tensor(class_alpha).type_as(pred)
    loss = -class_alpha*(torch.pow((1-probs), gamma))*log_p 
        
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        #assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        if self.use_sigmoid:
            self.cls_criterion = sigmoid_focal_loss
        else:
            self.cls_criterion = softmax_focal_loss

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * self.cls_criterion(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
