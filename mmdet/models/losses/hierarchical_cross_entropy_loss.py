import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss





def single_level_cross_entropy(pred, label, wordtree):
    if label == 0 :
        return 0

    if isinstance(label,torch.Tensor): 
        label_i = int(label.cpu().numpy())
    else:
        label_i = label
        label = torch.tensor(label).type_as(pred).long()
    
    # 找到上一级节点的idx和同一级所有节点的idx
    parent = wordtree[label_i]['parent']
    same_level = wordtree[parent]['children'].type_as(label)
    
    # 对同一级的节点求解softmax
    level_pred = pred[same_level-1]
    level_label = torch.where(same_level==label)[0]
    level_loss = (-F.log_softmax(level_pred,dim=0)[level_label])[0]
    
    # 对上一级的节点求解softmax
    parent_loss = single_level_cross_entropy(pred, parent, wordtree)
    
    return level_loss + parent_loss

def hierarchical_cross_entropy(preds, labels, wordtree, weight=None, reduction='mean', avg_factor=None):
    N = preds.size(0)
    loss = torch.zeros(N).type_as(preds)
    for i in list(range(N)):
        pred = preds[i,:]
        label = labels[i]
        
        loss[i] += single_level_cross_entropy(pred, label, wordtree)
        
    if reduction=='mean':
        loss = loss.sum()/N
    elif reduction=='sum':
        loss = loss.sum()
    else:
        pass
      
    return loss





@LOSSES.register_module
class HierarchicalCrossEntropyLoss(nn.Module):
    '''
    wordtree example:
    # 目前不支持背景类的使用，只适合yolo的用法
    wordtree = { 0: {'parent': -1, 'children': [1,2] },  # 0属于根节点,不对应网络输出
           1: {'parent': 0, 'children': None},
           2: {'parent': 0, 'children': [3,4]},
          3: {'parent': 2, 'children': None},
          4: {'parent': 2, 'children': None},
         }
    '''

    def __init__(self,
                 use_sigmoid=False,
                 wordtree=None,
                 reduction='mean',
                 loss_weight=1.0):
        super(HierarchicalCrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False)
        self.use_sigmoid = use_sigmoid
        
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        self.wordtree = wordtree
        
        # 将children转为tensor存储
        for k in wordtree:
            if self.wordtree[k]['children'] is not None:
                self.wordtree[k]['children'] = torch.tensor( self.wordtree[k]['children'] )
        
        self.cls_criterion = hierarchical_cross_entropy
        #if self.use_sigmoid:
        #    self.cls_criterion = binary_cross_entropy
        #else:
        #    self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction=None,
                **kwargs):
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction if reduction else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.wordtree,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
    
    def parase_output(self, cls_scores, root=0, root_weight=1.0, min_split=0.5):
        '''
        '''
        
        # 对同一级进行softmax
        same_level = self.wordtree[root]['children']
        cls_scores[:, same_level-1] = F.softmax(cls_scores[:, same_level-1], dim=1) * root_weight
        
        # 对children进行softmax
        for children in same_level:
            root_weight = cls_scores[:, children-1].clone().view(-1,1)
            root_weight[root_weight<=min_split] = 0
            if self.wordtree[children.item()]['children'] is None:
                continue
            else:
                cls_scores[:, children-1][cls_scores[:, children-1]>min_split] = -1
            
                cls_scores = self.parase_output(cls_scores, root=children.item(), root_weight=root_weight, min_split=0.1)
            
        return cls_scores
    
    
