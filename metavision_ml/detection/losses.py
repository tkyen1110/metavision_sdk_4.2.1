# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
detection losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce(loss, mode='none'):
    """
    reduce mode

    Args:
        loss: multidim loss
        mode: either "mean", "sum" or "none"
    Returns:
        reduced loss
    """
    if mode == 'mean':
        loss = loss.mean()
    elif mode == 'sum':
        loss = loss.sum()
    return loss


def softmax_focal_loss(pred, target, reduction='none'):
    """Softmax focal loss

    Args:
        pred: [N, A, C+1]
        target: [N, A]  (-1: ignore, 0: background, [1,C]: classes)
        reduction: 'sum', 'mean', 'none'
    Returns:
        reduced loss 
    """
    alpha = 0.25
    gamma = 2.0
    num_classes = pred.size(-1)
    pred = pred.view(-1, num_classes)
    target = target.view(-1)
    r = torch.arange(pred.size(0))
    ce = F.log_softmax(pred, dim=-1)[r, target.clamp_(0)]
    pt = torch.exp(ce)
    weights = (1 - pt).pow(gamma)

    # alpha version
    # p = y > 0
    # weights = (alpha * p + (1 - alpha) * (1 - p)) * weights.pow(gamma)

    loss = -(weights * ce)
    loss[target < 0] = 0
    return reduce(loss, reduction)


def smooth_l1_loss(pred, target, beta=0.11, reduction='sum'):
    """ smooth l1 loss

    Args:
        pred: positive anchors predictions [N, 4]
        target: positive anchors targets   [N, 4]
        beta: limit between l2 and l1 behavior
    """
    x = (pred - target).abs()
    l1 = x - 0.5 * beta
    l2 = (0.5 * x ** 2 / beta).to(l1)
    reg_loss = torch.where(x >= beta, l1, l2)
    return reduce(reg_loss, reduction)


class DetectionLoss(nn.Module):
    """Loss for Detection following SSD.

    This class returns 2 losses: 
    - one for anchor classification 
    - one for anchor refinement.

    Args:
        cls_loss_func (str): classification type loss
    """

    def __init__(self, cls_loss_func='softmax_focal_loss'):
        super(DetectionLoss, self).__init__()
        self.cls_loss_func = globals()[cls_loss_func]
        self.reg_loss_func = smooth_l1_loss

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        pos = cls_targets > 0
        num_pos = max(1, pos.sum().item())
        cls_loss = self.cls_loss_func(cls_preds, cls_targets, 'sum') / num_pos

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        loc_loss = self.reg_loss_func(loc_preds[mask], loc_targets[mask].to(loc_preds), reduction='sum') / num_pos
        return {"loc_loss": loc_loss, "cls_loss": cls_loss}
