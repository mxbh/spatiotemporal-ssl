# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn
from torch.nn import functional as F


def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits, targets, reduction)


class BCELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        mask = (targets != -1).to(targets.dtype) # 0:ignore, 1:non-ignore
        losses = F.binary_cross_entropy_with_logits(logits, targets, weight=mask, reduction='none')
        
        if reduction == 'none':
            return losses
        elif reduction == 'sum':
            return losses.sum()
        elif reduction == 'mean':
            return losses.sum() / mask.sum()
        else:
            raise ValueError(f'Unknown reduction: {reduction}, must be one of ["none", "sum", "mean"].')


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, logits, targets, reduction='none'):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = probs_pos * targets
            pt1 = probs_neg * (1 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        if reduction == 'sum':
            loss = -loss.sum()
        elif reduction == 'mean':
            loss = -loss.mean()
        elif reduction == 'none':
            loss = -loss
        else:
            raise ValueError(f'Unknown reduction: {reduction}!')
            
        return loss
