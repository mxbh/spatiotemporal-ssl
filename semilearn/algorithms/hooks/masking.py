
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.core.hooks import Hook


class MaskingHook(Hook):
    """
    Base MaskingHook, used for computing the mask of unlabeled (consistency) loss
    define MaskingHook in each algorithm when needed, and call hook inside each train_step
    easy support for other settings
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def update(self, *args, **kwargs):
        pass
    
    @torch.no_grad()
    def masking(self, algorithm, 
                      logits_x_lb=None, logits_x_ulb=None, idx_lb=None, idx_ulb=None,
                      softmax_x_lb=True, softmax_x_ulb=True,
                      *args, **kwargs):
        """
        generate mask for unlabeled loss

        Args:
            algorithm: base algorithm
            logits_x_lb: labeled batch logits (or probs, need to set softmax_x_lb to False)
            logits_x_ulb: unlabeled batch logits (or probs, need to set softmax_x_ulb to False)
            idx_lb: labeled batch index
            idx_ulb: unlabeled batch index
            softmax_x_lb: flag of using softmax on labeled logits 
            softmax_x_ulb: flag of using softmax on unlabeled logits 
        """
        raise NotImplementedError


class FixedThresholdingHook(MaskingHook):
    """
    Common Fixed Threshold used in fixmatch, uda, pseudo label, et. al.
    """

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, compute_prob_x_ulb=True, *args, **kwargs):
        if not algorithm.multilabel:
            if compute_prob_x_ulb:
                # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
                probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
            else:
                # logits is already probs
                probs_x_ulb = logits_x_ulb.detach()
            max_probs, _ = torch.max(probs_x_ulb, dim=-1)
            mask = max_probs.ge(algorithm.p_cutoff).to(max_probs.dtype)
            return mask
        else:
            if compute_prob_x_ulb:
                # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
                probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
            else:
                # logits is already probs
                probs_x_ulb = logits_x_ulb.detach()
            
            if isinstance(algorithm.p_cutoff, float):
                max_probs = torch.maximum(probs_x_ulb, (1 - probs_x_ulb))
                mask = max_probs.ge(algorithm.p_cutoff).to(max_probs.dtype)
            else:
                assert len(algorithm.p_cutoff) == 2, algorithm.p_cutoff # for now assume pos and neg threshold, later maybe include thresholds for different classes
                th0, th1 = algorithm.p_cutoff
                mask0 = probs_x_ulb.le(th0)
                mask1 = probs_x_ulb.ge(th1)
                mask = torch.logical_or(mask0, mask1).to(probs_x_ulb.dtype)
            return mask


# class RampupWeightingHook(MaskingHook):
#     """
#     Common Rampup weight used in mean teacher, pi model, et. al.
#     """
#     def masking(self, algorithm, *args, **kwargs):
#         return np.clip(algorithm.it / (algorithm.unsup_warm_up * algorithm.num_train_iter),  a_min=0.0, a_max=1.0)