# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook


def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val


def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


def binary_entropy_loss(mask, logits_s, prob_model, label_hist):
    # select samples
    mask_sum = mask.sum(dim=0)
    prob_s = logits_s.sigmoid() # logits_s.softmax(dim=-1)
    pred_label_s = (prob_s >= 0.5).int() # torch.max(prob_s, dim=-1)

    hist_s = (pred_label_s * mask).sum(dim=0) / mask_sum

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    prob_model_scaler_1 = replace_inf_to_zero(1 / label_hist).detach()
    prob_model_scaler_0 = replace_inf_to_zero(1 / (1 - label_hist)).detach()
    mod_prob_model_1 = prob_model * prob_model_scaler_1
    mod_prob_model_0 = (1 - prob_model) * prob_model_scaler_0
    mod_prob_model = mod_prob_model_1 / (mod_prob_model_1 + mod_prob_model_0)

    # modulate mean prob
    mean_prob_scaler_s_1 = replace_inf_to_zero(1 / hist_s).detach()
    mean_prob_scaler_s_0 = replace_inf_to_zero(1 / (1 - hist_s)).detach()
    mod_mean_prob_s_1 = (prob_s * mask).sum(dim=0, keepdim=True) / mask_sum * mean_prob_scaler_s_1
    mod_mean_prob_s_0 = ((1 - prob_s) * mask).sum(dim=0, keepdim=True) / mask_sum * mean_prob_scaler_s_0
    mod_mean_prob_s = mod_mean_prob_s_1 / (mod_mean_prob_s_1 + mod_mean_prob_s_0)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12) + (1 - mod_prob_model) * torch.log(1 - mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


class FreeMatchThresholdingHook(MaskingHook):
    """
    SAT in FreeMatch
    """
    def __init__(self, num_classes, multilabel=False, momentum=0.999,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.m = momentum
        
        if not self.multilabel:
            self.p_model = torch.ones((self.num_classes)) / self.num_classes
            self.label_hist = torch.ones((self.num_classes)) / self.num_classes
            self.time_p = self.p_model.mean()
        else:
            self.p_model = torch.ones((self.num_classes)) / 2
            self.label_hist = torch.ones((self.num_classes)) / 2
            self.time_p = torch.ones((self.num_classes)) / 2
    
    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)

        if not self.multilabel:
            max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)
        else:
            max_probs = torch.maximum(probs_x_ulb, 1 - probs_x_ulb)

        if algorithm.use_quantile:
            if self.multilabel:
                raise not NotImplementedError()
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            if not self.multilabel:
                self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
            else:
                self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean(dim=0)
        
        if algorithm.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)

        if not self.multilabel:
            hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
            self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())
        else:
            hist = (probs_x_ulb >= 0.5).float().mean(dim=0)
            self.label_hist = self.label_hist * self.m + (1 - self.m) * hist        

        algorithm.p_model = self.p_model 
        algorithm.label_hist = self.label_hist 
        algorithm.time_p = self.time_p 
    

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, compute_prob_x_ulb=True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if compute_prob_x_ulb:
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        if not self.multilabel:
            max_probs, max_idx = probs_x_ulb.max(dim=-1) # [N], [N]
            mod = self.p_model / torch.max(self.p_model, dim=-1)[0] # [C]
            mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)  # [N] = [1] * [N]
        else:
            max_probs = torch.maximum(probs_x_ulb, 1 - probs_x_ulb) # [N,C]
            max_idx = (probs_x_ulb >= 0.5).int() # [N,C]
            mod_1 =      self.p_model  / torch.maximum(self.p_model, 1 - self.p_model) # [C]
            mod_0 = (1 - self.p_model) / torch.maximum(self.p_model, 1 - self.p_model) # [C]
            mask = max_probs.ge(self.time_p * (max_idx * mod_1 + (1 - max_idx) * mod_0)) # [N,C] = [C] * [N,C]
            
        return mask
