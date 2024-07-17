# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.



import torch

from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook


class SoftMatchWeightingHook(MaskingHook):
    """
    SoftMatch learnable truncated Gaussian weighting
    """
    def __init__(self, num_classes, multilabel=False, n_sigma=2, momentum=0.999, per_class=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        if not self.multilabel:
            if not self.per_class:
                self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
                self.prob_max_var_t = torch.tensor(1.0)
            else:
                self.prob_max_mu_t = torch.ones((self.num_classes)) / self.args.num_classes
                self.prob_max_var_t =  torch.ones((self.num_classes))
        else:
            if not self.per_class:
                self.prob_max_mu_t = torch.full((self.num_classes,), fill_value=0.5) #torch.tensor(1.0 / self.num_classes)
                self.prob_max_var_t = torch.full((self.num_classes,), fill_value=1.0) #torch.tensor(1.0)
            else:
                self.prob_max_mu_t = torch.ones((2, self.num_classes)) / 2.
                self.prob_max_var_t =  torch.ones((2, self.num_classes))

    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = self.concat_all_gather(probs_x_ulb)
        if not self.multilabel:
            max_probs, max_idx = probs_x_ulb.max(dim=-1)
            if not self.per_class:
                prob_max_mu_t = torch.mean(max_probs) # torch.quantile(max_probs, 0.5)
                prob_max_var_t = torch.var(max_probs, unbiased=True)
                self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
                self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
            else:
                prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
                prob_max_var_t = torch.ones_like(self.prob_max_var_t)
                for i in range(self.num_classes):
                    prob = max_probs[max_idx == i]
                    if len(prob) > 1:
                        prob_max_mu_t[i] = torch.mean(prob)
                        prob_max_var_t[i] = torch.var(prob, unbiased=True)
                self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
                self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        else:
            max_probs = torch.maximum(1 - probs_x_ulb, probs_x_ulb)
            max_idx = (probs_x_ulb >= 0.5).long()
            if not self.per_class: # classes are separately anyway now, but softmax classes correspond to 0 and 1
                prob_max_mu_t = torch.mean(max_probs, dim=0) # torch.quantile(max_probs, 0.5)
                prob_max_var_t = torch.var(max_probs, dim=0, unbiased=True)
                self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t #.item()
                self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t #.item()
            else:
                prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
                prob_max_var_t = torch.ones_like(self.prob_max_var_t)
                for i in range(self.num_classes):
                    prob_0 = max_probs[:,i][max_idx[:,i] == 0]
                    prob_1 = max_probs[:,i][max_idx[:,i] == 1]
                    if len(prob_0) > 1:
                        prob_max_mu_t[0,i] = torch.mean(prob_0)
                        prob_max_var_t[0,i] = torch.var(prob_0, unbiased=True)
                    if len(prob_1) > 1:
                        prob_max_mu_t[1,i] = torch.mean(prob_1)
                        prob_max_var_t[1,i] = torch.var(prob_1, unbiased=True)

                self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
                self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t

        return max_probs, max_idx
    
    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, compute_prob_x_ulb=True, *args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if compute_prob_x_ulb:
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        max_probs, max_idx = self.update(algorithm, probs_x_ulb)

        # compute weight
        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            if not self.multilabel:
                mu = self.prob_max_mu_t[max_idx]
                var = self.prob_max_var_t[max_idx]
            else:
                mu = torch.where(max_idx == 0, self.prob_max_mu_t[0], self.prob_max_mu_t[1]) #self.prob_max_mu_t[max_idx]
                var = torch.where(max_idx == 0, self.prob_max_var_t[0], self.prob_max_var_t[1]) #self.prob_max_var_t[max_idx]                
        
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        return mask
