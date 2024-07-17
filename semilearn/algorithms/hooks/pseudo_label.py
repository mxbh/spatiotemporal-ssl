# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from semilearn.core.hooks import Hook
from semilearn.algorithms.utils import smooth_targets

class PseudoLabelingHook(Hook):
    """
    Pseudo Labeling Hook
    """
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def gen_ulb_targets(self, 
                        algorithm, 
                        logits, 
                        use_hard_label=True, 
                        T=1.0,
                        compute_prob=True, # whether to compute softmax or sigmoid for logits, input must be logits
                        label_smoothing=0.0):
        
        """
        generate pseudo-labels from logits/probs

        Args:
            algorithm: base algorithm
            logits: logits (or probs, need to set softmax to False)
            use_hard_label: flag of using hard labels instead of soft labels
            T: temperature parameters
            softmax: flag of using softmax on logits
            label_smoothing: label_smoothing parameter
        """

        logits = logits.detach()
        if use_hard_label:
            # return hard label directly
            if not algorithm.multilabel:
                pseudo_label = torch.argmax(logits, dim=-1)
                if label_smoothing:
                    pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
            else:
                pseudo_label = (logits >= 0).to(logits.dtype)
                if label_smoothing:
                    smooth_label = torch.where(pseudo_label == 1, 1 - label_smoothing, label_smoothing)
                    pseudo_label = smooth_label.to(logits.dtype).to(logits.device)

            return pseudo_label
        
        # return soft label
        if compute_prob:
            # pseudo_label = torch.softmax(logits / T, dim=-1)
            pseudo_label = algorithm.compute_prob(logits / T)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits

        return pseudo_label
