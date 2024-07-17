import torch 
import torch.nn as nn 
from torch.nn import functional as F

def cosine_dissim(a, b):
    return -F.cosine_similarity(a, b).mean()

class DistillationLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        if type == 'none':
            self.loss_fn = None
        elif type == 'mse':
            self.loss_fn = F.mse_loss
        elif type == 'mae':
            self.loss_fn = F.l1_loss
        elif type == 'cos':
            self.loss_fn = cosine_dissim
        else:
            raise NotImplementedError(f'Unknown distillation loss: {type}!')

    def forward(self, teacher_feat_lb, teacher_feat_ulb, student_feat_lb, student_feat_ulb):
        if self.loss_fn is None:
            return torch.zeros(1, dtype=student_feat_lb.dtype, device=student_feat_lb.device)
        
        teacher_feat = torch.cat((teacher_feat_lb, teacher_feat_ulb), dim=0).detach()
        student_feat = torch.cat((student_feat_lb, student_feat_ulb), dim=0)
        loss = self.loss_fn(student_feat, teacher_feat)
        return loss
        