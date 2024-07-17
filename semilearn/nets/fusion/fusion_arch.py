import torch
import torch.nn as nn
import torch.nn.functional as F
from ..vit.vit import Mlp


class EarlyFusionArch(nn.Module):
    def __init__(
            self, 
            joint_backbone,
            student_backbone, 
            num_classes,
            feat_dim, 
            metainfo_in_features,
            metainfo_dropout=0.0
        ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.metainfo_in_features = metainfo_in_features
        self.metainfo_dropout = metainfo_dropout

        self.joint_backbone = joint_backbone
        self.student_backbone = student_backbone
        self.metainfo_enc = Mlp(
            in_features=metainfo_in_features,
            hidden_features=self.feat_dim,
            out_features=self.feat_dim,
            act_layer=nn.GELU,
            drop=metainfo_dropout
        )
        self.student_head = nn.Linear(self.feat_dim, self.num_classes)
        self.joint_head = nn.Linear(self.feat_dim, self.num_classes)


    def forward(self, x, metainfo=None):
        if metainfo is None:
            return self.forward_student(x)
        else:
            return self.forward_joint(x, metainfo)
    
    def forward_joint(self, x, metainfo):
        metainfo_embed = self.metainfo_enc(metainfo) # [B,C]
        feat = self.joint_backbone.extract(x, metainfo_embed) # [B,1+P+1,C] if vit
        if feat.ndim == 4: # [B,C,H,W]
            # do avg pooling to get rid of spatial dimensions
            global_feat = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
            meta_feat = metainfo_embed # TODO: this is not a valid replacement
        elif feat.ndim == 3: # [B,1+N+1,C]
            global_feat = feat[:,0]
            meta_feat = feat[:,-1]
        else:
            raise ValueError(f'Expected 3 or 4 dimensional features, got {feat.shape}!')
        
        joint_logits = self.joint_head(global_feat)
        result_dict = {
            'joint_logits': joint_logits,
            'joint_feat': feat,
            'joint_global_feat': global_feat,
            'joint_meta_feat': meta_feat,
            'metainfo_embed': metainfo_embed
        }
        return result_dict
    
    def forward_student(self, x):
        feat = self.student_backbone.extract(x)
        if feat.ndim == 4: # [B,C,H,W]
            # do avg pooling to get rid of spatial dimensions
            global_feat = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
            meta_feat = None
        elif feat.ndim == 3: # [B,1+N+1,C]
            global_feat = feat[:,0]
            meta_feat = feat[:,-1]
        else:
            raise ValueError(f'Expected 3 or 4 dimensional features, got {feat.shape}!')
        
        student_logits = self.student_head(global_feat)
        result_dict = {
            'student_logits': student_logits,
            'student_feat': feat,
            'student_global_feat': global_feat,
            'student_meta_feat': meta_feat
        }
        return result_dict
