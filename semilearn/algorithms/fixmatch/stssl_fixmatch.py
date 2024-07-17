import torch
from semilearn.core.stssl_algorithmbase import STSSLAlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.core.criterions import DistillationLoss
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('stssl_fixmatch')
class STSSLFixMatch(STSSLAlgorithmBase):
    """
    Pseudolabeller uses meta info in addition to images.
    A second student network only uses images.
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(
            T=args.T, 
            p_cutoff=args.p_cutoff, 
            hard_label=args.hard_label,
            lambda_d=args.lambda_d,
            distill_loss_type=args.distill_loss_type
        )
    
    def init(self, T, p_cutoff, hard_label=True, lambda_d=1.0, distill_loss_type='mse'):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.lambda_d = lambda_d
        self.distill_loss_type = distill_loss_type
        self.distill_loss = DistillationLoss(type=distill_loss_type)
        # student should use meta token only if distillation is used
        # TODO: this is quite a hack, ignored for other architectures
        if self.distill_loss.loss_fn is not None and self.lambda_d > 0:
            self.model.student_backbone.always_meta_token = True
        else:
            self.model.student_backbone.always_meta_token = False

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, metainfo_lb, metainfo_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                metainfo = torch.cat((metainfo_lb, metainfo_ulb, metainfo_ulb))
                joint_outputs = self.model.forward_joint(inputs, metainfo)
                logits_x_lb = joint_outputs['joint_logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = joint_outputs['joint_logits'][num_lb:].chunk(2)
                feats_x_lb = joint_outputs['joint_meta_feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = joint_outputs['joint_meta_feat'][num_lb:].chunk(2)
                # visual-only student
                student_outputs = self.model.forward_student(torch.cat((x_lb, x_ulb_s)))
                student_logits_x_lb = student_outputs['student_logits'][:num_lb]
                student_logits_x_ulb_s = student_outputs['student_logits'][num_lb:]#.chunk(2)
                student_feats_x_lb = student_outputs['student_meta_feat'][:num_lb]
                student_feats_x_ulb_s = student_outputs['student_meta_feat'][num_lb:]#.chunk(2)
            else:
                raise NotImplementedError()
            
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            student_sup_loss = self.ce_loss(student_logits_x_lb, y_lb, reduction='mean')

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, compute_prob_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          compute_prob=True)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce' if not self.multilabel else 'bce',
                                               mask=mask)
            student_unsup_loss = self.consistency_loss(student_logits_x_ulb_s,
                                                       pseudo_label,
                                                       'ce' if not self.multilabel else 'bce',
                                                       mask=mask)
            
            distill_loss = self.distill_loss(
                teacher_feat_lb=feats_x_lb,
                teacher_feat_ulb=feats_x_ulb_s,
                student_feat_lb=student_feats_x_lb,
                student_feat_ulb=student_feats_x_ulb_s
            )

            total_loss = sup_loss + self.lambda_u * unsup_loss
            student_total_loss = student_sup_loss + self.lambda_u * student_unsup_loss + self.lambda_d * distill_loss

        out_dict = self.process_out_dict(
            loss=total_loss,
            student_loss=student_total_loss, 
            feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         student_sup_loss=student_sup_loss.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         student_unsup_loss=student_unsup_loss.item(),
                                         total_loss=total_loss.item(), 
                                         student_total_loss=student_total_loss.item(),
                                         util_ratio=mask.float().mean().item(),
                                         student_distill_loss=distill_loss.item()
                                         )
        return out_dict, log_dict


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
