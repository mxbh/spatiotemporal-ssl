import math
import torch

from .utils import SoftMatchWeightingHook
from semilearn.core.stssl_algorithmbase import STSSLAlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.core.criterions import DistillationLoss
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('stssl_softmatch')
class STSSLSoftMatch(STSSLAlgorithmBase):
    """
    Pseudolabeller uses meta info in addition to images.
    A second student network only uses images.
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(
            T=args.T, 
            hard_label=args.hard_label, 
            dist_align=args.dist_align, 
            dist_uniform=args.dist_uniform, 
            ema_p=args.ema_p, 
            n_sigma=args.n_sigma, 
            per_class=args.per_class,
            lambda_d=args.lambda_d,
            distill_loss_type=args.distill_loss_type
        )
    
    def init(
            self, 
            T, 
            hard_label=True, 
            dist_align=True, 
            dist_uniform=True, 
            ema_p=0.999, 
            n_sigma=2, 
            per_class=False, 
            lambda_d=1.0, 
            distill_loss_type='mse'
        ):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p
        self.n_sigma = n_sigma
        self.per_class = per_class
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
        if not self.multilabel:
            p_target_type = 'uniform' if self.args.dist_uniform else 'model'
            p_target = None 
        else:
            # TODO: this is hardcoded for bigearthnet
            p_target_type = 'gt'
            p_target = torch.from_numpy(self.loader_dict['train_lb'].dataset.df[f'multi_hot_labels{self.num_classes}'].mean(axis=0))
        self.register_hook(DistAlignEMAHook(num_classes=self.num_classes,
                                            multilabel=self.multilabel,
                                            momentum=self.args.ema_p, 
                                            p_target_type=p_target_type, 
                                            p_target=p_target), "DistAlignHook")
        self.register_hook(SoftMatchWeightingHook(num_classes=self.num_classes,
                                                  multilabel=self.multilabel, 
                                                  n_sigma=self.args.n_sigma, 
                                                  momentum=self.args.ema_p, 
                                                  per_class=self.args.per_class), "MaskingHook")
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
                # student
                student_inputs = torch.cat((x_lb, x_ulb_s))
                student_outputs = self.model.forward_student(student_inputs) # no metainfo here
                student_logits_x_lb = student_outputs['student_logits'][:num_lb]
                student_logits_x_ulb_s = student_outputs['student_logits'][num_lb:]
                student_feats_x_lb = student_outputs['student_meta_feat'][:num_lb]
                student_feats_x_ulb_s = student_outputs['student_meta_feat'][num_lb:]
            else:
                raise NotImplementedError()

            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            student_sup_loss = self.ce_loss(student_logits_x_lb, y_lb, reduction='mean')

            probs_x_lb = self.compute_prob(logits_x_lb.detach())
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # uniform distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, compute_prob_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          # make sure this is logits, not dist aligned probs
                                          # uniform alignment in softmatch do not use aligned probs for generating pseudo labels
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          compute_prob=True,
                                          T=self.T)

            # calculate loss
            unsup_loss = self.consistency_loss(
                logits_x_ulb_s,
                pseudo_label,
                'ce' if not self.multilabel else 'bce',
                mask=mask
            )
            student_unsup_loss = self.consistency_loss(
                student_logits_x_ulb_s,
                pseudo_label,
                'ce' if not self.multilabel else 'bce',
                mask=mask
            )

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


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()
        return save_dict


    def load_model(self, load_path, strict=True):
        checkpoint = super().load_model(load_path, strict=strict)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
        ]