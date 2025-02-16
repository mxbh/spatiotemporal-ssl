import torch
from semilearn.core.stssl_algorithmbase import STSSLAlgorithmBase
from semilearn.core.criterions import AsymmetricLoss
from semilearn.core.utils import ALGORITHMS
from semilearn.core.criterions import DistillationLoss
from .utils import CAPPseudoLabelingHook


@ALGORITHMS.register('stssl_cap')
class STSSLCAP(STSSLAlgorithmBase):
    '''
    CAP is naturally a multilabel SSL algorithm.
    https://papers.neurips.cc/paper_files/paper/2023/file/5195825ee60d7efc1e42b7f3f3137040-Paper-Conference.pdf
    '''

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        if not args.multilabel:
            raise ValueError('CAP is only designed for multilabel problems.')
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(
            pseudolabel_from_ema=args.pseudolabel_from_ema, 
            ema_t=args.ema_t,
            pos_per=args.pos_per,
            neg_per=args.neg_per,
            use_asl=args.use_asl,
            gamma_neg=args.gamma_neg,
            gamma_pos=args.gamma_pos,
            clip=args.clip,
            eps=args.eps,
            disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss,
            lambda_d=args.lambda_d,
            distill_loss_type=args.distill_loss_type
        )
    
    def init(
            self, 
            pseudolabel_from_ema=True,
            ema_t=0.999,
            pos_per=1.0,
            neg_per=1.0,
            use_asl=True,
            gamma_neg=4, 
            gamma_pos=1, 
            clip=0.05, 
            eps=1e-8, 
            disable_torch_grad_focal_loss=True, 
            lambda_d=1.0, 
            distill_loss_type='mse'
        ):
        self.pseudolabel_from_ema = pseudolabel_from_ema
        self.ema_t = ema_t
        self.pos_label_freq = torch.from_numpy(self.loader_dict['train_lb'].dataset.df[f'multi_hot_labels{self.num_classes}'].mean(axis=0))
        self.neg_label_freq = 1 - self.pos_label_freq
        self.pos_per = pos_per
        self.neg_per = neg_per
        self.pseudolabel_from_ema = pseudolabel_from_ema
        if use_asl:
            self.ce_loss = AsymmetricLoss(
                gamma_neg=gamma_neg, 
                gamma_pos=gamma_pos, 
                clip=clip, eps=eps, 
                disable_torch_grad_focal_loss=disable_torch_grad_focal_loss
            )
        self.use_hard_label = True # for compatibility
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
        self.register_hook(CAPPseudoLabelingHook(num_classes=self.num_classes, momentum=self.args.ema_t), "PseudoLabelingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, metainfo_lb, metainfo_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                if not self.pseudolabel_from_ema:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    metainfo = torch.cat((metainfo_lb, metainfo_ulb, metainfo_ulb))
                    joint_outputs = self.model.forward_joint(inputs, metainfo)
                    logits_x_lb = joint_outputs['joint_logits'][:num_lb]
                    logits_x_ulb_w, logits_x_ulb_s = joint_outputs['joint_logits'][num_lb:].chunk(2)
                    feats_x_lb = joint_outputs['joint_meta_feat'][:num_lb]
                    feats_x_ulb_w, feats_x_ulb_s = joint_outputs['joint_meta_feat'][num_lb:].chunk(2)
                else:
                    inputs = torch.cat((x_lb, x_ulb_s))
                    metainfo = torch.cat((metainfo_lb, metainfo_ulb))
                    joint_outputs = self.model.forward_joint(inputs, metainfo)
                    logits_x_lb = joint_outputs['joint_logits'][:num_lb]
                    logits_x_ulb_s = joint_outputs['joint_logits'][num_lb:]
                    feats_x_lb = joint_outputs['joint_meta_feat'][:num_lb]
                    feats_x_ulb_s = joint_outputs['joint_meta_feat'][num_lb:]
                    with torch.no_grad():
                        ema_outputs = self.ema_model.forward_joint(x_ulb_w, metainfo_ulb)
                        logits_x_ulb_w = ema_outputs['joint_logits']
                        feats_x_ulb_w = ema_outputs['joint_feat']
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

            pseudo_label, mask = self.call_hook("gen_targets_and_mask", "PseudoLabelingHook",
                                                logits=logits_x_ulb_w.detach(),
                                                )

            unsup_loss = (self.ce_loss(logits_x_ulb_s, pseudo_label, reduction='none') * mask).sum() / (mask.sum() + 1e-8)
            student_unsup_loss = (self.ce_loss(student_logits_x_ulb_s, pseudo_label, reduction='none') * mask).sum() / (mask.sum() + 1e-8)                   

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