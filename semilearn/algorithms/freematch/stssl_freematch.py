import torch
import torch.nn.functional as F

from .utils import entropy_loss, binary_entropy_loss, FreeMatchThresholdingHook
from semilearn.core.stssl_algorithmbase import STSSLAlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.core.criterions import DistillationLoss
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

@ALGORITHMS.register('stssl_freematch')
class STSSLFreeMatch(STSSLAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(
            T=args.T, 
            hard_label=args.hard_label, 
            ema_p=args.ema_p, 
            use_quantile=args.use_quantile, 
            clip_thresh=args.clip_thresh,
            supervised_warmup_iters=args.sup_warmup_iters,
            lambda_d=args.lambda_d,
            distill_loss_type=args.distill_loss_type
        )
        self.lambda_e = args.ent_loss_ratio

    def init(
            self, 
            T, 
            hard_label=True, 
            ema_p=0.999, 
            use_quantile=True, 
            clip_thresh=False, 
            supervised_warmup_iters=0,
            lambda_d=1.0, 
            distill_loss_type='mse'
        ):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh
        self.supervised_warmup_iters = supervised_warmup_iters
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
        self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, 
                                                     multilabel=self.multilabel, 
                                                     momentum=self.args.ema_p), "MaskingHook")
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

            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          compute_prob=True)
            
            # calculate unlabeled loss
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
            
            # calculate entropy loss
            if mask.sum() > 0:
                if not self.multilabel:
                    ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
                    student_ent_loss, _ = entropy_loss(mask, student_logits_x_ulb_s, self.p_model, self.label_hist)
                else:
                    ent_loss, _ = binary_entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
                    student_ent_loss, _ = binary_entropy_loss(mask, student_logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
               student_ent_loss = 0.0

            distill_loss = self.distill_loss(
                teacher_feat_lb=feats_x_lb,
                teacher_feat_ulb=feats_x_ulb_s,
                student_feat_lb=student_feats_x_lb,
                student_feat_ulb=student_feats_x_ulb_s
            )

            if self.it < self.supervised_warmup_iters:
                total_loss = sup_loss
                student_total_loss = student_sup_loss
            else:
                total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss
                student_total_loss = student_sup_loss + self.lambda_u * student_unsup_loss + self.lambda_e * student_ent_loss + self.lambda_d * distill_loss
        
        out_dict = self.process_out_dict(
            loss=total_loss, 
            student_loss=student_total_loss,
            feat=feat_dict
        )
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
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        return save_dict


    def load_model(self, load_path, strict=True):
        checkpoint = super().load_model(load_path, strict=strict)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
        ]