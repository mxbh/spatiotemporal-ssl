import torch
from semilearn.core.stssl_algorithmbase import STSSLAlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.core.criterions import DistillationLoss
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('stssl_udal')
class STSSLUDAL(STSSLAlgorithmBase):

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(
            a_min=args.a_min,
            k=args.k,
            T=args.T, 
            p_cutoff=args.p_cutoff, 
            ema_p=args.ema_p, 
            hard_label=args.hard_label,
            lambda_d=args.lambda_d,
            distill_loss_type=args.distill_loss_type
        )
    
    def init(self, a_min, k, T, p_cutoff, ema_p=0.999, hard_label=True, lambda_d=1.0, distill_loss_type='mse'):
        self.a_min = a_min
        self.k = k
        self.T = T
        self.p_cutoff = p_cutoff
        self.ema_p = ema_p
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
        self.register_hook(DistAlignEMAHook(num_classes=self.num_classes,
                                            multilabel=self.multilabel,
                                            momentum=self.args.ema_p,
                                            p_target_type=p_target_type,
                                            p_target=p_target), "StudentDistAlignHook")
        
        super().set_hooks()

    def compute_adjustment_dist(self, current_dist): 
        # p_data: class distribution of labeled data 
        # p_model: moving average of model’s predictions on unlabeled data
        # current_epoch (g): current epoch of training (out of max_epoch total)
        # k: rate at which a_min is approached 
        # a_min: minimum value of alpha for dist. alignment 
        factor = 1.0 - (1.0 - self.a_min) * (self.it / self.num_train_iter) ** self.k
        # normalize ensures the argument sums to 1 
        p_data = self.hooks_dict['DistAlignHook'].p_target
        if not self.multilabel:
            target_dist = p_data**factor / (p_data**factor).sum()
        else:
            target_dist = (p_data**factor) / ((1 - p_data)**factor + p_data**factor)
        return current_dist / (target_dist + 1e-9)
    
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

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            probs_x_lb = self.compute_prob(logits_x_lb.detach())
            student_probs_x_ulb_s = self.compute_prob(student_logits_x_ulb_s.detach())
            student_probs_x_lb = self.compute_prob(student_logits_x_lb.detach())

            self.call_hook("update_p", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach(), probs_x_lb=probs_x_lb.detach())
            p_model = self.hooks_dict['DistAlignHook'].p_model
            self.call_hook("update_p", "StudentDistAlignHook", probs_x_ulb=student_probs_x_ulb_s.detach(), probs_x_lb=student_probs_x_lb.detach())
            p_student = self.hooks_dict['DistAlignHook'].p_model
            p_data = self.hooks_dict['DistAlignHook'].p_target

            p_data_adjusted = self.compute_adjustment_dist(p_data)
            p_model_adjusted = self.compute_adjustment_dist(p_model)
            p_student_adjusted = self.compute_adjustment_dist(p_student)

            sup_loss = self.ce_loss(logits_x_lb + p_data_adjusted.log(), y_lb, reduction='mean')
            student_sup_loss = self.ce_loss(student_logits_x_lb + p_data_adjusted.log(), y_lb, reduction='mean')

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, compute_prob_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          compute_prob=True)

            unsup_loss = self.consistency_loss(logits_x_ulb_s + p_model_adjusted.log(),
                                               pseudo_label,
                                               'ce' if not self.multilabel else 'bce',
                                               mask=mask)
            student_unsup_loss = self.consistency_loss(student_logits_x_ulb_s + p_student_adjusted.log(),
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
        

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict['p_student'] = self.hooks_dict['StudentDistAlignHook'].p_model.cpu()
        save_dict['p_target_student'] = self.hooks_dict['StudentDistAlignHook'].p_model.cpu()
    
        return save_dict


    def load_model(self, load_path, strict=True):
        checkpoint = super().load_model(load_path, strict=strict)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.hooks_dict['StudentDistAlignHook'].p_model = checkpoint['p_student'].cuda(self.args.gpu)
        self.hooks_dict['StudentDistAlignHook'].p_target = checkpoint['p_target_student'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--ema_p', float, 0.999),
        ]