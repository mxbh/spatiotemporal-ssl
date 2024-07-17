import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.criterions import AsymmetricLoss
from semilearn.core.utils import ALGORITHMS
from .utils import CAPPseudoLabelingHook


@ALGORITHMS.register('cap')
class CAP(AlgorithmBase):
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
            disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss
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
            disable_torch_grad_focal_loss=True
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

    def set_hooks(self):
        self.register_hook(CAPPseudoLabelingHook(num_classes=self.num_classes, momentum=self.args.ema_t), "PseudoLabelingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                if not self.pseudolabel_from_ema:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    outputs = self.model(inputs)
                    logits_x_lb = outputs['logits'][:num_lb]
                    logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                    feats_x_lb = outputs['feat'][:num_lb]
                    feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
                else:
                    inputs = torch.cat((x_lb, x_ulb_s))
                    outputs = self.model(inputs)
                    logits_x_lb = outputs['logits'][:num_lb]
                    logits_x_ulb_s = outputs['logits'][num_lb:]
                    feats_x_lb = outputs['feat'][:num_lb]
                    feats_x_ulb_s = outputs['feat'][num_lb:]
                    with torch.no_grad():
                        ema_outputs = self.ema_model(x_ulb_w)
                        logits_x_ulb_w = ema_outputs['logits']
                        feats_x_ulb_w = ema_outputs['feat']
            else:
                raise NotImplementedError()
            
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            pseudo_label, mask = self.call_hook("gen_targets_and_mask", "PseudoLabelingHook",
                                                logits=logits_x_ulb_w.detach(),
                                                )

            unsup_loss = (self.ce_loss(logits_x_ulb_s, pseudo_label, reduction='none') * mask).sum() / (mask.sum() + 1e-8)
                                        

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict