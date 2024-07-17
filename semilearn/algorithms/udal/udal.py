import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('udal')
class UDAL(AlgorithmBase):
    """ Unifying Distribution Alignment as a Loss for Imbalanced Semi-supervised Learning
    """
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
        )
    
    def init(self, a_min, k, T, p_cutoff, ema_p=0.999, hard_label=True):
        self.a_min = a_min
        self.k = k
        self.T = T
        self.p_cutoff = p_cutoff
        self.ema_p = ema_p
        self.use_hard_label = hard_label
    
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
                                            p_target=p_target,), "DistAlignHook")
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

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                raise NotImplementedError()

            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            probs_x_lb = self.compute_prob(logits_x_lb.detach())

            self.call_hook("update_p", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach(), probs_x_lb=probs_x_lb.detach())
            p_model = self.hooks_dict['DistAlignHook'].p_model
            p_data = self.hooks_dict['DistAlignHook'].p_target

            p_data_adjusted = self.compute_adjustment_dist(p_data)
            p_model_adjusted = self.compute_adjustment_dist(p_model)
            sup_loss = self.ce_loss(logits_x_lb + p_data_adjusted.log(), y_lb, reduction='mean')

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

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
    
        return save_dict

    def load_model(self, load_path, strict=True):
        checkpoint = super().load_model(load_path, strict=strict)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
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