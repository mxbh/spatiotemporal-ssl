import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from semilearn.core.stssl_algorithmbase import STSSLAlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.core.criterions import DistillationLoss
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from sklearn.metrics import (
    average_precision_score, 
    f1_score, 
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

@ALGORITHMS.register('stssl_cdmad')
class STSSLCDMAD(STSSLAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(
            p_cutoff=args.p_cutoff, 
            debias_start=args.debias_start_iter,
            lambda_d=args.lambda_d,
            distill_loss_type=args.distill_loss_type
        )

    def init(self, p_cutoff, debias_start, lambda_d=1.0, distill_loss_type='mse'):
        self.p_cutoff = p_cutoff
        self.debias_start = debias_start
        self.use_hard_label = False
        self.lambda_d = lambda_d
        self.distill_loss_type = distill_loss_type
        self.distill_loss = DistillationLoss(type=distill_loss_type)
        # student should use meta token only if distillation is used
        # TODO: this is quite a hack, ignored for other architectures
        if self.distill_loss.loss_fn is not None and self.lambda_d > 0:
            self.model.student_backbone.always_meta_token = True
        else:
            self.model.student_backbone.always_meta_token = False
        # for determining bias, remove meta token from attention
        self.model.joint_backbone.always_meta_token = False
        

    def set_hooks(self):
        #self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def get_bias(self, x):
        with torch.no_grad():
            if self.it >= self.debias_start:
                white = torch.ones((1,)+ x.shape[1:], device=x.device)
                return self.model.joint_backbone.forward(white, metainfo=None)['logits'].detach()
            else:
                return torch.zeros((1, self.num_classes), device=x.device)
            
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
            
            bias = self.get_bias(x_lb)
            debiased_probs_x_ulb_w = self.compute_prob((logits_x_ulb_w - bias).detach())
            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=debiased_probs_x_ulb_w, compute_prob_x_ulb=False)
            unsup_loss = self.consistency_loss(
                logits=logits_x_ulb_s,
                targets=debiased_probs_x_ulb_w,
                name='ce' if not self.multilabel else 'bce',
                mask=mask
            )
            student_unsup_loss = self.consistency_loss(
                logits=student_logits_x_ulb_s,
                targets=debiased_probs_x_ulb_w,
                name='ce' if not self.multilabel else 'bce',
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
    

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        """
        custom evaluation function for CDMAD because of debiasing at test time
        """
        self.model.eval()
        self.ema.apply_shadow()

        meta_input = False
        if not meta_input:
            out_key = 'student_' + out_key
            self.print_fn(f'Evaluating student without meta input!')
        else:
            out_key = 'joint_' + out_key
            self.print_fn(f'Evaluating teacher with meta input!')

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
        with torch.no_grad():
            bias = None
            for data in tqdm(eval_loader, total=len(eval_loader)):
                x = data["x_lb"]
                y = data["y_lb"]
                metainfo = self.process_metainfo(data["metainfo_lb"])

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                if not meta_input:
                    logits = self.model(x)[out_key]
                else:
                    logits = self.model(x, metainfo)[out_key]

                if bias is None:
                    bias = self.get_bias(x)
                logits = logits - bias

                if not self.multilabel:
                    loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
                    y_pred.append(torch.max(logits, dim=-1)[1].cpu().numpy())
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
                    y_pred.append((logits > 0.5).cpu().int().numpy())
                y_true.append(y.cpu().numpy())
                y_logits.append(logits.cpu().numpy())
                total_loss += loss.item() * num_batch
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_logits = np.concatenate(y_logits, axis=0)

        if not self.multilabel:
            top1 = accuracy_score(y_true, y_pred)
            balanced_top1 = balanced_accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            F1 = f1_score(y_true, y_pred, average="macro")
            cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
            self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))
            eval_dict = {
                eval_dest + "/loss": total_loss / total_num,
                eval_dest + "/top-1-acc": top1,
                eval_dest + "/balanced_acc": balanced_top1,
                eval_dest + "/precision": precision,
                eval_dest + "/recall": recall,
                eval_dest + "/F1": F1,
            }
        else:
            y_prob = 1. / (1 + np.exp(-y_logits))
            ap = average_precision_score(y_true, y_prob, average="macro")
            ap_micro = average_precision_score(y_true, y_prob, average="micro")
            f1 = f1_score(y_true, y_pred, average="macro")
            f1_micro = f1_score(y_true, y_pred, average="micro")
            self.print_fn(classification_report(y_true, y_pred))
            eval_dict = {
                eval_dest + "/loss": total_loss / total_num,
                eval_dest + "/AP": ap,
                eval_dest + "/AP_micro": ap_micro,
                eval_dest + "/F1": f1,
                eval_dest + "/F1_micro": f1_micro,
            }  
        if return_logits:
            eval_dict[eval_dest + "/logits"] = y_logits

        self.ema.restore()
        self.model.train()
        return eval_dict