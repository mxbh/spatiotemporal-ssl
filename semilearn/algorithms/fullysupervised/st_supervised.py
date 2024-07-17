import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
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
from semilearn.core.utils import ALGORITHMS
from semilearn.core.stssl_algorithmbase import STSSLAlgorithmBase


@ALGORITHMS.register('st_supervised')
class STSupervised(STSSLAlgorithmBase):

    def train_step(self, x_lb, y_lb, metainfo_lb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outputs = self.model(x_lb, metainfo_lb)
            logits_x_lb = outputs['logits']
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

        out_dict = self.process_out_dict(loss=sup_loss) 
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())
        return out_dict, log_dict


    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()
        self.print_fn('Evaluating supervised model with meta input!')

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
        with torch.no_grad():
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

                logits = self.model(x, metainfo)[out_key]

                if not self.multilabel:
                    loss = F.cross_entropy(logits, y, reduction="mean", ignore_index=-1)
                    y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
                    y_pred.append((logits > 0.5).cpu().int().numpy())

                y_true.extend(y.cpu().numpy())
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
