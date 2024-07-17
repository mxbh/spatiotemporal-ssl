import contextlib
import os
from collections import OrderedDict
from inspect import signature

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, classification_report
from datetime import datetime
from semilearn.core.criterions import CELoss, BCELoss, ConsistencyLoss
from semilearn.nets.fusion import EarlyFusionArch
from semilearn.core.utils import Bn_Controller
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda.amp import GradScaler, autocast
from .algorithmbase import AlgorithmBase

class STSSLAlgorithmBase(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        # common arguments
        self.args = args
        self.multilabel = args.multilabel if hasattr(args, 'multilabel') else False
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm
        self.patience_iters = args.patience_iters if hasattr(args, 'patience_iters') else None
        self.stop_early_now = False

        # common utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        # common model related parameters
        self.it = 0
        self.start_epoch = 0
        self.best_it = 0
        if not self.multilabel:
            self.best_eval_acc = 0.0
        else:
            self.best_eval_ap = 0.0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None

        # build dataset
        self.dataset_dict = self.set_dataset()

        # build data loader
        self.loader_dict = self.set_data_loader()

        # cv, nlp, speech builder different arguments
        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build supervised loss and unsupervised loss
        self.ce_loss = CELoss() if not self.multilabel else BCELoss()
        self.consistency_loss = ConsistencyLoss()

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks
        self.hooks_dict = OrderedDict()  # actual object to be used to call hooks
        self.set_hooks()

    def set_model(self):
        """
        initialize model
        """
        backbone_teacher = self.net_builder(
            num_classes=self.num_classes,
            pretrained=self.args.use_pretrain,
            pretrained_path=self.args.pretrain_path,
        )
        backbone_student = self.net_builder(
            num_classes=self.num_classes,
            pretrained=self.args.use_pretrain,
            pretrained_path=self.args.pretrain_path,
        )
        model = EarlyFusionArch(
            joint_backbone=backbone_teacher,
            student_backbone=backbone_student,
            num_classes=self.num_classes,
            feat_dim=self.args.feat_dim,
            metainfo_in_features=4,
            metainfo_dropout=0.,
        )
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.set_model()
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        super(STSSLAlgorithmBase, self).set_hooks()
        if self.args.use_wandb:
            self.hooks_dict['WANDBHook'].log_key_list.extend([
                'train/student_sup_loss', 'train/student_unsup_loss', 'train/student_total_loss', 'train/student_distill_loss'
            ])

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(
                self.loader_dict["train_lb"], self.loader_dict["train_ulb"]
            ):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(
                    **self.process_batch(**data_lb, **data_ulb)
                )
                self.call_hook("after_train_step")
                self.it += 1

                if self.stop_early_now:
                    break

            self.call_hook("after_train_epoch")

            if self.stop_early_now:
                break

        self.call_hook("after_run")

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        """
        evaluation function
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

    def process_metainfo(self, metainfo):
        latlng = torch.stack((metainfo['lat'] / 90, metainfo['lng'] / 180), dim=1).float().cuda(self.gpu)              
        day = [(datetime.strptime(d, '%Y-%m-%d %H:%M:%S').timetuple().tm_yday if isinstance(d, str) else float('nan')) \
            for d in metainfo['date']]
        relative_day = torch.tensor(day) / 365 * 2 * np.pi # ignore leap year
        date_enc = torch.stack((relative_day.sin(), relative_day.cos()), dim=1).float().cuda(self.gpu)
        enc = torch.cat((latlng, date_enc), dim=1)
        return enc

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE: **kwargs should have the same arguments to train_step function as keys to
        work properly.
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if arg not in input_args:
                continue

            if var is None:
                continue
            
            if arg.startswith('metainfo'):
                input_dict[arg] = self.process_metainfo(var)
                continue

            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict
