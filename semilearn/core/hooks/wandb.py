# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import wandb
from .hook import Hook


class WANDBHook(Hook):
    """
    Wandb Hook
    """

    def __init__(self, log_key_list):
        super().__init__()
        self.log_key_list = log_key_list

    def before_run(self, algorithm):
        # job_id = '_'.join(algorithm.args.save_name.split('_')[:-1])
        name = algorithm.save_name
        project = 'ssl' #algorithm.save_dir.split('/')[-1]

        # tags
        benchmark = f'benchmark: {project}'
        dataset = f'dataset: {algorithm.args.dataset}'
        data_setting = f'setting: {algorithm.args.dataset}_lb{algorithm.args.num_labels}_{algorithm.args.lb_imb_ratio}_ulb{algorithm.args.ulb_num_labels}_{algorithm.args.ulb_imb_ratio}'
        alg = f'alg: {algorithm.args.algorithm}'
        imb_alg = f'imb_alg: {algorithm.args.imb_algorithm}'
        tags = [benchmark, dataset, data_setting, alg, imb_alg] 
        # if algorithm.args.resume:
        #     resume = 'auto'
        # else:
        #     resume = 'never'
        resume = 'never'
        
        save_dir = os.path.join(algorithm.args.save_dir, algorithm.args.save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.run = wandb.init(name=name, 
                              tags=tags, 
                              config=algorithm.args.__dict__, 
                              project=project, 
                              resume=resume,
                              dir=save_dir)


    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            log_dict = {}
            for key, item in algorithm.log_dict.items():
                if key in self.log_key_list:
                    log_dict[key] = item
            self.run.log(log_dict, step=algorithm.it)
    
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            if hasattr(algorithm, 'best_eval_acc'):
                self.run.log({'eval/best-acc': algorithm.best_eval_acc}, step=algorithm.it)
            elif hasattr(algorithm, 'best_eval_ap'):
                self.run.log({'eval/best_AP': algorithm.best_eval_ap}, step=algorithm.it)
    
    def after_run(self, algorithm):
        self.run.log(algorithm.test_dict) #, step=algorithm.it)
        self.run.finish()