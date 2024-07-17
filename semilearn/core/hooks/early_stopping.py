from .hook import Hook

class EarlyStoppingHook(Hook):

    def after_train_step(self, algorithm):
        if hasattr(algorithm, 'patience_iters') and algorithm.patience_iters is not None:
            if self.every_n_iters(algorithm, algorithm.num_eval_iter):
                steps_since_best_it = algorithm.it - algorithm.best_it
                if steps_since_best_it >= algorithm.patience_iters:
                    algorithm.stop_early_now = True
                    algorithm.print_fn('Stopping early.')

