import torch
from semilearn.core.hooks import Hook


class TensorQueue:
    def __init__(self, queue_length, d):
        self.len = queue_length
        self.d = d
        self.tensor = torch.empty(self.len, self.d, dtype=torch.float)
        self.pointer = 0
        self.queue_full = False

    def push(self, t):
        if not self.tensor.is_cuda and t.is_cuda:
            self.tensor = self.tensor.to(t.device)

        n = len(t)
        p0 = self.pointer
        p1 = self.pointer + n
        if p1 <= self.len:
            self.tensor[p0:p1] = t
            self.pointer = p1
        elif p1 > self.len:
            n1 = self.len - p0
            n2 = p1 - self.len
            self.tensor[p0:self.len] = t[:n1]
            self.tensor[0:n2] = t[n1:]
            self.pointer = n2
            self.queue_full = True

    def get_quantile(self, q):
        '''
        q is assumed to be a vector of size d.
        '''
        n = self.len if self.queue_full else self.pointer
        indices = torch.floor(n * q).clamp(0, n - 1).int()
        sorted_tensor = torch.sort(self.tensor[:n], dim=0, descending=False).values
        quantiles = sorted_tensor[indices, range(sorted_tensor.shape[1])]
        return quantiles


class CAPPseudoLabelingHook(Hook):
    def __init__(self, num_classes, momentum=0.999):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum
        queue_length = 64_000
        self.probs_queue = TensorQueue(queue_length=queue_length, d=self.num_classes)

    def get_thresholds(self, algorithm, probs):
        self.probs_queue.push(probs)
        thre_vec = self.probs_queue.get_quantile(1 - algorithm.pos_label_freq)
        pos_thre = self.probs_queue.get_quantile(1 - algorithm.pos_per * algorithm.pos_label_freq)
        neg_thre = self.probs_queue.get_quantile(algorithm.neg_per * algorithm.neg_label_freq)
        return thre_vec, pos_thre, neg_thre

    @torch.no_grad()
    def gen_targets_and_mask(self, algorithm, logits):
        probs = algorithm.compute_prob(logits.detach())
        thre_vec, pos_thre, neg_thre = self.get_thresholds(algorithm, probs)
        
        pseudo_labels = (probs >= thre_vec).to(probs.dtype)
        mask = torch.zeros_like(pseudo_labels)
        mask[probs >= pos_thre] = 1
        mask[probs <= neg_thre] = 1
        
        return pseudo_labels, mask