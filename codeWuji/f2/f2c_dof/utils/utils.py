import torch
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.4f}({:.2f}) '.format(self.avg, self.val)


class AverageMeterList(object):
    def __init__(self, len_):
        self.len_ = len_
        self.AML = [AverageMeter() for _ in range(len_)]
        self.reset()

    def reset(self):
        for AM in self.AML:
            AM.reset()

    def update(self, val_list, n=1):
        for val, AM in zip(val_list, self.AML):
            AM.update(val, n)

    def avg(self):
        return [AM.avg for AM in self.AML]

    def __repr__(self):
        res = ""
        for AM in self.AML:
            res += AM.__repr__()
        return res


class MyLoss(object):
    def __init__(self):
        self.nLL = nn.NLLLoss()

    def __call__(self, output, label, eta1, eta2):
        loss = self.nLL(output, label) * 1
        # loss += -torch.mean(torch.log(eta1)) * 1
        # loss += -torch.mean(torch.log(eta2)) * 1
        return loss
