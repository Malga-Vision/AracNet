import os
import torch
from torch.nn.functional import one_hot
import random
import numpy as np

def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

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
        if n>0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def get_subgroup_masks(labels, num_classes=None,device="cuda:0"):
    labels = [label.to(device) for label in labels]
    nb_labels = len(labels)
    if num_classes is None:
        num_classes = (torch.max(labels).item()+1,)*nb_labels
    mask = one_hot(labels[0], num_classes=num_classes[0])[
        (..., )+(None,)*(nb_labels-1)]
    for i in range(1, nb_labels):
        mask_bias = one_hot(labels[i], num_classes=num_classes[i])
        for j in range(i):
            mask_bias = mask_bias.unsqueeze(1)
        mask_bias = mask_bias[(...,)+(None,)*(nb_labels-i-1)]
        mask = mask*mask_bias
    return (mask > 0).to(device)


class AverageMeterSubgroups(object):
    """Computes and stores the average and current value for a metric for each subgroup"""

    def __init__(self, size, val_type=torch.float32, device="cuda:0"):
        """
        nb_classes_per_label should be a tuple of the number of values taken by a specific label type
        for instance, if we have :
            - a target that can take 7 values,
            - bias 1 that can take 6 values,
            - bias 2 that can take 3 values,
        then nb_classes_per_label = (7,6,3)"""
        self.size = size
        self.val_type = val_type
        self.device = device
        self.reset()

    def reset(self):
        self.val = torch.zeros(size=self.size, dtype=self.val_type).to(self.device)
        self.avg = torch.zeros(size=self.size, dtype=self.val_type).to(self.device)
        self.std = torch.zeros(size=self.size, dtype=self.val_type).to(self.device)
        self.sum = torch.zeros(size=self.size, dtype=self.val_type).to(self.device)
        self.sq_sum = torch.zeros(size=self.size, dtype=self.val_type).to(self.device)
        self.count = torch.zeros(size=self.size, dtype=torch.long).to(self.device)

    def update(self, val, subgroup_tensor):
        self.val = val
        n = torch.sum(subgroup_tensor, dim=0).to(device=self.device)
        self.sum += val * n
        self.sq_sum += n * val**2
        self.count += n
        non_zero_indices = self.count != 0
        self.avg[non_zero_indices] = self.sum[non_zero_indices] / \
            self.count[non_zero_indices]
        self.std[non_zero_indices] = (
            self.sq_sum[non_zero_indices] / self.count[non_zero_indices] - self.avg[non_zero_indices]**2) ** 0.5


def accuracy_subgroup(output, target, subgroup_masks, topk=(1,), num_classes=10):
    with torch.no_grad():
        pred    = torch.argmax(output, dim=1)[(..., ) + (None, ) * (len(subgroup_masks.size()) - 1)]
        target  = target[(..., ) + (None, ) * (len(subgroup_masks.size()) - 1)]
        correct = pred.eq(target) * subgroup_masks
        nb_samples = torch.sum(subgroup_masks, dim=0, dtype=torch.long)
        res = torch.full_like(nb_samples, fill_value=0, dtype=torch.float32)
        res[nb_samples != 0] = torch.sum(correct, dim=0, dtype=torch.float32)[
            nb_samples != 0].mul_(100.0 / nb_samples[nb_samples != 0])
        return res


def add_dims_index(tensor: torch.Tensor, nb_dims: int, index: int):
    """ """
    for i in range(nb_dims):
        tensor = tensor.unsqueeze(index)
    return tensor


def regroup_by(subgroup_metric, alignment=("aligned", "aligned"), per_class=False):
    with torch.no_grad():
        # dim_to_sum = tuple(range(len(subgroup_metric.avg.size())))
        metric = subgroup_metric.avg
        count = subgroup_metric.count
        for i in range(len(alignment)):
            eye_i = torch.eye(n=metric.size(0), m=metric.size(i+1)).cuda()
            if len(alignment)>1:
                eye_i = (
                    add_dims_index(
                        add_dims_index(
                            eye_i, 
                            nb_dims=i, 
                            index=1),
                        nb_dims=len(alignment)-i-1, 
                        index=-1))
            if alignment[i] == "aligned":
                metric = metric*eye_i
                count = count*eye_i
            elif alignment[i] == "misaligned":
                metric = metric*(1-eye_i)
                count = count*(1-eye_i)
            # print("acc = ", metric[count!=0])
            # print("acc*count", metric[count!=0]*count[count!=0])
        if per_class:
            acc = torch.sum(                metric*count, dim=tuple(range(1, len(metric.size()))))
            count = torch.sum(count,      dim=tuple(
                range(1, len(metric.size()))))
        else:
            # , dim=tuple(range(len(metric.size()))))
            acc = torch.sum(metric*count)
            # ,      dim=tuple(range(len(metric.size()))))
            count = torch.sum(count)
        return acc/count, count 
