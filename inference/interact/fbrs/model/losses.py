import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import misc


class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, size_average=True, detach_delimeter=True,
                 eps=1e-12, scale=1.0,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._k_sum = 0

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(one_hot, pred, 1 - pred)
        pt = torch.where(sample_weight, pt, torch.ones_like(pt))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult

        ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
        sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
        if np.any(ignore_area == 0):
            self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)

        one_hot = label > 0
        pt = torch.where(one_hot, pred, 1 - pred)

        t = label != -1
        alpha = torch.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        sample_weight = label != -1

        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(label == 1, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))
