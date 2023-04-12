"""
/*****************************************************************************/

BatchNorm2dSync with multi-gpu

/*****************************************************************************/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    # python 3
    from queue import Queue
except ImportError:
    # python 2
    from Queue import Queue

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from isegm.model.syncbn.modules.functional import batchnorm2d_sync


class _BatchNorm(nn.Module):
    """
    Customized BatchNorm from nn.BatchNorm
    >> added freeze attribute to enable bn freeze.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.freezed = False
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        compute_stats = not self.freezed and \
            self.training and self.track_running_stats

        ret = F.batch_norm(input, self.running_mean, self.running_var,
                           self.weight, self.bias, compute_stats,
                           self.momentum, self.eps)
        return ret

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, '\
               'affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(
                   **self.__dict__)


class BatchNorm2dNoSync(_BatchNorm):
    """
    Equivalent to nn.BatchNorm2d
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class BatchNorm2dSync(BatchNorm2dNoSync):
    """
    BatchNorm2d with automatic multi-GPU Sync
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2dSync, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self.sync_enabled = True
        self.devices = list(range(torch.cuda.device_count()))
        if len(self.devices) > 1:
            # Initialize queues
            self.worker_ids = self.devices[1:]
            self.master_queue = Queue(len(self.worker_ids))
            self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        compute_stats = not self.freezed and \
            self.training and self.track_running_stats
        if self.sync_enabled and compute_stats and len(self.devices) > 1:
            if x.get_device() == self.devices[0]:
                # Master mode
                extra = {
                    "is_master": True,
                    "master_queue": self.master_queue,
                    "worker_queues": self.worker_queues,
                    "worker_ids": self.worker_ids
                }
            else:
                # Worker mode
                extra = {
                    "is_master": False,
                    "master_queue": self.master_queue,
                    "worker_queue": self.worker_queues[
                        self.worker_ids.index(x.get_device())]
                }
            return batchnorm2d_sync(x, self.weight, self.bias,
                                    self.running_mean, self.running_var,
                                    extra, compute_stats, self.momentum,
                                    self.eps)
        return super(BatchNorm2dSync, self).forward(x)

    def __repr__(self):
        """repr"""
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
            'affine={affine}, ' \
            'track_running_stats={track_running_stats},' \
            'devices={devices})'
        return rep.format(name=self.__class__.__name__, **self.__dict__)

#BatchNorm2d = BatchNorm2dNoSync
BatchNorm2d = BatchNorm2dSync
