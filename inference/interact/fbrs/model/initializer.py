import torch
import torch.nn as nn
import numpy as np


class Initializer(object):
    def __init__(self, local_init=True, gamma=None):
        self.local_init = local_init
        self.gamma = gamma

    def __call__(self, m):
        if getattr(m, '__initialized', False):
            return

        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                          nn.GroupNorm, nn.SyncBatchNorm)) or 'BatchNorm' in m.__class__.__name__:
            if m.weight is not None:
                self._init_gamma(m.weight.data)
            if m.bias is not None:
                self._init_beta(m.bias.data)
        else:
            if getattr(m, 'weight', None) is not None:
                self._init_weight(m.weight.data)
            if getattr(m, 'bias', None) is not None:
                self._init_bias(m.bias.data)

        if self.local_init:
            object.__setattr__(m, '__initialized', True)

    def _init_weight(self, data):
        nn.init.uniform_(data, -0.07, 0.07)

    def _init_bias(self, data):
        nn.init.constant_(data, 0)

    def _init_gamma(self, data):
        if self.gamma is None:
            nn.init.constant_(data, 1.0)
        else:
            nn.init.normal_(data, 1.0, self.gamma)

    def _init_beta(self, data):
        nn.init.constant_(data, 0)


class Bilinear(Initializer):
    def __init__(self, scale, groups, in_channels, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.groups = groups
        self.in_channels = in_channels

    def _init_weight(self, data):
        """Reset the weight and bias."""
        bilinear_kernel = self.get_bilinear_kernel(self.scale)
        weight = torch.zeros_like(data)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            weight[i, j] = bilinear_kernel
        data[:] = weight

    @staticmethod
    def get_bilinear_kernel(scale):
        """Generate a bilinear upsampling kernel."""
        kernel_size = 2 * scale - scale % 2
        scale = (kernel_size + 1) // 2
        center = scale - 0.5 * (1 + kernel_size % 2)

        og = np.ogrid[:kernel_size, :kernel_size]
        kernel = (1 - np.abs(og[0] - center) / scale) * (1 - np.abs(og[1] - center) / scale)

        return torch.tensor(kernel, dtype=torch.float32)


class XavierGluon(Initializer):
    def __init__(self, rnd_type='uniform', factor_type='avg', magnitude=3, **kwargs):
        super().__init__(**kwargs)

        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)

    def _init_weight(self, arr):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(arr)

        if self.factor_type == 'avg':
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == 'in':
            factor = fan_in
        elif self.factor_type == 'out':
            factor = fan_out
        else:
            raise ValueError('Incorrect factor type')
        scale = np.sqrt(self.magnitude / factor)

        if self.rnd_type == 'uniform':
            nn.init.uniform_(arr, -scale, scale)
        elif self.rnd_type == 'gaussian':
            nn.init.normal_(arr, 0, scale)
        else:
            raise ValueError('Unknown random type')
