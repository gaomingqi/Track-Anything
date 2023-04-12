"""
Group-specific modules
They handle features that also depends on the mask. 
Features are typically of shape
    batch_size * num_objects * num_channels * H * W

All of them are permutation equivariant w.r.t. to the num_objects dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_groups(g, ratio, mode, align_corners):
    batch_size, num_objects = g.shape[:2]
    g = F.interpolate(g.flatten(start_dim=0, end_dim=1), 
                scale_factor=ratio, mode=mode, align_corners=align_corners)
    g = g.view(batch_size, num_objects, *g.shape[1:])
    return g

def upsample_groups(g, ratio=2, mode='bilinear', align_corners=False):
    return interpolate_groups(g, ratio, mode, align_corners)

def downsample_groups(g, ratio=1/2, mode='area', align_corners=None):
    return interpolate_groups(g, ratio, mode, align_corners)


class GConv2D(nn.Conv2d):
    def forward(self, g):
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


class GroupResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = GConv2D(out_dim, out_dim, kernel_size=3, padding=1)
 
    def forward(self, g):
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        
        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g


class MainToGroupDistributor(nn.Module):
    def __init__(self, x_transform=None, method='cat', reverse_order=False):
        super().__init__()

        self.x_transform = x_transform
        self.method = method
        self.reverse_order = reverse_order

    def forward(self, x, g):
        num_objects = g.shape[1]

        if self.x_transform is not None:
            x = self.x_transform(x)

        if self.method == 'cat':
            if self.reverse_order:
                g = torch.cat([g, x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1)], 2)
            else:
                g = torch.cat([x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1), g], 2)
        elif self.method == 'add':
            g = x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1) + g
        else:
            raise NotImplementedError

        return g
