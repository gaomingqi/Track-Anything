import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

from mmcv.cnn import ConvModule
from mmengine.runner import load_checkpoint


class FlowCompletionLoss(nn.Module):
    """Flow completion loss"""
    def __init__(self):
        super().__init__()
        self.fix_spynet = SPyNet()
        for p in self.fix_spynet.parameters():
            p.requires_grad = False

        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, gt_local_frames):
        b, l_t, c, h, w = gt_local_frames.size()

        with torch.no_grad():
            # compute gt forward and backward flows
            gt_local_frames = F.interpolate(gt_local_frames.view(-1, c, h, w),
                                            scale_factor=1 / 4,
                                            mode='bilinear',
                                            align_corners=True,
                                            recompute_scale_factor=True)
            gt_local_frames = gt_local_frames.view(b, l_t, c, h // 4, w // 4)
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(
                -1, c, h // 4, w // 4)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(
                -1, c, h // 4, w // 4)
            gt_flows_forward = self.fix_spynet(gtlf_1, gtlf_2)
            gt_flows_backward = self.fix_spynet(gtlf_2, gtlf_1)

        # calculate loss for flow completion
        forward_flow_loss = self.l1_criterion(
            pred_flows[0].view(-1, 2, h // 4, w // 4), gt_flows_forward)
        backward_flow_loss = self.l1_criterion(
            pred_flows[1].view(-1, 2, h // 4, w // 4), gt_flows_backward)
        flow_loss = forward_flow_loss + backward_flow_loss

        return flow_loss


class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """
    def __init__(
        self,
        use_pretrain=True,
        pretrained='https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'
    ):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if use_pretrain:
            if isinstance(pretrained, str):
                print("load pretrained SPyNet...")
                load_checkpoint(self, pretrained, strict=True)
            elif pretrained is not None:
                raise TypeError('[pretrained] should be str or None, '
                                f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Note that in this function, the images are already resized to a
        multiple of 32.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(input=ref[-1],
                             kernel_size=2,
                             stride=2,
                             count_include_pad=False))
            supp.append(
                F.avg_pool2d(input=supp[-1],
                             kernel_size=2,
                             stride=2,
                             count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(input=flow,
                                        scale_factor=2,
                                        mode='bilinear',
                                        align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(supp[level],
                          flow_up.permute(0, 2, 3, 1).contiguous(),
                          padding_mode='border'), flow_up
            ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.
        This function computes the optical flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(input=ref,
                            size=(h_up, w_up),
                            mode='bilinear',
                            align_corners=False)
        supp = F.interpolate(input=supp,
                             size=(h_up, w_up),
                             mode='bilinear',
                             align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(input=self.compute_flow(ref, supp),
                             size=(h, w),
                             mode='bilinear',
                             align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """
    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(in_channels=8,
                       out_channels=32,
                       kernel_size=7,
                       stride=1,
                       padding=3,
                       norm_cfg=None,
                       act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=32,
                       out_channels=64,
                       kernel_size=7,
                       stride=1,
                       padding=3,
                       norm_cfg=None,
                       act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=64,
                       out_channels=32,
                       kernel_size=7,
                       stride=1,
                       padding=3,
                       norm_cfg=None,
                       act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=32,
                       out_channels=16,
                       kernel_size=7,
                       stride=1,
                       padding=3,
                       norm_cfg=None,
                       act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=16,
                       out_channels=2,
                       kernel_size=7,
                       stride=1,
                       padding=3,
                       norm_cfg=None,
                       act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].
        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x,
                           grid_flow,
                           mode=interpolation,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return output


def initial_mask_flow(mask):
    """
    mask 1 indicates valid pixel 0 indicates unknown pixel
    """
    B, T, C, H, W = mask.shape

    # calculate relative position
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    grid_y, grid_x = grid_y.type_as(mask), grid_x.type_as(mask)
    abs_relative_pos_y = H - torch.abs(grid_y[None, :, :] - grid_y[:, None, :])
    relative_pos_y = H - (grid_y[None, :, :] - grid_y[:, None, :])

    abs_relative_pos_x = W - torch.abs(grid_x[:, None, :] - grid_x[:, :, None])
    relative_pos_x = W - (grid_x[:, None, :] - grid_x[:, :, None])

    # calculate the nearest indices
    pos_up = mask.unsqueeze(3).repeat(
        1, 1, 1, H, 1, 1).flip(4) * abs_relative_pos_y[None, None, None] * (
            relative_pos_y <= H)[None, None, None]
    nearest_indice_up = pos_up.max(dim=4)[1]

    pos_down = mask.unsqueeze(3).repeat(1, 1, 1, H, 1, 1) * abs_relative_pos_y[
        None, None, None] * (relative_pos_y <= H)[None, None, None]
    nearest_indice_down = (pos_down).max(dim=4)[1]

    pos_left = mask.unsqueeze(4).repeat(
        1, 1, 1, 1, W, 1).flip(5) * abs_relative_pos_x[None, None, None] * (
            relative_pos_x <= W)[None, None, None]
    nearest_indice_left = (pos_left).max(dim=5)[1]

    pos_right = mask.unsqueeze(4).repeat(
        1, 1, 1, 1, W, 1) * abs_relative_pos_x[None, None, None] * (
            relative_pos_x <= W)[None, None, None]
    nearest_indice_right = (pos_right).max(dim=5)[1]

    # NOTE: IMPORTANT !!! depending on how to use this offset
    initial_offset_up = -(nearest_indice_up - grid_y[None, None, None]).flip(3)
    initial_offset_down = nearest_indice_down - grid_y[None, None, None]

    initial_offset_left = -(nearest_indice_left -
                            grid_x[None, None, None]).flip(4)
    initial_offset_right = nearest_indice_right - grid_x[None, None, None]

    # nearest_indice_x = (mask.unsqueeze(1).repeat(1, img_width, 1) * relative_pos_x).max(dim=2)[1]
    # initial_offset_x = nearest_indice_x - grid_x

    # handle the boundary cases
    final_offset_down = (initial_offset_down < 0) * initial_offset_up + (
        initial_offset_down > 0) * initial_offset_down
    final_offset_up = (initial_offset_up > 0) * initial_offset_down + (
        initial_offset_up < 0) * initial_offset_up
    final_offset_right = (initial_offset_right < 0) * initial_offset_left + (
        initial_offset_right > 0) * initial_offset_right
    final_offset_left = (initial_offset_left > 0) * initial_offset_right + (
        initial_offset_left < 0) * initial_offset_left
    zero_offset = torch.zeros_like(final_offset_down)
    # out = torch.cat([final_offset_left, zero_offset, final_offset_right, zero_offset, zero_offset, final_offset_up, zero_offset, final_offset_down], dim=2)
    out = torch.cat([
        zero_offset, final_offset_left, zero_offset, final_offset_right,
        final_offset_up, zero_offset, final_offset_down, zero_offset
    ],
                    dim=2)

    return out
