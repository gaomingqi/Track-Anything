import torch
import numpy as np
from ..interact.s2m.s2m_network import deeplabv3plus_resnet50 as S2M

from util.tensor_util import pad_divide_by, unpad


class S2MController:
    """
    A controller for Scribble-to-Mask (for user interaction, not for DAVIS)
    Takes the image, previous mask, and scribbles to produce a new mask
    ignore_class is usually 255 
    0 is NOT the ignore class -- it is the label for the background
    """
    def __init__(self, s2m_net:S2M, num_objects, ignore_class, device='cuda:0'):
        self.s2m_net = s2m_net
        self.num_objects = num_objects
        self.ignore_class = ignore_class
        self.device = device

    def interact(self, image, prev_mask, scr_mask):
        image = image.to(self.device, non_blocking=True)
        prev_mask = prev_mask.unsqueeze(0)

        h, w = image.shape[-2:]
        unaggre_mask = torch.zeros((self.num_objects, h, w), dtype=torch.float32, device=image.device)

        for ki in range(1, self.num_objects+1):
            p_srb = (scr_mask==ki).astype(np.uint8)
            n_srb = ((scr_mask!=ki) * (scr_mask!=self.ignore_class)).astype(np.uint8)

            Rs = torch.from_numpy(np.stack([p_srb, n_srb], 0)).unsqueeze(0).float().to(image.device)

            inputs = torch.cat([image, (prev_mask==ki).float().unsqueeze(0), Rs], 1)
            inputs, pads = pad_divide_by(inputs, 16)

            unaggre_mask[ki-1] = unpad(torch.sigmoid(self.s2m_net(inputs)), pads)

        return unaggre_mask