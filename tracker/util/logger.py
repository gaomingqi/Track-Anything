"""
Dumps things to tensorboard and console
"""

import os
import warnings

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def fix_width_trunc(x):
    return ('{:.9s}'.format('{:0.9f}'.format(x)))

class TensorboardLogger:
    def __init__(self, short_id, id, git_info):
        self.short_id = short_id
        if self.short_id == 'NULL':
            self.short_id = 'DEBUG'

        if id is None:
            self.no_log = True
            warnings.warn('Logging has been disbaled.')
        else:
            self.no_log = False

            self.inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

            self.inv_seg_trans = transforms.Normalize(
                mean=[-0.5/0.5],
                std=[1/0.5])

            log_path = os.path.join('.', 'logs', '%s' % id)
            self.logger = SummaryWriter(log_path)

        self.log_string('git', git_info)

    def log_scalar(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        self.logger.add_scalar(tag, x, step)

    def log_metrics(self, l1_tag, l2_tag, val, step, f=None):
        tag = l1_tag + '/' + l2_tag
        text = '{:s} - It {:6d} [{:5s}] [{:13}]: {:s}'.format(self.short_id, step, l1_tag.upper(), l2_tag, fix_width_trunc(val))
        print(text)
        if f is not None:
            f.write(text + '\n')
            f.flush()
        self.log_scalar(tag, val, step)

    def log_im(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = self.inv_im_trans(x)
        x = tensor_to_numpy(x)
        self.logger.add_image(tag, x, step)

    def log_cv2(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = x.transpose((2, 0, 1))
        self.logger.add_image(tag, x, step)

    def log_seg(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = self.inv_seg_trans(x)
        x = tensor_to_numpy(x)
        self.logger.add_image(tag, x, step)

    def log_gray(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = tensor_to_numpy(x)
        self.logger.add_image(tag, x, step)

    def log_string(self, tag, x):
        print(tag, x)
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        self.logger.add_text(tag, x)
        