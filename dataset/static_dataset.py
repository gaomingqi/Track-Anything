import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.tps import random_tps_warp
from dataset.reseed import reseed


class StaticTransformDataset(Dataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.

    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """
    def __init__(self, parameters, num_frames=3, max_num_obj=1):
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.im_list = []
        for parameter in parameters:
            root, method, multiplier = parameter
            if method == 0:
                # Get images
                classes = os.listdir(root)
                for c in classes:
                    imgs = os.listdir(path.join(root, c))
                    jpg_list = [im for im in imgs if 'jpg' in im[-3:].lower()]

                    joint_list = [path.join(root, c, im) for im in jpg_list]
                    self.im_list.extend(joint_list * multiplier)

            elif method == 1:
                self.im_list.extend([path.join(root, im) for im in os.listdir(root) if '.jpg' in im] * multiplier)

        print(f'{len(self.im_list)} images found.')

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
            transforms.Resize(384, InterpolationMode.BICUBIC),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=0),
            transforms.Resize(384, InterpolationMode.NEAREST),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=0),
        ])


        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _get_sample(self, idx):
        im = Image.open(self.im_list[idx]).convert('RGB')
        gt = Image.open(self.im_list[idx][:-3]+'png').convert('L')

        sequence_seed = np.random.randint(2147483647)

        images = []
        masks = []
        for _ in range(self.num_frames):
            reseed(sequence_seed)
            this_im = self.all_im_dual_transform(im)
            this_im = self.all_im_lone_transform(this_im)
            reseed(sequence_seed)
            this_gt = self.all_gt_dual_transform(gt)

            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            this_im = self.pair_im_dual_transform(this_im)
            this_im = self.pair_im_lone_transform(this_im)
            reseed(pairwise_seed)
            this_gt = self.pair_gt_dual_transform(this_gt)

            # Use TPS only some of the times
            # Not because TPS is bad -- just that it is too slow and I need to speed up data loading
            if np.random.rand() < 0.33:
                this_im, this_gt = random_tps_warp(this_im, this_gt, scale=0.02)

            this_im = self.final_im_transform(this_im)
            this_gt = self.final_gt_transform(this_gt)

            images.append(this_im)
            masks.append(this_gt)

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)

        return images, masks.numpy()

    def __getitem__(self, idx):
        additional_objects = np.random.randint(self.max_num_obj)
        indices = [idx, *np.random.randint(self.__len__(), size=additional_objects)]

        merged_images = None
        merged_masks = np.zeros((self.num_frames, 384, 384), dtype=np.int64)

        for i, list_id in enumerate(indices):
            images, masks = self._get_sample(list_id)
            if merged_images is None:
                merged_images = images
            else:
                merged_images = merged_images*(1-masks) + images*masks
            merged_masks[masks[:,0]>0.5] = (i+1)

        masks = merged_masks

        labels = np.unique(masks[0])
        # Remove background
        labels = labels[labels!=0]
        target_objects = labels.tolist()

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks==l)
            cls_gt[this_mask] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        info = {}
        info['name'] = self.im_list[idx]
        info['num_objects'] = max(1, len(target_objects))

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': merged_images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info
        }

        return data


    def __len__(self):
        return len(self.im_list)
