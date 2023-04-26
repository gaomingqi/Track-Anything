import os
import glob
from PIL import Image

import torch
import yaml
import cv2
import importlib
import numpy as np
from tqdm import tqdm

from inpainter.util.tensor_util import resize_frames, resize_masks


class BaseInpainter:
    def __init__(self, E2FGVI_checkpoint, device) -> None:
        """
        E2FGVI_checkpoint: checkpoint of inpainter (version hq, with multi-resolution support)
        """
        net = importlib.import_module('inpainter.model.e2fgvi_hq')
        self.model = net.InpaintGenerator().to(device)
        self.model.load_state_dict(torch.load(E2FGVI_checkpoint, map_location=device))
        self.model.eval()
        self.device = device
        # load configurations
        with open("inpainter/config/config.yaml", 'r') as stream: 
            config = yaml.safe_load(stream) 
        self.neighbor_stride = config['neighbor_stride']
        self.num_ref = config['num_ref']
        self.step = config['step']

    # sample reference frames from the whole video
    def get_ref_index(self, f, neighbor_ids, length):
        ref_index = []
        if self.num_ref == -1:
            for i in range(0, length, self.step):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, f - self.step * (self.num_ref // 2))
            end_idx = min(length, f + self.step * (self.num_ref // 2))
            for i in range(start_idx, end_idx + 1, self.step):
                if i not in neighbor_ids:
                    if len(ref_index) > self.num_ref:
                        break
                    ref_index.append(i)
        return ref_index

    def inpaint(self, frames, masks, dilate_radius=15, ratio=1):
        """
        frames: numpy array, T, H, W, 3
        masks: numpy array, T, H, W
        dilate_radius: radius when applying dilation on masks
        ratio: down-sample ratio

        Output:
        inpainted_frames: numpy array, T, H, W, 3
        """
        assert frames.shape[:3] == masks.shape, 'different size between frames and masks'
        assert ratio > 0 and ratio <= 1, 'ratio must in (0, 1]'
        masks = masks.copy()
        masks = np.clip(masks, 0, 1)
        kernel = cv2.getStructuringElement(2, (dilate_radius, dilate_radius))
        masks = np.stack([cv2.dilate(mask, kernel) for mask in masks], 0)

        T, H, W = masks.shape
        # size: (w, h)
        if ratio == 1:
            size = None
        else:
            size = [int(W*ratio), int(H*ratio)]
            if size[0] % 2 > 0:
                size[0] += 1
            if size[1] % 2 > 0:
                size[1] += 1
        
        masks = np.expand_dims(masks, axis=3)    # expand to T, H, W, 1
        binary_masks = resize_masks(masks, tuple(size))
        frames = resize_frames(frames, tuple(size))          # T, H, W, 3
        # frames and binary_masks are numpy arrays

        h, w = frames.shape[1:3]
        video_length = T

        # convert to tensor
        imgs = (torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().unsqueeze(0).float().div(255)) * 2 - 1
        masks = torch.from_numpy(binary_masks).permute(0, 3, 1, 2).contiguous().unsqueeze(0)

        imgs, masks = imgs.to(self.device), masks.to(self.device)
        comp_frames = [None] * video_length

        for f in tqdm(range(0, video_length, self.neighbor_stride), desc='Inpainting image'):
            neighbor_ids = [
                i for i in range(max(0, f - self.neighbor_stride),
                                min(video_length, f + self.neighbor_stride + 1))
            ]
            ref_ids = self.get_ref_index(f, neighbor_ids, video_length)
            selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [3])],
                    3)[:, :, :, :h + h_pad, :]
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [4])],
                    4)[:, :, :, :, :w + w_pad]
                pred_imgs, _ = self.model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = pred_imgs[i].astype(np.uint8) * binary_masks[idx] + frames[idx] * (
                            1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5
        
        inpainted_frames = np.stack(comp_frames, 0)
        return inpainted_frames.astype(np.uint8)

if __name__ == '__main__':

    frame_path = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/parkour', '*.jpg'))
    frame_path.sort()
    mask_path = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/Annotations/480p/parkour', "*.png"))
    mask_path.sort()
    save_path = '/ssd1/gaomingqi/results/inpainting/parkour'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    frames = []
    masks = []
    for fid, mid in zip(frame_path, mask_path):
        frames.append(Image.open(fid).convert('RGB'))
        masks.append(Image.open(mid).convert('P'))

    frames = np.stack(frames, 0)
    masks = np.stack(masks, 0)

    # ----------------------------------------------
    # how to use
    # ----------------------------------------------
    # 1/3: set checkpoint and device
    checkpoint = '/ssd1/gaomingqi/checkpoints/E2FGVI-HQ-CVPR22.pth'
    device = 'cuda:6'
    # 2/3: initialise inpainter
    base_inpainter = BaseInpainter(checkpoint, device)
    # 3/3: inpainting (frames: numpy array, T, H, W, 3; masks: numpy array, T, H, W)
    # ratio: (0, 1], ratio for down sample, default value is 1
    inpainted_frames = base_inpainter.inpaint(frames, masks, ratio=1)   # numpy array, T, H, W, 3
    # ----------------------------------------------
    # end
    # ----------------------------------------------
    # save
    for ti, inpainted_frame in enumerate(inpainted_frames):
        frame = Image.fromarray(inpainted_frame).convert('RGB')
        frame.save(os.path.join(save_path, f'{ti:05d}.jpg'))
