# input: frame list, first frame mask
# output: segmentation results on all frames
import os
import glob
import numpy as np
from PIL import Image

import torch
import yaml
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.data.mask_mapper import MaskMapper

# for data transormation
from torchvision import transforms
from dataset.range_transform import im_normalization
import torch.nn.functional as F

import sys
sys.path.insert(0, sys.path[0]+"/../")
from tools.painter import mask_painter


class BaseTracker:
    def __init__(self, device, xmem_checkpoint) -> None:
        """
        device: model device
        xmem_checkpoint: checkpoint of XMem model
        """
        # load configurations
        with open("tracker/config/config.yaml", 'r') as stream: 
            config = yaml.safe_load(stream) 
        # initialise XMem
        network = XMem(config, xmem_checkpoint).to(device).eval()
        # initialise IncerenceCore
        self.tracker = InferenceCore(network, config)
        # data transformation
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        self.device = device

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    def track(self, frames, first_frame_annotation):
        """
        Input: 
        frames: numpy arrays: T, H, W, 3 (T: number of frames)
        first_frame_annotation: numpy array: H, W

        Output:
        masks: numpy arrays: H, W
        """
        shape = np.array(frames).shape[1:3] # H, W
        frame_list = [self.im_transform(frame).to(self.device) for frame in frames]
        frame_tensors = torch.stack(frame_list, dim=0)
        
        # data transformation
        mapper = MaskMapper()

        vid_length = len(frame_tensors)

        for ti, frame_tensor in enumerate(frame_tensors):
            if ti == 0:
                mask, labels = mapper.convert_mask(first_frame_annotation)
                mask = torch.Tensor(mask).to(self.device)
                self.tracker.set_all_labels(list(mapper.remappings.values()))
            else:
                mask = None
                labels = None
            
            # track one frame
            prob = self.tracker.step(frame_tensor, mask, labels, end=(ti==vid_length-1))
            
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            painted_image = mask_painter(frames[ti], out_mask)
            # save
            painted_image = Image.fromarray(painted_image)
            painted_image.save(f'/ssd1/gaomingqi/results/TrackA/{ti}.png')


if __name__ == '__main__':
    # video frames
    video_path_list = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/dance-twirl', '*.jpg'))
    video_path_list.sort()
    # first frame
    first_frame_path = '/ssd1/gaomingqi/datasets/davis/Annotations/480p/dance-twirl/00000.png'
    # load frames
<<<<<<< HEAD
    frames = []
=======
    frames = ["test_confict"]
>>>>>>> 5ca44baea36b7c66043342afc9ffb966e6d24417
    for video_path in video_path_list:
        frames.append(np.array(Image.open(video_path).convert('RGB')))
    frames = np.stack(frames, 0)    # N, H, W, C
    # load first frame annotation
    first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C

    # ----------------------------------------------------------
    # initalise tracker
    # ----------------------------------------------------------
    device = 'cuda:0'
    XMEM_checkpoint = '/ssd1/gaomingqi/checkpoints/XMem-s012.pth'
    tracker = BaseTracker('cuda:0', XMEM_checkpoint)
    
    # track anything given in the first frame annotation
    tracker.track(frames, first_frame_annotation)
