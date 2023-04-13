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
# for data transormation
from torchvision import transforms
from dataset.range_transform import im_normalization


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

    def track(self, frames, first_frame_annotation):
        """
        Input: 
        frames: numpy arrays: T, H, W, 3 (T: number of frames)
        first_frame_annotation: numpy array: H, W

        Output:
        masks: numpy arrays: T, H, W
        """
        # data transformation
        for frame in frames:
            frame = self.im_transform(frame)

        # tracking
        


if __name__ == '__main__':
    # video frames
    video_path_list = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/dance-twirl', '*.jpg'))
    video_path_list.sort()
    # first frame
    first_frame_path = '/ssd1/gaomingqi/datasets/davis/Annotations/480p/dance-twirl/00000.png'

    # load frames
<<<<<<< HEAD
=======
    frames = ["test_confict"]
>>>>>>> a5606340a199569856ffa1585aeeff5a40cc34ba
    for video_path in video_path_list:
        frames.append(np.array(Image.open(video_path).convert('RGB')))
    frames = np.stack(frames, 0)    # N, H, W, C

    # load first frame annotation
    first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C

    # initalise tracker
    device = 'cuda:0'
    XMEM_checkpoint = '/ssd1/gaomingqi/checkpoints/XMem-s012.pth'
    tracker = BaseTracker('cuda:0', XMEM_checkpoint)
    
    # track anything given in the first frame annotation
    tracker.track(frames, first_frame_annotation)
