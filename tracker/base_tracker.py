# import for debugging
import os
import glob
import numpy as np
from PIL import Image
# import for base_tracker
import torch
import yaml
import torch.nn.functional as F
from model.network import XMem
from inference.inference_core import InferenceCore
from util.mask_mapper import MaskMapper
from torchvision import transforms
from util.range_transform import im_normalization
import sys
sys.path.insert(0, sys.path[0]+"/../")
from tools.painter import mask_painter


class BaseTracker:
    def __init__(self, xmem_checkpoint, device) -> None:
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
        
        # changable properties
        self.mapper = MaskMapper()
        self.initialised = False

    @torch.no_grad()
    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        """
        Input: 
        frames: numpy arrays (H, W, 3)
        first_frame_annotation: numpy array (H, W)

        Output:
        mask: numpy arrays (H, W)
        prob: numpy arrays, probability map (H, W)
        painted_image: numpy array (H, W, 3)
        """
        if first_frame_annotation is not None:
            # initialisation
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.Tensor(mask).to(self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask = None
            labels = None

        # prepare inputs
        frame_tensor = self.im_transform(frame).to(self.device)
        # track one frame
        prob = self.tracker.step(frame_tensor, mask, labels)
        # convert to mask
        out_mask = torch.argmax(prob, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
        painted_image = mask_painter(frame, out_mask)

        # mask, _, painted_frame
        return out_mask, prob, painted_image

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()


if __name__ == '__main__':
    # video frames
    video_path_list = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/dance-twirl', '*.jpg'))
    video_path_list.sort()
    # first frame
    first_frame_path = '/ssd1/gaomingqi/datasets/davis/Annotations/480p/dance-twirl/00000.png'
    # load frames
    frames = []
    for video_path in video_path_list:
        frames.append(np.array(Image.open(video_path).convert('RGB')))
    frames = np.stack(frames, 0)    # N, H, W, C
    # load first frame annotation
    first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C

    # ----------------------------------------------------------
    # initalise tracker
    # ----------------------------------------------------------
    device = 'cuda:1'
    XMEM_checkpoint = '/ssd1/gaomingqi/checkpoints/XMem-s012.pth'
    tracker = BaseTracker(XMEM_checkpoint, device)
    
    # track anything given in the first frame annotation
    for ti, frame in enumerate(frames):
        if ti == 0:
            mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
        else:
            mask, prob, painted_image = tracker.track(frame)
        # save
        painted_image = Image.fromarray(painted_image)
        painted_image.save(f'/ssd1/gaomingqi/results/TrackA/dance-twirl/{ti:05d}.png')

    # ----------------------------------------------------------
    # another video
    # ----------------------------------------------------------
    # video frames
    video_path_list = glob.glob(os.path.join('/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/camel', '*.jpg'))
    video_path_list.sort()
    # first frame
    first_frame_path = '/ssd1/gaomingqi/datasets/davis/Annotations/480p/camel/00000.png'
    # load frames
    frames = []
    for video_path in video_path_list:
        frames.append(np.array(Image.open(video_path).convert('RGB')))
    frames = np.stack(frames, 0)    # N, H, W, C
    # load first frame annotation
    first_frame_annotation = np.array(Image.open(first_frame_path).convert('P'))    # H, W, C
    
    print('first video done. clear.')

    tracker.clear_memory()
    # track anything given in the first frame annotation
    for ti, frame in enumerate(frames):
        if ti == 0:
            mask, prob, painted_image = tracker.track(frame, first_frame_annotation)
        else:
            mask, prob, painted_image = tracker.track(frame)
        # save
        painted_image = Image.fromarray(painted_image)
        painted_image.save(f'/ssd1/gaomingqi/results/TrackA/camel/{ti:05d}.png')
