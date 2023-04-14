from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
import numpy as np
import argparse



class TrackingAnything():
    def __init__(self, sam_checkpoint, xmem_checkpoint, args):
        self.args = args
        self.samcontroler = SamControler(sam_checkpoint, args.sam_model_type, args.device)
        self.xmem = BaseTracker(xmem_checkpoint, device=args.device, )
    
   
    def inference_step(self, first_flag: bool, interact_flag: bool, image: np.ndarray, 
                       same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
        if first_flag:
            mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
            return mask, logit, painted_image
        
        if interact_flag:
            mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
            return mask, logit, painted_image
        
        mask, logit, painted_image = self.xmem.track(image, logit)
        return mask, logit, painted_image
    
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels,logits, multimask)
        return mask, logit, painted_image
    
    def interact(self, image: np.ndarray, same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
        mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
        return mask, logit, painted_image

    def generator(self, image: np.ndarray, logits:np.ndarray):
        mask, logit, painted_image = self.xmem.track(image, logits)
        return mask, logit, painted_image
        
        
def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=6080, help="only useful when running gradio applications")  
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args 