from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
import numpy as np



class TrackingAnything():
    def __init__(self, cfg):
        self.cfg = cfg
        self.samcontroler = SamControler(cfg.sam_checkpoint, cfg.model_type, cfg.device)
        self.xmem = BaseTracker(cfg.device, cfg.xmem_checkpoint)
    
   
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
        
        
        