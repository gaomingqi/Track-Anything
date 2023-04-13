import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import PIL
from mask_painter import mask_painter
from base_segmenter import BaseSegmenter



def initialize():
    '''
    initialize sam controler
    '''
    SAM_checkpoint= '/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    device = "cuda:0"
    sam_controler = BaseSegmenter(SAM_checkpoint, model_type, device)
    return sam_controler


def seg_again(sam_controler, image: np.ndarray):
    '''
    it is used when interact in video
    '''
    sam_controler.reset_image()
    sam_controler.set_image(image)
    return
    

def first_frame_click(sam_controler, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
    '''
    it is used in first frame in video
    '''
    sam_controler.set_image(image) 
    prompts = {
        'point_coords': points,
        'point_labels': labels,
    }
    masks, scores, logits = sam_controler.predict(prompts, 'point', multimask)
    mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
    return mask, logit

def interact_loop(sam_controler, same: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, image:np.ndarray=None, multimask=True):
    if same: 
        '''
        true; loop in the same image
        '''
        prompts = {
            'point_coords': points,
            'point_labels': labels,
            'mask_input': logits[None, :, :]
        }
        masks, scores, logits = sam_controler.predict(prompts, 'both', multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        return mask, logit
    else:
        '''
        loop in the different image, interact in the video 
        '''
        if image is None:
            raise('Image error')
        else:
            seg_again(sam_controler, image)
        prompts = {
            'point_coords': points,
            'point_labels': labels,
        }
        masks, scores, logits = sam_controler.predict(prompts, 'point', multimask)
        mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        return mask, logit
        
    


if __name__ == "__main__":
    points = np.array([[500, 375], [1125, 625]])
    labels = np.array([1, 1])
    image = cv2.imread('/hhd3/gaoshang/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    sam_controler = initialize()
    mask, logit = first_frame_click(sam_controler,image, points, labels, multimask=True)
    painted_image = mask_painter(image, mask.astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_point.jpg', painted_image)
    
    mask, logit = interact_loop(sam_controler,True, points, np.array([1, 0]), logit, multimask=True)
    painted_image = mask_painter(image, mask.astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_same.jpg', painted_image)
    
    mask, logit = interact_loop(sam_controler,False, points, labels, image = image, multimask=True)
    painted_image = mask_painter(image, mask.astype('uint8'), background_alpha=0.8)
    painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
    cv2.imwrite('/hhd3/gaoshang/truck_diff.jpg', painted_image)
    
    
    
    
    
    
    


    
    
    