import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import PIL
from .mask_painter import mask_painter as mask_painter2
from .base_segmenter import BaseSegmenter
from .painter import mask_painter, point_painter
import os
import requests
import sys 


mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5


class SamControler():
    def __init__(self, SAM_checkpoint, model_type, device):
        '''
        initialize sam controler
        '''

    
        self.sam_controler = BaseSegmenter(SAM_checkpoint, model_type, device)
        
    
    # def seg_again(self, image: np.ndarray):
    #     '''
    #     it is used when interact in video
    #     '''
    #     self.sam_controler.reset_image()
    #     self.sam_controler.set_image(image)
    #     return 
    
    
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True,mask_color=3):
        '''
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        '''
        # self.sam_controler.set_image(image)
        origal_image = self.sam_controler.orignal_image
        neg_flag = labels[-1]
        if neg_flag==1:
            #find neg
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            prompts = {
                'point_coords': points,
                'point_labels': labels,
                'mask_input': logit[None, :, :]
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'both', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        else:
           #find positive
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            
        
        assert len(points)==len(labels)
        
        painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
        painted_image = Image.fromarray(painted_image)
        
        return mask, logit, painted_image
    
    # def interact_loop(self, image:np.ndarray, same: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     origal_image = self.sam_controler.orignal_image
    #     if same: 
    #         '''
    #         true; loop in the same image
    #         '''
    #         prompts = {
    #             'point_coords': points,
    #             'point_labels': labels,
    #             'mask_input': logits[None, :, :]
    #         }
    #         masks, scores, logits = self.sam_controler.predict(prompts, 'both', multimask)
    #         mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            
    #         painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
    #         painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
    #         painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
    #         painted_image = Image.fromarray(painted_image)

    #         return mask, logit, painted_image
    #     else:
    #         '''
    #         loop in the different image, interact in the video 
    #         '''
    #         if image is None:
    #             raise('Image error')
    #         else:
    #             self.seg_again(image)
    #         prompts = {
    #             'point_coords': points,
    #             'point_labels': labels,
    #         }
    #         masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
    #         mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            
    #         painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
    #         painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
    #         painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
    #         painted_image = Image.fromarray(painted_image)

    #         return mask, logit, painted_image
        
    




# def initialize():
#     '''
#     initialize sam controler
#     '''
#     checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
#     folder = "segmenter"
#     SAM_checkpoint= './checkpoints/sam_vit_h_4b8939.pth'
#     download_checkpoint(checkpoint_url, folder, SAM_checkpoint)
    

#     model_type = 'vit_h'
#     device = "cuda:0"
#     sam_controler = BaseSegmenter(SAM_checkpoint, model_type, device)
#     return sam_controler


# def seg_again(sam_controler, image: np.ndarray):
#     '''
#     it is used when interact in video
#     '''
#     sam_controler.reset_image()
#     sam_controler.set_image(image)
#     return
    

# def first_frame_click(sam_controler, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
#     '''
#     it is used in first frame in video
#     return: mask, logit, painted image(mask+point)
#     '''
#     sam_controler.set_image(image) 
#     prompts = {
#         'point_coords': points,
#         'point_labels': labels,
#     }
#     masks, scores, logits = sam_controler.predict(prompts, 'point', multimask)
#     mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
    
#     assert len(points)==len(labels)
    
#     painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
#     painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
#     painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
#     painted_image = Image.fromarray(painted_image)
    
#     return mask, logit, painted_image

# def interact_loop(sam_controler, image:np.ndarray, same: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
#     if same: 
#         '''
#         true; loop in the same image
#         '''
#         prompts = {
#             'point_coords': points,
#             'point_labels': labels,
#             'mask_input': logits[None, :, :]
#         }
#         masks, scores, logits = sam_controler.predict(prompts, 'both', multimask)
#         mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        
#         painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
#         painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
#         painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
#         painted_image = Image.fromarray(painted_image)

#         return mask, logit, painted_image
#     else:
#         '''
#         loop in the different image, interact in the video 
#         '''
#         if image is None:
#             raise('Image error')
#         else:
#             seg_again(sam_controler, image)
#         prompts = {
#             'point_coords': points,
#             'point_labels': labels,
#         }
#         masks, scores, logits = sam_controler.predict(prompts, 'point', multimask)
#         mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        
#         painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
#         painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
#         painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
#         painted_image = Image.fromarray(painted_image)

#         return mask, logit, painted_image
        
    


# if __name__ == "__main__":
#     points = np.array([[500, 375], [1125, 625]])
#     labels = np.array([1, 1])
#     image = cv2.imread('/hhd3/gaoshang/truck.jpg')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     sam_controler = initialize()
#     mask, logit, painted_image_full = first_frame_click(sam_controler,image, points, labels, multimask=True)
#     painted_image = mask_painter2(image, mask.astype('uint8'), background_alpha=0.8)
#     painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
#     cv2.imwrite('/hhd3/gaoshang/truck_point.jpg', painted_image)
#     cv2.imwrite('/hhd3/gaoshang/truck_change.jpg', image)
#     painted_image_full.save('/hhd3/gaoshang/truck_point_full.jpg')
    
#     mask, logit, painted_image_full = interact_loop(sam_controler,image,True, points, np.array([1, 0]), logit, multimask=True)
#     painted_image = mask_painter2(image, mask.astype('uint8'), background_alpha=0.8)
#     painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
#     cv2.imwrite('/hhd3/gaoshang/truck_same.jpg', painted_image)
#     painted_image_full.save('/hhd3/gaoshang/truck_same_full.jpg')
    
#     mask, logit, painted_image_full = interact_loop(sam_controler,image, False, points, labels, multimask=True)
#     painted_image = mask_painter2(image, mask.astype('uint8'), background_alpha=0.8)
#     painted_image = cv2.cvtColor(painted_image, cv2.COLOR_RGB2BGR)  # numpy array (h, w, 3)
#     cv2.imwrite('/hhd3/gaoshang/truck_diff.jpg', painted_image)
#     painted_image_full.save('/hhd3/gaoshang/truck_diff_full.jpg')
    
    
    
    
    
    
    


    
    
    