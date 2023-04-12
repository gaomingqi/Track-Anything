"""
Contains all the types of interaction related to the GUI
Not related to automatic evaluation in the DAVIS dataset

You can inherit the Interaction class to create new interaction types
undo is (sometimes partially) supported
"""


import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
from .interactive_utils import color_map, index_numpy_to_one_hot_torch


def aggregate_sbg(prob, keep_bg=False, hard=False):
    device = prob.device
    k, h, w = prob.shape
    ex_prob = torch.zeros((k+1, h, w), device=device)
    ex_prob[0] = 0.5
    ex_prob[1:] = prob
    ex_prob = torch.clamp(ex_prob, 1e-7, 1-1e-7)
    logits = torch.log((ex_prob /(1-ex_prob)))

    if hard:
        # Very low temperature o((⊙﹏⊙))o 🥶
        logits *= 1000

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]

def aggregate_wbg(prob, keep_bg=False, hard=False):
    k, h, w = prob.shape
    new_prob = torch.cat([
        torch.prod(1-prob, dim=0, keepdim=True),
        prob
    ], 0).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))

    if hard:
        # Very low temperature o((⊙﹏⊙))o 🥶
        logits *= 1000

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]

class Interaction:
    def __init__(self, image, prev_mask, true_size, controller):
        self.image = image 
        self.prev_mask = prev_mask
        self.controller = controller
        self.start_time = time.time()

        self.h, self.w = true_size

        self.out_prob = None
        self.out_mask = None

    def predict(self):
        pass


class FreeInteraction(Interaction):
    def __init__(self, image, prev_mask, true_size, num_objects):
        """
        prev_mask should be index format numpy array
        """
        super().__init__(image, prev_mask, true_size, None)

        self.K = num_objects

        self.drawn_map = self.prev_mask.copy()
        self.curr_path = [[] for _ in range(self.K + 1)]

        self.size = None

    def set_size(self, size):
        self.size = size

    """
    k - object id
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """
    def push_point(self, x, y, k, vis=None):
        if vis is not None:
            vis_map, vis_alpha = vis
        selected = self.curr_path[k]
        selected.append((x, y))
        if len(selected) >= 2:
            cv2.line(self.drawn_map, 
                (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                k, thickness=self.size)

            # Plot visualization
            if vis is not None:
                # Visualization for drawing
                if k == 0:
                    vis_map = cv2.line(vis_map, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        color_map[k], thickness=self.size)
                else:
                    vis_map = cv2.line(vis_map, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        color_map[k], thickness=self.size)
                # Visualization on/off boolean filter
                vis_alpha = cv2.line(vis_alpha, 
                    (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                    (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                    0.75, thickness=self.size)

        if vis is not None:
            return vis_map, vis_alpha

    def end_path(self):
        # Complete the drawing
        self.curr_path = [[] for _ in range(self.K + 1)]

    def predict(self):
        self.out_prob = index_numpy_to_one_hot_torch(self.drawn_map, self.K+1).cuda()
        # self.out_prob = torch.from_numpy(self.drawn_map).float().cuda()
        # self.out_prob, _ = pad_divide_by(self.out_prob, 16, self.out_prob.shape[-2:])
        # self.out_prob = aggregate_sbg(self.out_prob, keep_bg=True)
        return self.out_prob

class ScribbleInteraction(Interaction):
    def __init__(self, image, prev_mask, true_size, controller, num_objects):
        """
        prev_mask should be in an indexed form
        """
        super().__init__(image, prev_mask, true_size, controller)

        self.K = num_objects

        self.drawn_map = np.empty((self.h, self.w), dtype=np.uint8)
        self.drawn_map.fill(255)
        # background + k
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.size = 3

    """
    k - object id
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """
    def push_point(self, x, y, k, vis=None):
        if vis is not None:
            vis_map, vis_alpha = vis
        selected = self.curr_path[k]
        selected.append((x, y))
        if len(selected) >= 2:
            self.drawn_map = cv2.line(self.drawn_map, 
                (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                k, thickness=self.size)

            # Plot visualization
            if vis is not None:
                # Visualization for drawing
                if k == 0:
                    vis_map = cv2.line(vis_map, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        color_map[k], thickness=self.size)
                else:
                    vis_map = cv2.line(vis_map, 
                            (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                            (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                            color_map[k], thickness=self.size)
                # Visualization on/off boolean filter
                vis_alpha = cv2.line(vis_alpha, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        0.75, thickness=self.size)

        # Optional vis return
        if vis is not None:
            return vis_map, vis_alpha

    def end_path(self):
        # Complete the drawing
        self.curr_path = [[] for _ in range(self.K + 1)]

    def predict(self):
        self.out_prob = self.controller.interact(self.image.unsqueeze(0), self.prev_mask, self.drawn_map)
        self.out_prob = aggregate_wbg(self.out_prob, keep_bg=True, hard=True)
        return self.out_prob


class ClickInteraction(Interaction):
    def __init__(self, image, prev_mask, true_size, controller, tar_obj):
        """
        prev_mask in a prob. form
        """
        super().__init__(image, prev_mask, true_size, controller)
        self.tar_obj = tar_obj

        # negative/positive for each object
        self.pos_clicks = []
        self.neg_clicks = []

        self.out_prob = self.prev_mask.clone()

    """
    neg - Negative interaction or not
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """
    def push_point(self, x, y, neg, vis=None):
        # Clicks
        if neg:
            self.neg_clicks.append((x, y))
        else:
            self.pos_clicks.append((x, y))

        # Do the prediction
        self.obj_mask = self.controller.interact(self.image.unsqueeze(0), x, y, not neg)

        # Plot visualization
        if vis is not None:
            vis_map, vis_alpha = vis
            # Visualization for clicks
            if neg:
                vis_map = cv2.circle(vis_map, 
                        (int(round(x)), int(round(y))),
                        2, color_map[0], thickness=-1)
            else:
                vis_map = cv2.circle(vis_map, 
                        (int(round(x)), int(round(y))),
                        2, color_map[self.tar_obj], thickness=-1)

            vis_alpha = cv2.circle(vis_alpha, 
                        (int(round(x)), int(round(y))),
                        2, 1, thickness=-1)

            # Optional vis return
            return vis_map, vis_alpha

    def predict(self):
        self.out_prob = self.prev_mask.clone()
        # a small hack to allow the interacting object to overwrite existing masks
        # without remembering all the object probabilities
        self.out_prob = torch.clamp(self.out_prob, max=0.9)
        self.out_prob[self.tar_obj] = self.obj_mask
        self.out_prob = aggregate_wbg(self.out_prob[1:], keep_bg=True, hard=True)
        return self.out_prob
