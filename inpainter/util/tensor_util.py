import cv2
import numpy as np

# resize frames
def resize_frames(frames, size=None):
    """
    size: (w, h)
    """
    if size is not None:
        frames = [cv2.resize(f, size) for f in frames]
        frames = np.stack(frames, 0)

    return frames

# resize frames
def resize_masks(masks, size=None):
    """
    size: (w, h)
    """
    if size is not None:
        masks = [np.expand_dims(cv2.resize(m, size), 2) for m in masks]
        masks = np.stack(masks, 0)

    return masks
