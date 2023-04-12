# Modifed from https://github.com/seoungwugoh/ivs-demo

import numpy as np

import torch
import torch.nn.functional as F
from util.palette import davis_palette
from dataset.range_transform import im_normalization

def image_to_torch(frame: np.ndarray, device='cuda'):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device)/255
    frame_norm = im_normalization(frame)
    return frame_norm, frame

def torch_prob_to_numpy_mask(prob):
    mask = torch.argmax(prob, dim=0)
    mask = mask.cpu().numpy().astype(np.uint8)
    return mask

def index_numpy_to_one_hot_torch(mask, num_classes):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()

"""
Some constants fro visualization
"""
color_map_np = np.frombuffer(davis_palette, dtype=np.uint8).reshape(-1, 3).copy()
# scales for better visualization
color_map_np = (color_map_np.astype(np.float32)*1.5).clip(0, 255).astype(np.uint8)
color_map = color_map_np.tolist()
if torch.cuda.is_available():
    color_map_torch = torch.from_numpy(color_map_np).cuda() / 255

grayscale_weights = np.array([[0.3,0.59,0.11]]).astype(np.float32)
if torch.cuda.is_available():
    grayscale_weights_torch = torch.from_numpy(grayscale_weights).cuda().unsqueeze(0)

def get_visualization(mode, image, mask, layer, target_object):
    if mode == 'fade':
        return overlay_davis(image, mask, fade=True)
    elif mode == 'davis':
        return overlay_davis(image, mask)
    elif mode == 'light':
        return overlay_davis(image, mask, 0.9)
    elif mode == 'popup':
        return overlay_popup(image, mask, target_object)
    elif mode == 'layered':
        if layer is None:
            print('Layer file not given. Defaulting to DAVIS.')
            return overlay_davis(image, mask)
        else:
            return overlay_layer(image, mask, layer, target_object)
    else:
        raise NotImplementedError

def get_visualization_torch(mode, image, prob, layer, target_object):
    if mode == 'fade':
        return overlay_davis_torch(image, prob, fade=True)
    elif mode == 'davis':
        return overlay_davis_torch(image, prob)
    elif mode == 'light':
        return overlay_davis_torch(image, prob, 0.9)
    elif mode == 'popup':
        return overlay_popup_torch(image, prob, target_object)
    elif mode == 'layered':
        if layer is None:
            print('Layer file not given. Defaulting to DAVIS.')
            return overlay_davis_torch(image, prob)
        else:
            return overlay_layer_torch(image, prob, layer, target_object)
    else:
        raise NotImplementedError

def overlay_davis(image, mask, alpha=0.5, fade=False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)

def overlay_popup(image, mask, target_object):
    # Keep foreground colored. Convert background to grayscale.
    im_overlay = image.copy()

    binary_mask = ~(np.isin(mask, target_object))
    colored_region = (im_overlay[binary_mask]*grayscale_weights).sum(-1, keepdims=-1)
    im_overlay[binary_mask] = colored_region
    return im_overlay.astype(image.dtype)

def overlay_layer(image, mask, layer, target_object):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    obj_mask = (np.isin(mask, target_object)).astype(np.float32)
    layer_alpha = layer[:, :, 3].astype(np.float32) / 255
    layer_rgb = layer[:, :, :3]
    background_alpha = np.maximum(obj_mask, layer_alpha)[:,:,np.newaxis]
    obj_mask = obj_mask[:,:,np.newaxis]
    im_overlay = (image*(1-background_alpha) + layer_rgb*(1-obj_mask) + image*obj_mask).clip(0, 255)
    return im_overlay.astype(image.dtype)

def overlay_davis_torch(image, mask, alpha=0.5, fade=False):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # Changes the image in-place to avoid copying
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.argmax(mask, dim=0)

    colored_mask = color_map_torch[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6

    im_overlay = (im_overlay*255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay

def overlay_popup_torch(image, mask, target_object):
    # Keep foreground colored. Convert background to grayscale.
    image = image.permute(1, 2, 0)
    
    if len(target_object) == 0:
        obj_mask = torch.zeros_like(mask[0]).unsqueeze(2)
    else:
        # I should not need to convert this to numpy.
        # uUsing list works most of the time but consistently fails
        # if I include first object -> exclude it -> include it again.
        # I check everywhere and it makes absolutely no sense.
        # I am blaming this on PyTorch and calling it a day
        obj_mask = mask[np.array(target_object,dtype=np.int32)].sum(0).unsqueeze(2)
    gray_image = (image*grayscale_weights_torch).sum(-1, keepdim=True)
    im_overlay = obj_mask*image + (1-obj_mask)*gray_image

    im_overlay = (im_overlay*255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay

def overlay_layer_torch(image, mask, layer, target_object):
    # insert a layer between foreground and background
    # The CPU version is less accurate because we are using the hard mask
    # The GPU version has softer edges as it uses soft probabilities
    image = image.permute(1, 2, 0)

    if len(target_object) == 0:
        obj_mask = torch.zeros_like(mask[0])
    else:
        # I should not need to convert this to numpy.
        # uUsing list works most of the time but consistently fails
        # if I include first object -> exclude it -> include it again.
        # I check everywhere and it makes absolutely no sense.
        # I am blaming this on PyTorch and calling it a day
        obj_mask = mask[np.array(target_object,dtype=np.int32)].sum(0)
    layer_alpha = layer[:, :, 3]
    layer_rgb = layer[:, :, :3]
    background_alpha = torch.maximum(obj_mask, layer_alpha).unsqueeze(2)
    obj_mask = obj_mask.unsqueeze(2)
    im_overlay = (image*(1-background_alpha) + layer_rgb*(1-obj_mask) + image*obj_mask).clip(0, 1)

    im_overlay = (im_overlay*255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay
