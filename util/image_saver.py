import cv2
import numpy as np

import torch
from dataset.range_transform import inv_im_trans
from collections import defaultdict

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def transpose_np(x):
    return np.transpose(x, [1,2,0])

def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_im(x):
    x = detach_to_cpu(x)
    x = inv_im_trans(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

# Predefined key <-> caption dict
key_captions = {
    'im': 'Image', 
    'gt': 'GT', 
}

"""
Return an image array with captions
keys in dictionary will be used as caption if not provided
values should contain lists of cv2 images
"""
def get_image_array(images, grid_shape, captions={}):
    h, w = grid_shape
    cate_counts = len(images)
    rows_counts = len(next(iter(images.values())))

    font = cv2.FONT_HERSHEY_SIMPLEX

    output_image = np.zeros([w*cate_counts, h*(rows_counts+1), 3], dtype=np.uint8)
    col_cnt = 0
    for k, v in images.items():

        # Default as key value itself
        caption = captions.get(k, k)

        # Handles new line character
        dy = 40
        for i, line in enumerate(caption.split('\n')):
            cv2.putText(output_image, line, (10, col_cnt*w+100+i*dy),
                     font, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Put images
        for row_cnt, img in enumerate(v):
            im_shape = img.shape
            if len(im_shape) == 2:
                img = img[..., np.newaxis]

            img = (img * 255).astype('uint8')

            output_image[(col_cnt+0)*w:(col_cnt+1)*w,
                         (row_cnt+1)*h:(row_cnt+2)*h, :] = img
            
        col_cnt += 1

    return output_image

def base_transform(im, size):
        im = tensor_to_np_float(im)
        if len(im.shape) == 3:
            im = im.transpose((1, 2, 0))
        else:
            im = im[:, :, None]

        # Resize
        if im.shape[1] != size:
            im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)

        return im.clip(0, 1)

def im_transform(im, size):
        return base_transform(inv_im_trans(detach_to_cpu(im)), size=size)

def mask_transform(mask, size):
    return base_transform(detach_to_cpu(mask), size=size)

def out_transform(mask, size):
    return base_transform(detach_to_cpu(torch.sigmoid(mask)), size=size)

def pool_pairs(images, size, num_objects):
    req_images = defaultdict(list)

    b, t = images['rgb'].shape[:2]

    # limit the number of images saved
    b = min(2, b)

    # find max num objects
    max_num_objects = max(num_objects[:b])

    GT_suffix = ''
    for bi in range(b):
        GT_suffix += ' \n%s' % images['info']['name'][bi][-25:-4]

    for bi in range(b):
        for ti in range(t):
            req_images['RGB'].append(im_transform(images['rgb'][bi,ti], size))
            for oi in range(max_num_objects):
                if ti == 0 or oi >= num_objects[bi]:
                    req_images['Mask_%d'%oi].append(mask_transform(images['first_frame_gt'][bi][0,oi], size))
                    # req_images['Mask_X8_%d'%oi].append(mask_transform(images['first_frame_gt'][bi][0,oi], size))
                    # req_images['Mask_X16_%d'%oi].append(mask_transform(images['first_frame_gt'][bi][0,oi], size))
                else:
                    req_images['Mask_%d'%oi].append(mask_transform(images['masks_%d'%ti][bi][oi], size))
                    # req_images['Mask_%d'%oi].append(mask_transform(images['masks_%d'%ti][bi][oi][2], size))
                    # req_images['Mask_X8_%d'%oi].append(mask_transform(images['masks_%d'%ti][bi][oi][1], size))
                    # req_images['Mask_X16_%d'%oi].append(mask_transform(images['masks_%d'%ti][bi][oi][0], size))
                req_images['GT_%d_%s'%(oi, GT_suffix)].append(mask_transform(images['cls_gt'][bi,ti,0]==(oi+1), size))
                # print((images['cls_gt'][bi,ti,0]==(oi+1)).shape)
                # print(mask_transform(images['cls_gt'][bi,ti,0]==(oi+1), size).shape)


    return get_image_array(req_images, size, key_captions)