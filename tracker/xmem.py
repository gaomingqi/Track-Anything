# input: frame list, first frame mask
# output: segmentation results on all frames
import os
import glob
import numpy as np
from PIL import Image


class XMem:
    # based on https://github.com/hkchengrex/XMem
    pass


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

