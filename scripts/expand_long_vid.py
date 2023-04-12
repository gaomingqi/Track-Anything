import sys
import os
from os import path
from shutil import copy2

input_path = sys.argv[1]
output_path = sys.argv[2]
multiplier = int(sys.argv[3])
image_path = path.join(input_path, 'JPEGImages')
gt_path = path.join(input_path, 'Annotations')

videos = sorted(os.listdir(image_path))

for vid in videos:
    os.makedirs(path.join(output_path, 'JPEGImages', vid), exist_ok=True)
    os.makedirs(path.join(output_path, 'Annotations', vid), exist_ok=True)
    frames = sorted(os.listdir(path.join(image_path, vid)))

    num_frames = len(frames)
    counter = 0
    output_counter = 0
    direction = 1
    for _ in range(multiplier):
        for _ in range(num_frames):
            copy2(path.join(image_path, vid, frames[counter]), 
                    path.join(output_path, 'JPEGImages', vid, f'{output_counter:05d}.jpg'))

            mask_path = path.join(gt_path, vid, frames[counter].replace('.jpg', '.png'))
            if path.exists(mask_path):
                copy2(mask_path, 
                    path.join(output_path, 'Annotations', vid, f'{output_counter:05d}.png'))

            counter += direction
            output_counter += 1
            if counter == 0 or counter == len(frames) - 1:
                direction *= -1
