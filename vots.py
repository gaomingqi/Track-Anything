import numpy as np
import torch
import torchvision
import os
from PIL import Image
# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    # height, width, layers = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # print(output_path)
    # for frame in frames:
    #     video.write(frame)
    
    # video.release()
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

sequence = "/home/caption_anthing/Track-Anything/vots/sequences/kite-4/color"

frames = [os.path.join(sequence, i) for i in os.listdir(sequence)]
frames.sort()

pil_images = [np.asanyarray(Image.open(i)) for i in frames]

generate_video_from_frames(pil_images, "./vots/kite-4.mp4", fps=30)


