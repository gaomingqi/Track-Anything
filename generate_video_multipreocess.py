import torchvision
import os
import numpy as np
import torch
import cv2
# from tqdm import tqdm
from multiprocessing import Pool
def read_image_from_userfolder(image_path):
    # if type:
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image
def generate_video_from_frames(frames_path, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    if os.path.exists(output_path):
        return output_path
    frames = []
    # print("read frames from sequence")
    for file in frames_path:
        frames.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # print("generate video from frames for preview")
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def process_seq(seq):
    frames = [os.path.join(votdir, "sequences", seq, "color", i) for i in os.listdir(os.path.join(votdir, "sequences", seq, "color"))]
    frames.sort()
    video_path = generate_video_from_frames(frames, output_path=os.path.join(votdir, "frame2video", "{}.mp4".format(seq)))
    print("{}.mp4 is ok".format(seq))
    return video_path


votdir = "/home/dataset/vots2023/"
sequence_list = os.listdir(os.path.join(votdir, "sequences"))
num_process = 8
with Pool(num_process) as p:
    video_paths = p.map(process_seq, sequence_list)