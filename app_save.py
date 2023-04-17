import gradio as gr
from demo import automask_image_app, automask_video_app, sahi_autoseg_app
import argparse
import cv2
import time   
from PIL import Image
import numpy as np
import os
import sys
sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")
from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
import torch 
import concurrent.futures
import queue

def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def pause_video(play_state):
    print("user pause_video")
    play_state.append(time.time())
    return play_state

def play_video(play_state):
    print("user play_video")
    play_state.append(time.time())
    return play_state

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt
    
def get_frames_from_video(video_input, play_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    # video_name = video_path.split('/')[-1]
    
    try:
        timestamp = play_state[1] - play_state[0]
    except:
        timestamp = 0
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))

    # for index, frame in enumerate(frames):
        # frames[index] = np.asarray(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    
    key_frame_index = int(timestamp * fps)
    nearest_frame = frames[key_frame_index]
    frames_split = [frames[:key_frame_index], frames[key_frame_index:], nearest_frame]
    # output_path='./seperate.mp4'
    # torchvision.io.write_video(output_path, frames[1], fps=fps, video_codec="libx264")

    # set image in sam when select the template frame
    model.samcontroler.sam_controler.set_image(nearest_frame)
    return frames_split, nearest_frame, nearest_frame, fps

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
    
    # for frame in frames:
    #     video.write(frame)
    
    # video.release()
    frames = torch.from_numpy(np.asarray(frames))
    output_path='./output.mp4'
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def model_reset():
    model.xmem.clear_memory()
    return None

def sam_refine(origin_frame, point_prompt, click_state, logit, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
    
    # prompt for sam model
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    # default value
    # points = np.array([[evt.index[0],evt.index[1]]])
    # labels= np.array([1])
    if len(logit)==0:
        logit = None
    
    mask, logit, painted_image = model.first_frame_click( 
                                                      image=origin_frame, 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    return painted_image, click_state, logit, mask



def vos_tracking_video(video_state, template_mask,fps,video_input):

    masks, logits, painted_images = model.generator(images=video_state[1], template_mask=template_mask)
    video_output = generate_video_from_frames(painted_images, output_path="./output.mp4", fps=fps)
    # image_selection_slider = gr.Slider(minimum=1, maximum=len(video_state[1]), value=1, label="Image Selection", interactive=True)
    video_name = video_input.split('/')[-1].split('.')[0]
    result_path = os.path.join('/hhd3/gaoshang/Track-Anything/results/'+video_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    i=0
    for mask in masks:
        np.save(os.path.join(result_path,'{:05}.npy'.format(i)), mask)
        i+=1
    return video_output, painted_images, masks, logits

def vos_tracking_image(image_selection_slider, painted_images):

    # images = video_state[1]
    percentage = image_selection_slider / 100
    select_frame_num = int(percentage * len(painted_images))
    return painted_images[select_frame_num], select_frame_num

def interactive_correction(video_state, point_prompt, click_state, select_correction_frame, evt: gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    refine_image = video_state[1][select_correction_frame]
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
    
    # prompt for sam model
    prompt = get_prompt(click_state=click_state, click_input=coordinate)
    model.samcontroler.seg_again(refine_image)
    corrected_mask, corrected_logit, corrected_painted_image = model.first_frame_click( 
                                                      image=refine_image, 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    return corrected_painted_image, [corrected_mask, corrected_logit, corrected_painted_image]

def correct_track(video_state, select_correction_frame, corrected_state, masks, logits, painted_images, fps, video_input):
    model.xmem.clear_memory()
    # inference the following images
    following_images = video_state[1][select_correction_frame:]
    corrected_masks, corrected_logits, corrected_painted_images = model.generator(images=following_images, template_mask=corrected_state[0])
    masks = masks[:select_correction_frame] + corrected_masks
    logits = logits[:select_correction_frame] + corrected_logits
    painted_images = painted_images[:select_correction_frame] + corrected_painted_images
    video_output = generate_video_from_frames(painted_images, output_path="./output.mp4", fps=fps)

    video_name = video_input.split('/')[-1].split('.')[0]
    result_path = os.path.join('/hhd3/gaoshang/Track-Anything/results/'+video_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    i=0
    for mask in masks:
        np.save(os.path.join(result_path,'{:05}.npy'.format(i)), mask)
        i+=1
    return video_output, painted_images, logits, masks 

# check and download checkpoints if needed
SAM_checkpoint = "sam_vit_h_4b8939.pth" 
sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
folder ="./checkpoints"
SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, SAM_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)

# args, defined in track_anything.py
args = parse_augment()
args.port = 12207
args.device = "cuda:5"

model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, args)

with gr.Blocks() as iface:
    """
        state for 
    """
    state = gr.State([])
    play_state = gr.State([])
    video_state = gr.State([[],[],[]])
    click_state = gr.State([[],[]])
    logits = gr.State([])
    masks = gr.State([])
    painted_images = gr.State([])
    origin_image = gr.State(None)
    template_mask = gr.State(None)
    select_correction_frame = gr.State(None)
    corrected_state = gr.State([[],[],[]])
    fps = gr.State([])
    # video_name = gr.State([])
    # queue value for image refresh, origin image, mask, logits, painted image



    with gr.Row():

        # for user video input
        with gr.Column(scale=1.0):
            video_input = gr.Video().style(height=720)

            # listen to the user action for play and pause input video
            video_input.play(fn=play_video, inputs=play_state, outputs=play_state, scroll_to_output=True, show_progress=True)
            video_input.pause(fn=pause_video, inputs=play_state, outputs=play_state)
          

            with gr.Row(scale=1):
                # put the template frame under the radio button
                with gr.Column(scale=0.5):
                     # click points settins, negative or positive, mode continuous or single
                    with gr.Row():
                        with gr.Row(scale=0.5):
                            point_prompt = gr.Radio(
                                choices=["Positive",  "Negative"],
                                value="Positive",
                                label="Point Prompt",
                                interactive=True)
                            click_mode = gr.Radio(
                                choices=["Continuous",  "Single"],
                                value="Continuous",
                                label="Clicking Mode",
                                interactive=True)
                        with gr.Row(scale=0.5):
                            clear_button_clike = gr.Button(value="Clear Clicks", interactive=True).style(height=160)
                            clear_button_image = gr.Button(value="Clear Image", interactive=True)
                    template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame").style(height=360)
                    with gr.Column():
                        template_select_button = gr.Button(value="Template select", interactive=True, variant="primary")
                    
                   
            
                with gr.Column(scale=0.5):


                    # for intermedia result check and correction
                    # intermedia_image = gr.Image(type="pil", interactive=True, elem_id="intermedia_frame").style(height=360)
                    video_output = gr.Video().style(height=360)
                    tracking_video_predict_button = gr.Button(value="Tracking")

                    image_output = gr.Image(type="pil", interactive=True, elem_id="image_output").style(height=360)
                    image_selection_slider = gr.Slider(minimum=0, maximum=100, step=0.1, value=0, label="Image Selection", interactive=True)
                    correct_track_button = gr.Button(value="Interactive Correction")

    template_frame.select(
        fn=sam_refine,
        inputs=[
            origin_image, point_prompt, click_state, logits
        ],
        outputs=[
            template_frame, click_state, logits, template_mask
        ]
    )
            
    template_select_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, 
            play_state
        ],
        # outputs=[video_state, template_frame, origin_image, fps, video_name],
        outputs=[video_state, template_frame, origin_image, fps],
    )   

    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, template_mask, fps, video_input],
        outputs=[video_output, painted_images, masks, logits]
    )
    image_selection_slider.release(fn=vos_tracking_image, 
                                   inputs=[image_selection_slider, painted_images], outputs=[image_output, select_correction_frame], api_name="select_image")
    # correction
    image_output.select(
        fn=interactive_correction,
        inputs=[video_state, point_prompt, click_state, select_correction_frame],
        outputs=[image_output, corrected_state]
    )
    correct_track_button.click(
        fn=correct_track,
        inputs=[video_state, select_correction_frame, corrected_state, masks, logits, painted_images, fps,video_input],
        outputs=[video_output, painted_images, logits, masks ]
    )
   
    
    
    # clear input
    video_input.clear(
        lambda: ([], [], [[], [], []], 
                 None, "", "", "", "", "", "", "", [[],[]],
                 None),
        [],
        [ state, play_state, video_state, 
         template_frame, video_output, image_output, origin_image, template_mask, painted_images, masks, logits, click_state,
         select_correction_frame],
        queue=False,
        show_progress=False
    )
    clear_button_image.click(
        fn=model_reset
    )
    clear_button_clike.click(
       lambda: ([[],[]]),
        [],
        [click_state],
        queue=False,
        show_progress=False
    ) 
iface.queue(concurrency_count=1)
iface.launch(debug=True, enable_queue=True, server_port=args.port, server_name="0.0.0.0")


    
