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

# download checkpoints
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

# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
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

    # initialize video_state
    video_state = {
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [None]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": 30
        }
    return video_state, gr.update(visible=True, maximum=len(frames), value=1)

# get the select frame from gradio slider
def select_template(image_selection_slider, video_state):

    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])


    return video_state["painted_images"][image_selection_slider], video_state

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    return painted_image, video_state, interactive_state

# tracking vos
def vos_tracking_video(video_state, interactive_state):
    model.xmem.clear_memory()
    following_frames = video_state["origin_images"][video_state["select_frame_number"]:]
    template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)

    video_state["masks"][video_state["select_frame_number"]:] = masks
    video_state["logits"][video_state["select_frame_number"]:] = logits
    video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    video_output = generate_video_from_frames(video_state["painted_images"], output_path="./result/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video
    interactive_state["inference_times"] += 1
    
    print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(interactive_state["inference_times"], 
                                                                                                                                           interactive_state["positive_click_times"]+interactive_state["negative_click_times"],
                                                                                                                                           interactive_state["positive_click_times"],
                                                                                                                                        interactive_state["negative_click_times"]))
    
    #### shanggao code for mask save
    if interactive_state["mask_save"]:
        if not os.path.exists('./result/mask/{}'.format(video_state["video_name"].split('.')[0])):
            os.makedirs('./result/mask/{}'.format(video_state["video_name"].split('.')[0]))
        i = 0
        print("save mask")
        for mask in video_state["masks"]:
            np.save(os.path.join('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), '{:05d}.npy'.format(i)), mask)
            i+=1
        # save_mask(video_state["masks"], video_state["video_name"])
    #### shanggao code for mask save
    return video_output, video_state, interactive_state

# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

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
args.port = 12212
args.device = "cuda:4"
args.mask_save = True

model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, args)

with gr.Blocks() as iface:
    """
        state for 
    """
    click_state = gr.State([[],[]])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save
    })
    video_state = gr.State(
        {
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        }
    )

    with gr.Row():

        # for user video input
        with gr.Column(scale=1.0):
            video_input = gr.Video().style(height=360)

          

            with gr.Row(scale=1):
                # put the template frame under the radio button
                with gr.Column(scale=0.5):
                    # extract frames
                    with gr.Column():
                        extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary") 

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
                    image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Image Selection", invisible=False)


                   
            
                with gr.Column(scale=0.5):
                    video_output = gr.Video().style(height=360)
                    tracking_video_predict_button = gr.Button(value="Tracking")

    # first step: get the video information 
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, video_state
        ],
        outputs=[video_state, image_selection_slider],
    )   

    # second step: select images from slider
    image_selection_slider.release(fn=select_template, 
                                   inputs=[image_selection_slider, video_state], 
                                   outputs=[template_frame, video_state], api_name="select_image")
    

    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state]
    )

    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, interactive_state],
        outputs=[video_output, video_state, interactive_state]
    )

    
    # clear input
    video_input.clear(
        lambda: (
        {
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        },
        {
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save
        },
        [[],[]]
                ),
        [],
        [ 
            video_state,
            interactive_state,
            click_state,
        ],
        queue=False,
        show_progress=False
    )
    clear_button_image.click(
        lambda: (
        {
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        },
        { 
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save
        },
        [[],[]]
                ),
        [],
        [ 
            video_state,
            interactive_state,
            click_state,
        ],

        queue=False,
        show_progress=False

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


    
