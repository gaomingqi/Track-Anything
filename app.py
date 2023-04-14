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
args.port=12219

model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, args)





def pause_video(play_state):
    print("user pause_video")
    play_state.append(time.time())
    return play_state

def play_video(play_state):
    print("user play_video")
    play_state.append(time.time())
    return play_state

# convert points input to prompt state
def get_prompt(inputs, click_state):
    points = []
    labels = []
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
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
    timestamp = play_state[1] - play_state[0]
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(frame)
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))

    for index, frame in enumerate(frames):
        frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    key_frame_index = int(timestamp * fps)
    nearest_frame = frames[key_frame_index]
    frames = [frames[:key_frame_index], frames[key_frame_index:], nearest_frame]
    return frames, nearest_frame

def inference_all(template_frame, evt:gr.SelectData):
    coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])

    # default value
    points = np.array([[evt.index[0],evt.index[1]]])
    labels= np.array([1])
    mask, logit, painted_image = model.inference_step(first_flag=True, interact_flag=False, image=np.asarray(template_frame), same_image_flag=False,points=points, labels=labels,logits=None,multimask=True)
    return painted_image


with gr.Blocks() as iface:
    """
        state for 
    """
    state = gr.State([])
    play_state = gr.State([])
    video_state = gr.State([[],[],[]])
    click_state = gr.State([[],[]])

    with gr.Row():

        # for user video input
        with gr.Column(scale=1.0):
            video_input = gr.Video().style(height=720)

            # listen to the user action for play and pause input video
            video_input.play(fn=play_video, inputs=play_state, outputs=play_state)
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
                    intermedia_image = gr.Image(type="pil", interactive=True, elem_id="intermedia_frame").style(height=360)

                    # seg_automask_video_points_per_batch = gr.Slider(
                    #     minimum=0,
                    #     maximum=64,
                    #     step=2,
                    #     value=64,
                    #     label="Points per Batch",
                    # )

            seg_automask_video_predict = gr.Button(value="Generator")
    
        
        # Display the first frame 
        # with gr.Column():
            # first_frame = gr.Image(type="pil", interactive=True, elem_id="first_frame")
            # seg_automask_firstframe = gr.Button(value="Find target")
            
            # video_input = gr.inputs.Video(type="mp4")

            # output = gr.outputs.Image(type="pil")

            # gr.Interface(fn=capture_frame, inputs=seg_automask_video_file, outputs=first_frame)

    # seg_automask_video_predict.click(
    #     fn=automask_video_app,
    #     inputs=[
    #         seg_automask_video_file,
    #         seg_automask_video_model_type,
    #         seg_automask_video_points_per_side,
    #         seg_automask_video_points_per_batch,
    #         seg_automask_video_min_area,
    #     ],
    #     outputs=[output_video],
    # )
    template_select_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, 
            play_state
        ],
        outputs=[video_state, template_frame],
    )

    template_frame.select(
        fn=inference_all,
        inputs=[
            template_frame
        ],
        outputs=[
            template_frame
        ]

    )

    # clear
    # clear_button_clike.click(
    #     lambda x: ([[], [], []], x, ""),
    #     [origin_image],
    #     [click_state, image_input, wiki_output],
    #     queue=False,
    #     show_progress=False
    # )
    # clear_button_image.click(
    #     lambda: (None, [], [], [[], [], []], "", ""),
    #     [],
    #     [image_input, chatbot, state, click_state, wiki_output, origin_image],
    #     queue=False,
    #     show_progress=False
    # )
    video_input.clear(
        lambda: (None, [], [], [[], [], []], None),
        [],
        [video_input, state, play_state, video_state, template_frame],
        queue=False,
        show_progress=False
    )

iface.queue(concurrency_count=1)
iface.launch(debug=True, enable_queue=True, server_port=args.port, server_name="0.0.0.0")


    
