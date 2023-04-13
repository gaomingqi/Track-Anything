import gradio as gr
from demo import automask_image_app, automask_video_app, sahi_autoseg_app
import argparse
import cv2
import time   
from PIL import Image
import numpy as np
def pause_video(play_state):
    print("user pause_video")
    play_state.append(time.time())
    return play_state

def play_video(play_state):
    print("user play_video")
    play_state.append(time.time())
    return play_state

def get_frames_from_video(video_input, play_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame-1], [nearest_frame+1], nearest_frame]
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

    for frame in frames:
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    key_frame_index = int(timestamp * fps)
    nearest_frame = frames[key_frame_index]
    frames = [frames[:key_frame_index], frames[key_frame_index:], nearest_frame]
    return frames, nearest_frame


with gr.Blocks() as iface:
    state = gr.State([])
    play_state = gr.State([])
    video_state = gr.State([[],[],[]])

    with gr.Row():
        with gr.Column(scale=1.0):
            video_input = gr.Video().style(height=720)

            # listen to the user action for play and pause input video
            video_input.play(fn=play_video, inputs=play_state, outputs=play_state)
            video_input.pause(fn=pause_video, inputs=play_state, outputs=play_state)
          
            with gr.Row():

                with gr.Row():
                    with gr.Column(scale=0.5):
                        template_frame = gr.Image(type="pil", interactive=True, elem_id="template_frame")
                        with gr.Column():
                            template_select_button = gr.Button(value="Template select", interactive=True, variant="primary")
                    
                    with gr.Column(scale=0.5):
                        with gr.Row(scale=0.4):
                            clear_button_clike = gr.Button(value="Clear Clicks", interactive=True)
                            clear_button_image = gr.Button(value="Clear Image", interactive=True)

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
iface.launch(debug=True, enable_queue=True, server_port=122, server_name="0.0.0.0")


    
