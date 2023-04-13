import gradio as gr
from demo import automask_image_app, automask_video_app, sahi_autoseg_app
import argparse
import cv2
import time   
def pause_video():
    print(time.time())
def play_video():
    print("play video")
    print(time.time)

def get_frames_from_video(video_path, timestamp):
    """
        video_path:str
        timestamp:float64
        return [[0:nearest_frame-1], [nearest_frame+1], nearest_frame]
    """
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
    
    key_frame_index = int(timestamp * fps)
    nearest_frame = frames[key_frame_index]
    frames = [frames[:key_frame_index], frames[key_frame_index:], nearest_frame]
    return frames


with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column(scale=1.0):
            seg_automask_video_file = gr.Video().style(height=720)
            seg_automask_video_file.play(fn=play_video)
            seg_automask_video_file.pause(fn=pause_video)
            with gr.Row():
                with gr.Column():
                    seg_automask_video_model_type = gr.Dropdown(
                        choices=[
                            "vit_h",
                            "vit_l",
                            "vit_b",
                        ],
                        value="vit_l",
                        label="Model Type",
                    )
                    seg_automask_video_min_area = gr.Number(
                        value=1000,
                        label="Min Area",
                    )

                with gr.Row():
                    with gr.Column():
                        seg_automask_video_points_per_side = gr.Slider(
                            minimum=0,
                            maximum=32,
                            step=2,
                            value=16,
                            label="Points per Side",
                        )
                        
                        seg_automask_video_points_per_batch = gr.Slider(
                            minimum=0,
                            maximum=64,
                            step=2,
                            value=64,
                            label="Points per Batch",
                        )

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

iface.queue(concurrency_count=1)
iface.launch(debug=True, enable_queue=True, server_port=12212, server_name="0.0.0.0")


    
