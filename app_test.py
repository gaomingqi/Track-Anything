import gradio as gr
import time

def capture_frame(video):
    frame = video.get_frame_at_sec(video.current_time)
    return frame

def capture_time(video):
    while True:
        if video.paused:
            time_paused = video.current_time
            return time_paused

iface = gr.Interface(fn=capture_frame, 
                     inputs=[gr.inputs.Video(type="mp4", label="Input video", 
                                             source="upload")],
                     outputs=["image"], 
                     server_port=12212, 
                     server_name="0.0.0.0",
                     capture_session=True)

video_player = iface.video[0]
video_player.pause = False

time_interface = gr.Interface(fn=capture_time,
                              inputs=[gr.inputs.Video(type="mp4", label="Input video",
                                                      source="upload", max_duration=10)],
                              outputs=["text"],
                              server_port=12212,
                              server_name="0.0.0.0",
                              capture_session=True)

time_interface.video[0].play = False
time_interface.video[0].pause = False

iface.launch()
time_interface.launch()
