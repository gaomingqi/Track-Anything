import gradio as gr

def update_iframe(slider_value):
    return f'''
    <script>
        window.addEventListener('message', function(event) {{
            if (event.data.sliderValue !== undefined) {{
                var iframe = document.getElementById("text_iframe");
                iframe.src = "http://localhost:5001/get_text?slider_value=" + event.data.sliderValue;
            }}
        }}, false);
    </script>
    <iframe id="text_iframe" src="http://localhost:5001/get_text?slider_value={slider_value}" style="width: 100%; height: 100%; border: none;"></iframe>
    '''

iface = gr.Interface(
    fn=update_iframe,
    inputs=gr.inputs.Slider(minimum=0, maximum=100, step=1, default=50),
    outputs=gr.outputs.HTML(),
    allow_flagging=False,
)

iface.launch(server_name='0.0.0.0', server_port=12212)
