# Track-Anything
**Track-Anything** is an Efficient Development Toolkit for Video Object Tracking and Segmentation, based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem). 

![image](https://github.com/gaomingqi/Track-Anything/blob/master/overview.png)

## Demo

https://user-images.githubusercontent.com/28050374/232312494-737066ea-93b2-4412-bdc3-30f8f0cf8647.mp4

## Get Started
#### Linux
```bash
# Clone the repository:
git clone https://github.com/gaomingqi/Track-Anything.git
cd Track-Anything

# Install dependencies:
pip install -r requirements.txt

# Run the Caption-Anything gradio demo.
python app.py --device cuda:0 --sam_model_type vit_h --port 12212
```

## Acknowledgement

The project is based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem). Thanks for the authors for their efforts.
