# Track-Anything

***Track-Anything*** is a flexible and interactive tool for video object tracking and segmentation. It is developed upon [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem), can specify anything to track and segment via user clicks only. During tracking, users can flexibly change the objects they wanna track or correct the region of interest if there are any ambiguities. These characteristics enable ***Track-Anything*** to be suitable for: 
- Video object tracking and segmentation with shot changes. 
- Data annnotation for video object tracking and segmentation.
- Object-centric downstream video tasks, such as video inpainting and editing. 

## Demo

<video src="./assets/demo_version_1.MP4" controls="controls" poster="./assets/poster_demo_version_1.png"></video>


### Video Object Tracking and Segmentation with Shot Changes

one gif/video

### Video Inpainting (with [E2FGVI](https://github.com/MCG-NKU/E2FGVI))

![image](./assets/inpainting.gif)

### Video Editing

one gif/video

## Get Started
#### Linux
```bash
# Clone the repository:
git clone https://github.com/gaomingqi/Track-Anything.git
cd Track-Anything

# Install dependencies: 
pip install -r requirements.txt

# Install dependencies for inpainting: 
pip install -U openmim
mim install mmcv

# Install dependencies for editing
pip install madgrad 

# Run the Track-Anything gradio demo.
python app.py --device cuda:0 --sam_model_type vit_h --port 12212
```

## Acknowledgement

The project is based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem). Thanks for the authors for their efforts.
