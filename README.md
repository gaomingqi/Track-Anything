<!-- ![](./assets/track-anything-logo.jpg) -->

<div align=center>
<img src="./assets/track-anything-logo.jpg"/>
</div>
<br/>
<div align=center>
<a src="https://img.shields.io/badge/%F0%9F%93%96-Open_in_Spaces-informational.svg?style=flat-square" href="https://arxiv.org/abs/2304.11968">
<img src="https://img.shields.io/badge/%F0%9F%93%96-Arxiv_2304.11968-red.svg?style=flat-square">
</a>
<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square" href="https://huggingface.co/spaces/watchtowerss/Track-Anything">
<img src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square">
</a>
<a src="https://img.shields.io/badge/%F0%9F%9A%80-SUSTech_VIP_Lab-important.svg?style=flat-square" href="https://zhengfenglab.com/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-SUSTech_VIP_Lab-important.svg?style=flat-square">
</a>
</div>

***Track-Anything*** is a flexible and interactive tool for video object tracking and segmentation. It is developed upon [Segment Anything](https://github.com/facebookresearch/segment-anything), can specify anything to track and segment via user clicks only. During tracking, users can flexibly change the objects they wanna track or correct the region of interest if there are any ambiguities. These characteristics enable ***Track-Anything*** to be suitable for: 
- Video object tracking and segmentation with shot changes. 
- Visualized development and data annnotation for video object tracking and segmentation.
- Object-centric downstream video tasks, such as video inpainting and editing. 

<div align=center>
<img src="./assets/avengers.gif"/>
</div>

<!-- ![avengers]() -->

## :rocket: Updates
- 2023/04/20: We deployed [[DEMO]](https://huggingface.co/spaces/watchtowerss/Track-Anything) on Hugging Face :hugs:! 

## Demo

https://user-images.githubusercontent.com/28050374/232842703-8395af24-b13e-4b8e-aafb-e94b61e6c449.MP4

### Multiple Object Tracking and Segmentation (with [XMem](https://github.com/hkchengrex/XMem))

https://user-images.githubusercontent.com/39208339/233035206-0a151004-6461-4deb-b782-d1dbfe691493.mp4

### Video Object Tracking and Segmentation with Shot Changes (with [XMem](https://github.com/hkchengrex/XMem))

https://user-images.githubusercontent.com/30309970/232848349-f5e29e71-2ea4-4529-ac9a-94b9ca1e7055.mp4

### Video Inpainting (with [E2FGVI](https://github.com/MCG-NKU/E2FGVI))

https://user-images.githubusercontent.com/28050374/232959816-07f2826f-d267-4dda-8ae5-a5132173b8f4.mp4

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

## Citation
If you find this work useful for your research or applications, please cite using this BibTeX:
```bibtex
@misc{yang2023track,
      title={Track Anything: Segment Anything Meets Videos}, 
      author={Jinyu Yang and Mingqi Gao and Zhe Li and Shang Gao and Fangjing Wang and Feng Zheng},
      year={2023},
      eprint={2304.11968},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

The project is based on [Segment Anything](https://github.com/facebookresearch/segment-anything), [XMem](https://github.com/hkchengrex/XMem), and [E2FGVI](https://github.com/MCG-NKU/E2FGVI). Thanks for the authors for their efforts.
