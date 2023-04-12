# Getting Started

Our code is tested on Ubuntu. I have briefly tested the GUI on Windows (with a PyQt5 fix in the heading of interactive_demo.py).

## Requirements

* Python 3.8+
* PyTorch 1.11+ (See [PyTorch](https://pytorch.org/) for installation instructions)
* `torchvision` corresponding to the PyTorch version
* OpenCV (try `pip install opencv-python`)
* Others: `pip install -r requirements.txt`

## Dataset

I recommend either softlinking (`ln -s`) existing data or use the provided `scripts/download_datasets.py` to structure the datasets as our format.

`python -m scripts.download_dataset`

The structure is the same as the one in STCN -- you can place XMem in the same folder as STCN and it will work.
The script uses Google Drive and sometimes fails when certain files are blocked from automatic download. You would have to do some manual work in that case.
It does not download BL30K because it is huge and we don't want to crash your harddisks.

```bash
├── XMem
├── BL30K
├── DAVIS
│   ├── 2016
│   │   ├── Annotations
│   │   └── ...
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── static
│   ├── BIG_small
│   └── ...
├── long_video_set
│   ├── long_video
│   ├── long_video_x3
│   ├── long_video_davis
│   └── ...
├── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   ├── train_480p
│   └── valid
└── YouTube2018
    ├── all_frames
    │   └── valid_all_frames
    └── valid
```

## Long-Time Video

It comes from [AFB-URR](https://github.com/xmlyqing00/AFB-URR). Please following their license when using this data. We release our extended version (X3) and corresponding `_davis` versions such that the DAVIS evaluation can be used directly. They can be downloaded [[here]](TODO). The script above would also attempt to download it.

### BL30K

You can either use the automatic script `download_bl30k.py` or download it manually from [MiVOS](https://github.com/hkchengrex/MiVOS/#bl30k). Note that each segment is about 115GB in size -- 700GB in total. You are going to need ~1TB of free disk space to run the script (including extraction buffer).
The script uses Google Drive and sometimes fails when certain files are blocked from automatic download. You would have to do some manual work in that case.
