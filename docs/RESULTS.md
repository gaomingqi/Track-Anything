# Results

## Preamble

Our code, by default, uses automatic mixed precision (AMP). Its effect on the output is negligible.
All speeds reported in the paper are recorded with AMP turned off (`--benchmark`).
Due to refactoring, there might be slight differences between the outputs produced by this code base with the precomputed results/results reported in the paper. This difference rarely leads to a change of the least significant figure (i.e., 0.1).

**For most complete results, please see the paper (and the appendix)!**

All available precomputed results can be found [[here]](https://drive.google.com/drive/folders/1UxHPXJbQLHjF5zYVn3XZCXfi_NYL81Bf?usp=sharing).

## Pretrained models

We provide four pretrained models for download:

1. XMem.pth (Default)
2. XMem-s012.pth (Trained with BL30K)
3. XMem-s2.pth (No pretraining on static images)
4. XMem-no-sensory (No sensory memory)

The model without pretraining is for reference. The model without sensory memory might be more suitable for tasks without spatial continuity, like mask tracking in a multi-camera 3D reconstruction setting, though I would encourage you to try the base model as well.

Download them from [[GitHub]](https://github.com/hkchengrex/XMem/releases/tag/v1.0) or [[Google Drive]](https://drive.google.com/drive/folders/1QYsog7zNzcxGXTGBzEhMUg8QVJwZB6D1?usp=sharing).

## Long-Time Video

[[Precomputed Results]](https://drive.google.com/drive/folders/1NADcetigH6d83mUvyb2rH4VVjwFA76Lh?usp=sharing)

### Long-Time Video (1X)

| Model |  J&F | J | F |
| --- | :--:|:--:|:---:|
| XMem | 89.8±0.2 | 88.0±0.2 | 91.6±0.2 |

### Long-Time Video (3X)

| Model |  J&F | J | F |
| --- | :--:|:--:|:---:|
| XMem | 90.0±0.4 | 88.2±0.3 | 91.8±0.4 |

## DAVIS

[[Precomputed Results]](https://drive.google.com/drive/folders/1XTOGevTedRSjHnFVsZyTdxJG-iHjO0Re?usp=sharing)

### DAVIS 2016

| Model |  J&F | J | F | FPS | FPS (AMP) |
| --- | :--:|:--:|:---:|:---:|:---:|
| XMem | 91.5 | 90.4 | 92.7 | 29.6 | 40.3 |
| XMem-s012 | 92.0 | 90.7 | 93.2 | 29.6 | 40.3 |
| XMem-s2 | 90.8 | 89.6 | 91.9 | 29.6 | 40.3 |

### DAVIS 2017 validation

| Model |  J&F | J | F | FPS | FPS (AMP) |
| --- | :--:|:--:|:---:|:---:|:---:|
| XMem | 86.2 | 82.9 | 89.5 | 22.6 | 33.9 |
| XMem-s012 | 87.7 | 84.0 | 91.4 | 22.6 | 33.9 |
| XMem-s2 | 84.5 | 81.4 | 87.6 | 22.6 | 33.9 |
| XMem-no-sensory | 85.1 | - | - | 23.1 | - |

### DAVIS 2017 test-dev

| Model |  J&F | J | F |
| --- | :--:|:--:|:---:|
| XMem | 81.0 | 77.4 | 84.5 |
| XMem-s012 | 81.2 | 77.6 | 84.7 |
| XMem-s2 | 79.8 | 61.4 | 68.1 |
| XMem-s012 (600p) | 82.5 | 79.1 | 85.8 |

## YouTubeVOS

We use all available frames in YouTubeVOS by default.
See [INFERENCE.md](./INFERENCE.md) if you want to evaluate with sparse frames for some reason.

[[Precomputed Results]](https://drive.google.com/drive/folders/1P_BmOdcG6OP5mWGqWzCZrhQJ7AaLME4E?usp=sharing)

[[Precomputed Results (sparse)]](https://drive.google.com/drive/folders/1IRV1fHepufUXM45EEbtl9D4pkoh9POSZ?usp=sharing)

### YouTubeVOS 2018 validation

| Model | G | J-Seen | F-Seen | J-Unseen | F-Unseen | FPS | FPS (AMP) |
| --- | :--:|:--:|:---:|:---:|:---:|:---:|:---:|
| XMem | 85.7 | 84.6 | 89.3 | 80.2 | 88.7 | 22.6 | 31.7 |
| XMem-s012 | 86.1 | 85.1 | 89.8 | 80.3 | 89.2 | 22.6 | 31.7 |
| XMem-s2 | 84.3 | 83.9 | 88.8 | 77.7 | 86.7 | 22.6 | 31.7 |
| XMem-no-sensory | 84.4 | - | - | - | - | 23.1 | - |

### YouTubeVOS 2019 validation

| Model | G | J-Seen | F-Seen | J-Unseen | F-Unseen |
| --- | :--:|:--:|:---:|:---:|:---:|
| XMem | 85.5 | 84.3 | 88.6 | 80.3 | 88.6 |
| XMem-s012 | 85.8 | 84.8 | 89.2 | 80.3 | 88.8 |
| XMem-s2 | 84.2 | 83.8 | 88.3 | 78.1 | 86.7 |

## Multi-scale evaluation

Please see the appendix for quantitative results.

[[DAVIS-MS Precomputed Results]](https://drive.google.com/drive/folders/1H3VHKDO09izp6KR3sE-LzWbjyM-jpftn?usp=sharing)

[[YouTubeVOS-MS Precomputed Results]](https://drive.google.com/drive/folders/1ww5HVRbMKXraLd2dy1rtk6kLjEawW9Kn?usp=sharing)
