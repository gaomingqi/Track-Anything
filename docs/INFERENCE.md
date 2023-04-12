# Inference

What is palette? Why is the output a "colored image"? How do I make those input masks that look like color images? See [PALETTE.md](./PALETTE.md).

1. Set up the datasets following [GETTING_STARTED.md](./GETTING_STARTED.md).
2. Download the pretrained models either using `./scripts/download_models.sh`, or manually and put them in `./saves` (create the folder if it doesn't exist). You can download them from [[GitHub]](https://github.com/hkchengrex/XMem/releases/tag/v1.0) or [[Google Drive]](https://drive.google.com/drive/folders/1QYsog7zNzcxGXTGBzEhMUg8QVJwZB6D1?usp=sharing).

All command-line inference are accessed with `eval.py`. See [RESULTS.md](./RESULTS.md) for an explanation of FPS and the differences between different models.

## Usage

```
python eval.py --model [path to model file] --output [where to save the output] --dataset [which dataset to evaluate on] --split [val for validation or test for test-dev]
```

See the code for a complete list of available command-line arguments.

Examples:
(``--model`` defaults to `./saves/XMem.pth`)

DAVIS 2017 validation:

```
python eval.py --output ../output/d17 --dataset D17
```

DAVIS 2016 validation:

```
python eval.py --output ../output/d16 --dataset D16
```

DAVIS 2017 test-dev:

```
python eval.py --output ../output/d17-td --dataset D17 --split test
```

YouTubeVOS 2018 validation:

```
python eval.py --output ../output/y18 --dataset Y18
```

Long-Time Video (3X) (note that `mem_every`, aka `r`, is set differently):

```
python eval.py --output ../output/lv3 --dataset LV3 --mem_every 10
```

## Getting quantitative results

We do not provide any tools for getting quantitative results here. We used the followings to get the results reported in the paper:

- DAVIS 2017 validation: [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)
- DAVIS 2016 validation: [davis2016-evaluation](https://github.com/hkchengrex/davis2016-evaluation) (Unofficial)
- DAVIS 2017 test-dev: [CodaLab](https://competitions.codalab.org/competitions/20516#participate)
- YouTubeVOS 2018 validation: [CodaLab](https://competitions.codalab.org/competitions/19544#results)
- YouTubeVOS 2019 validation: [CodaLab](https://competitions.codalab.org/competitions/20127#participate-submit_results)
- Long-Time Video: [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)

(For the Long-Time Video dataset, point `--davis_path` to either `long_video_davis` or `long_video_davis_x3`)

## On custom data

Structure your custom data like this:

```bash
├── custom_data_root
│   ├── JPEGImages
│   │   ├── video1
│   │   │   ├── 00001.jpg
│   │   │   ├── 00002.jpg
│   │   │   ├── ...
│   │   └── ...
│   ├── Annotations
│   │   ├── video1
│   │   │   ├── 00001.png
│   │   │   ├── ...
│   │   └── ...
```

We use `sort` to determine frame order. The annotations do not have have to be complete (e.g., first-frame only is fine). We use PIL to read the annotations and `np.unique` to determine objects. PNG palette will be used automatically if exists.

Then, point `--generic_path` to `custom_data_root` and specify `--dataset` as `G` (for generic).

## Multi-scale evaluation

Multi-scale evaluation is done in two steps. We first compute and save the object probabilities maps for different settings independently on hard-disks as `hkl` (hickle) files. Then, these maps are merged together with `merge_multi_score.py`.

Example for DAVIS 2017 validation MS:

Step 1 (can be done in parallel with multiple GPUs):

```bash
python eval.py --output ../output/d17_ms/720p --mem_every 3 --dataset D17 --save_scores --size 720
python eval.py --output ../output/d17_ms/720p_flip --mem_every 3 --dataset D17 --save_scores --size 720 --flip
```

Step 2:

```bash
python merge_multi_scale.py --dataset D --list ../output/d17_ms/720p ../output/d17_ms/720p_flip --output ../output/d17_ms_merged
```

Instead of `--list`, you can also use `--pattern` to specify a glob pattern. It also depends on your shell (e.g., `zsh` or `bash`).

## Advanced usage

To develop your own evaluation interface, see `./inference/` -- most importantly, `inference_core.py`.
