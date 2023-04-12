import os
from os import path
from argparse import ArgumentParser
import glob
from collections import defaultdict

import numpy as np
import hickle as hkl
from PIL import Image, ImagePalette

from progressbar import progressbar
from multiprocessing import Pool
from util import palette

from util.palette import davis_palette, youtube_palette
import shutil


def search_options(options, name):
    for option in options:
        if path.exists(path.join(option, name)):
            return path.join(option, name)
    else:
        return None

def process_vid(vid):
    vid_path = search_options(all_options, vid)
    if vid_path is not None:
        backward_mapping = hkl.load(path.join(vid_path, 'backward.hkl'))
    else:
        backward_mapping = None

    frames = os.listdir(path.join(all_options[0], vid))
    frames = [f for f in frames if 'backward' not in f]

    print(vid)
    if 'Y' in args.dataset:
        this_out_path = path.join(out_path, 'Annotations', vid)
    else:
        this_out_path = path.join(out_path, vid)
    os.makedirs(this_out_path, exist_ok=True)

    for f in frames:
        result_sum = None

        for option in all_options:
            if not path.exists(path.join(option, vid, f)):
                continue

            result = hkl.load(path.join(option, vid, f))
            if result_sum is None:
                result_sum = result.astype(np.float32)
            else:
                result_sum += result

        # argmax and to idx
        result_sum = np.argmax(result_sum, axis=0)

        # Remap the indices to the original domain
        if backward_mapping is not None:
            idx_mask = np.zeros_like(result_sum, dtype=np.uint8)
            for l, i in backward_mapping.items():
                idx_mask[result_sum==i] = l
        else:
            idx_mask = result_sum.astype(np.uint8)

        # Save the results
        img_E = Image.fromarray(idx_mask)
        img_E.putpalette(palette)
        img_E.save(path.join(this_out_path, f[:-4]+'.png'))


if __name__ == '__main__':
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='Y', help='D/Y, D for DAVIS; Y for YouTubeVOS')
    parser.add_argument('--list', nargs="+")
    parser.add_argument('--pattern', default=None, help='Glob patten. Can be used in place of list.')
    parser.add_argument('--output')
    parser.add_argument('--num_proc', default=4, type=int)
    args = parser.parse_args()

    out_path = args.output

    # Find the input candidates
    if args.pattern is None:
        all_options = args.list
    else:
        assert args.list is None, 'cannot specify both list and pattern'
        all_options = glob.glob(args.pattern)

    # Get the correct palette
    if 'D' in args.dataset:
        palette = ImagePalette.ImagePalette(mode='P', palette=davis_palette)
    elif 'Y' in args.dataset:
        palette = ImagePalette.ImagePalette(mode='P', palette=youtube_palette)
    else:
        raise NotImplementedError

    # Count of the number of videos in each candidate
    all_options = [path.join(o, 'Scores') for o in all_options]
    vid_count = defaultdict(int)
    for option in all_options:
        vid_in_here = sorted(os.listdir(option))
        for vid in vid_in_here:
            vid_count[vid] += 1

    all_vid = []
    count_to_vid = defaultdict(int)
    for k, v in vid_count.items():
        count_to_vid[v] += 1
        all_vid.append(k)

    for k, v in count_to_vid.items():
        print('Videos with count %d: %d' % (k, v))

    all_vid = sorted(all_vid)
    print('Total number of videos: ', len(all_vid))

    pool = Pool(processes=args.num_proc)
    for _ in progressbar(pool.imap_unordered(process_vid, all_vid), max_value=len(all_vid)):
       pass

    pool.close()
    pool.join()

    if 'D' in args.dataset:
        print('Making zip for DAVIS test-dev...')
        shutil.make_archive(args.output, 'zip', args.output)

    if 'Y' in args.dataset:
        print('Making zip for YouTubeVOS...')
        shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output, 'Annotations')
