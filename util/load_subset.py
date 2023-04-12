"""
load_subset.py - Presents a subset of data
DAVIS - only the training set
YouTubeVOS - I manually filtered some erroneous ones out but I haven't checked all
"""


def load_sub_davis(path='util/davis_subset.txt'):
    with open(path, mode='r') as f:
        subset = set(f.read().splitlines())
    return subset

def load_sub_yv(path='util/yv_subset.txt'):
    with open(path, mode='r') as f:
        subset = set(f.read().splitlines())
    return subset
