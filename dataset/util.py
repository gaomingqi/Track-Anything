import numpy as np


def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for ni, l in enumerate(labels):
        Ms[ni] = (masks == l).astype(np.uint8)
        
    return Ms
