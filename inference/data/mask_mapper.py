import numpy as np
import torch

from dataset.util import all_to_onehot


class MaskMapper:
    """
    This class is used to convert a indexed-mask to a one-hot representation.
    It also takes care of remapping non-continuous indices
    It has two modes:
        1. Default. Only masks with new indices are supposed to go into the remapper.
        This is also the case for YouTubeVOS.
        i.e., regions with index 0 are not "background", but "don't care".

        2. Exhaustive. Regions with index 0 are considered "background".
        Every single pixel is considered to be "labeled".
    """
    def __init__(self):
        self.labels = []
        self.remappings = {}

        # if coherent, no mapping is required
        self.coherent = True

    def convert_mask(self, mask, exhaustive=False):
        # mask is in index representation, H*W numpy array
        labels = np.unique(mask).astype(np.uint8)
        labels = labels[labels!=0].tolist()

        new_labels = list(set(labels) - set(self.labels))
        if not exhaustive:
            assert len(new_labels) == len(labels), 'Old labels found in non-exhaustive mode'

        # add new remappings
        for i, l in enumerate(new_labels):
            self.remappings[l] = i+len(self.labels)+1
            if self.coherent and i+len(self.labels)+1 != l:
                self.coherent = False

        if exhaustive:
            new_mapped_labels = range(1, len(self.labels)+len(new_labels)+1)
        else:
            if self.coherent:
                new_mapped_labels = new_labels
            else:
                new_mapped_labels = range(len(self.labels)+1, len(self.labels)+len(new_labels)+1)

        self.labels.extend(new_labels)
        mask = torch.from_numpy(all_to_onehot(mask, self.labels)).float()

        # mask num_objects*H*W
        return mask, new_mapped_labels


    def remap_index_mask(self, mask):
        # mask is in index representation, H*W numpy array
        if self.coherent:
            return mask

        new_mask = np.zeros_like(mask)
        for l, i in self.remappings.items():
            new_mask[mask==i] = l
        return new_mask