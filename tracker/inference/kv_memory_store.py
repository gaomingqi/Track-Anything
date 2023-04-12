import torch
from typing import List

class KeyValueMemoryStore:
    """
    Works for key/value pairs type storage
    e.g., working and long-term memory
    """

    """
    An object group is created when new objects enter the video
    Objects in the same group share the same temporal extent
    i.e., objects initialized in the same frame are in the same group
    For DAVIS/interactive, there is only one object group
    For YouTubeVOS, there can be multiple object groups
    """

    def __init__(self, count_usage: bool):
        self.count_usage = count_usage

        # keys are stored in a single tensor and are shared between groups/objects
        # values are stored as a list indexed by object groups
        self.k = None
        self.v = []
        self.obj_groups = []
        # for debugging only
        self.all_objects = []

        # shrinkage and selection are also single tensors
        self.s = self.e = None

        # usage
        if self.count_usage:
            self.use_count = self.life_count = None

    def add(self, key, value, shrinkage, selection, objects: List[int]):
        new_count = torch.zeros((key.shape[0], 1, key.shape[2]), device=key.device, dtype=torch.float32)
        new_life = torch.zeros((key.shape[0], 1, key.shape[2]), device=key.device, dtype=torch.float32) + 1e-7

        # add the key
        if self.k is None:
            self.k = key
            self.s = shrinkage
            self.e = selection
            if self.count_usage:
                self.use_count = new_count
                self.life_count = new_life
        else:
            self.k = torch.cat([self.k, key], -1)
            if shrinkage is not None:
                self.s = torch.cat([self.s, shrinkage], -1)
            if selection is not None:
                self.e = torch.cat([self.e, selection], -1)
            if self.count_usage:
                self.use_count = torch.cat([self.use_count, new_count], -1)
                self.life_count = torch.cat([self.life_count, new_life], -1)

        # add the value
        if objects is not None:
            # When objects is given, v is a tensor; used in working memory
            assert isinstance(value, torch.Tensor)
            # First consume objects that are already in the memory bank
            # cannot use set here because we need to preserve order
            # shift by one as background is not part of value
            remaining_objects = [obj-1 for obj in objects]
            for gi, group in enumerate(self.obj_groups):
                for obj in group:
                    # should properly raise an error if there are overlaps in obj_groups
                    remaining_objects.remove(obj)
                self.v[gi] = torch.cat([self.v[gi], value[group]], -1)

            # If there are remaining objects, add them as a new group
            if len(remaining_objects) > 0:
                new_group = list(remaining_objects)
                self.v.append(value[new_group])
                self.obj_groups.append(new_group)
                self.all_objects.extend(new_group)
                
                assert sorted(self.all_objects) == self.all_objects, 'Objects MUST be inserted in sorted order '
        else:
            # When objects is not given, v is a list that already has the object groups sorted
            # used in long-term memory
            assert isinstance(value, list)
            for gi, gv in enumerate(value):
                if gv is None:
                    continue
                if gi < self.num_groups:
                    self.v[gi] = torch.cat([self.v[gi], gv], -1)
                else:
                    self.v.append(gv)

    def update_usage(self, usage):
        # increase all life count by 1
        # increase use of indexed elements
        if not self.count_usage:
            return
        
        self.use_count += usage.view_as(self.use_count)
        self.life_count += 1

    def sieve_by_range(self, start: int, end: int, min_size: int):
        # keep only the elements *outside* of this range (with some boundary conditions)
        # i.e., concat (a[:start], a[end:])
        # min_size is only used for values, we do not sieve values under this size
        # (because they are not consolidated)

        if end == 0:
            # negative 0 would not work as the end index!
            self.k = self.k[:,:,:start]
            if self.count_usage:
                self.use_count = self.use_count[:,:,:start]
                self.life_count = self.life_count[:,:,:start]
            if self.s is not None:
                self.s = self.s[:,:,:start]
            if self.e is not None:
                self.e = self.e[:,:,:start]
            
            for gi in range(self.num_groups):
                if self.v[gi].shape[-1] >= min_size:
                    self.v[gi] = self.v[gi][:,:,:start]
        else:
            self.k = torch.cat([self.k[:,:,:start], self.k[:,:,end:]], -1)
            if self.count_usage:
                self.use_count = torch.cat([self.use_count[:,:,:start], self.use_count[:,:,end:]], -1)
                self.life_count = torch.cat([self.life_count[:,:,:start], self.life_count[:,:,end:]], -1)
            if self.s is not None:
                self.s = torch.cat([self.s[:,:,:start], self.s[:,:,end:]], -1)
            if self.e is not None:
                self.e = torch.cat([self.e[:,:,:start], self.e[:,:,end:]], -1)
            
            for gi in range(self.num_groups):
                if self.v[gi].shape[-1] >= min_size:
                    self.v[gi] = torch.cat([self.v[gi][:,:,:start], self.v[gi][:,:,end:]], -1)

    def remove_obsolete_features(self, max_size: int):
        # normalize with life duration
        usage = self.get_usage().flatten()

        values, _ = torch.topk(usage, k=(self.size-max_size), largest=False, sorted=True)
        survived = (usage > values[-1])

        self.k = self.k[:, :, survived]
        self.s = self.s[:, :, survived] if self.s is not None else None
        # Long-term memory does not store ek so this should not be needed
        self.e = self.e[:, :, survived] if self.e is not None else None
        if self.num_groups > 1:
            raise NotImplementedError("""The current data structure does not support feature removal with 
            multiple object groups (e.g., some objects start to appear later in the video)
            The indices for "survived" is based on keys but not all values are present for every key
            Basically we need to remap the indices for keys to values
            """)
        for gi in range(self.num_groups):
            self.v[gi] = self.v[gi][:, :, survived]

        self.use_count = self.use_count[:, :, survived]
        self.life_count = self.life_count[:, :, survived]

    def get_usage(self):
        # return normalized usage
        if not self.count_usage:
            raise RuntimeError('I did not count usage!')
        else:
            usage = self.use_count / self.life_count
            return usage

    def get_all_sliced(self, start: int, end: int):
        # return k, sk, ek, usage in order, sliced by start and end

        if end == 0:
            # negative 0 would not work as the end index!
            k = self.k[:,:,start:]
            sk = self.s[:,:,start:] if self.s is not None else None
            ek = self.e[:,:,start:] if self.e is not None else None
            usage = self.get_usage()[:,:,start:]
        else:
            k = self.k[:,:,start:end]
            sk = self.s[:,:,start:end] if self.s is not None else None
            ek = self.e[:,:,start:end] if self.e is not None else None
            usage = self.get_usage()[:,:,start:end]

        return k, sk, ek, usage

    def get_v_size(self, ni: int):
        return self.v[ni].shape[2]

    def engaged(self):
        return self.k is not None

    @property
    def size(self):
        if self.k is None:
            return 0
        else:
            return self.k.shape[-1]

    @property
    def num_groups(self):
        return len(self.v)

    @property
    def key(self):
        return self.k

    @property
    def value(self):
        return self.v

    @property
    def shrinkage(self):
        return self.s

    @property
    def selection(self):
        return self.e

