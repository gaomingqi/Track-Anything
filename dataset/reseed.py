import torch
import random

def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)