# Source: DeepAAI (https://github.com/enai4bio/DeepAAI/blob/main/utils/index_map.py)
import numpy as np
import random, sys
import torch



def get_map_index_for_sub_arr(sub_arr, raw_arr):
    map_arr = np.zeros(raw_arr.shape)
    map_arr.fill(-1)
    for idx in range(sub_arr.shape[0]):
        map_arr[sub_arr[idx]] = idx
    return map_arr



def set_random_seed():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True        