
import numpy as np
import os
import torch
import random

import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_random_seed(1, deterministic=False)

for k in range(5):
    os.chdir('your own path')
    os.system('python train.py --zhongzi '+str(k))
