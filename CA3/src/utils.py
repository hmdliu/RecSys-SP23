
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from .dataset import *

sum_metrics = lambda d: sum(d.values())

def log_uniform(low, high, digits):
    log_rval = np.random.uniform(np.log(low), np.log(high))
    return round(float(np.exp(log_rval)), digits)

def set_seed(seed=42):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_data_split():
    print('\nProcessing data using LOO and HP strategy...')
    loo_split()
    hp_split()