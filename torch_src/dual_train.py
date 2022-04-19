from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import json
import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import logging

from .data_utils import setup_seed

here = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger()


def train(h_params):
    train_set = h_params.train_set
    test_save = h_params.test_save
    checkpoints_dir = h_params.checkpoints_dir
    pretrained_model_path = h_params.pretrained_model_path
    log_dir = h_params.log_dir
    device = h_params.device

    seed = h_params.random_seed
    setup_seed(seed)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    lr = h_params.lr
    epoch = h_params.epoch
    batch_size = h_params.batch_size
    q_max_seq_len = h_params.q_max_seq_len
    p_max_seq_len = h_params.p_max_seq_len
    weight_decay = h_params.weight_decay
    warmup_proportion = h_params.warmup_proportion

    debug = h_params.debug

    if debug:
        epochs = 2
    
