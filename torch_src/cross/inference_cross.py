from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
import imp

import json
import os
from statistics import mode
import time
from sympy import im
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch.nn.functional as F
from sklearn import metrics

from .cross_model import Cross_Train_Model
from .cross_data_utils import CrossDataset


def infer(h_params):
    test_set = h_params.test_set
    model_path = h_params.model_path
    pretrained_model_path = h_params.pretrained_model_path
    vocab_path = h_params.vocab_path
    device = h_params.device
    debug = h_params.debug
    save_path = h_params.save_path

    batch_size = h_params.batch_size
    max_seq_len = h_params.max_seq_len

    

    dataset = CrossDataset(
        test_set,
        vocab_path,
        pretrained_model_path,
        max_seq_len=max_seq_len,
        do_lower_case=True,
        debug=debug)
    
    test_loader = DataLoader(dataset, batch_size=batch_size)

    model = Cross_Train_Model(h_params, is_predict=True)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    labels_pred = []
    all_count = 0

    with torch.no_grad():

        for i, sampled_batched in enumerate(tqdm(test_loader, desc='predict')):
            all_count += len(sampled_batched)
            token_ids = sampled_batched['token_ids'].to(device)
            token_type_ids = sampled_batched['token_type_ids'].to(device)
            attention_mask = sampled_batched['attention_mask'].to(device)

            logits = model(token_ids, token_type_ids, attention_mask)  # 和forward对应
            # [batch_size, label_set_size]
            pred_tag_ids = logits.argmax(1)
            labels_pred.extend(pred_tag_ids.tolist())
    
    print('所有预测的样本数：', all_count)
    print('score 条数:', len(labels_pred))
    with open(save_path, 'w') as f:
        for p in labels_pred:
            f.write('{}\n'.format(p))


