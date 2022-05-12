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
    all_score_save_path = h_params.all_score_save_path

    batch_size = h_params.batch_size
    max_seq_len = h_params.max_seq_len

    

    dataset = CrossDataset(
        test_set,
        vocab_path,
        pretrained_model_path,
        max_seq_len=max_seq_len,
        do_lower_case=True,
        shuffle=False,
        debug=debug)
    
    test_loader = DataLoader(dataset, batch_size=batch_size)

    model = Cross_Train_Model(h_params, is_predict=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    labels_pred = []
    labels_pred_score = []
    all_pred_score = []
    all_count = 0

    with torch.no_grad():

        for i, sampled_batched in enumerate(tqdm(test_loader, desc='predict')):
            token_ids = sampled_batched['token_ids'].to(device)
            bs = token_ids.shape[0]
            all_count += bs
            token_type_ids = sampled_batched['token_type_ids'].to(device)
            attention_mask = sampled_batched['attention_mask'].to(device)

            logits = model(token_ids, token_type_ids, attention_mask)  # 和forward对应
            logits = torch.softmax(logits, 1)
            # [batch_size, label_set_size]
            index = torch.arange(0, bs, 1)
            pred_tag_ids = logits.argmax(1)  # 0/1，看看谁最大
            pred_score = logits[index, pred_tag_ids]  # 每个预测的分数
            pred_score = torch.mul(pred_score, pred_tag_ids)  # 对于0的不保留分数
            labels_pred.extend(pred_tag_ids.tolist())
            labels_pred_score.extend(pred_score.tolist())
            
            all_pred_score.extend(logits.tolist())  # 保存所有的分数，用于debug..
            
    
    print('所有预测的样本数：', all_count)
    print('score 条数:', len(labels_pred))
    with open(save_path, 'w') as f:
        for p in all_pred_score:
            f.write('{}\n'.format(p[1]))  # 只记录预测为1的分数
    
    with open(all_score_save_path, 'w') as f:
        for p in all_pred_score:
            f.write('{}\t{}\n'.format(p[0], p[1]))


