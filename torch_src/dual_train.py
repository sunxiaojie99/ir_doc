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
import torch.nn.functional as F
from sklearn import metrics

from .data_utils import setup_seed, MyDataset, load_checkpoint, save_checkpoint
from .dual_hparams import hparams
from .dual_model import Dual_Train_Model
from .help_class import RAdam, EMA, set_lr


here = os.path.dirname(os.path.abspath(__file__))


def train(h_params):
    train_set = h_params.train_set
    test_save = h_params.test_save
    checkpoints_dir = h_params.checkpoints_dir
    checkpoint_file = h_params.checkpoint_file
    vocab_path = h_params.vocab_path
    pretrained_model_path = h_params.pretrained_model_path
    model_file = os.path.join(checkpoints_dir, 'params_model.bin')


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

    dataset = MyDataset(
        train_set,
        vocab_path,
        pretrained_model_path,
        q_max_seq_len=q_max_seq_len,
        p_max_seq_len=p_max_seq_len,
        do_lower_case=True,
        is_inference=False,
        debug=debug)
    
    train_loader = DataLoader(dataset, batch_size=batch_size)
    model = Dual_Train_Model(h_params).to(device)

    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        last_epoch = checkpoint_dict['last_epoch'] + 1
        model.load_state_dict(torch.load(model_file))  # 把之前保存的权重加载到现在的模型中
    else:
        checkpoint_dict = {}
        last_epoch = 0

    # 优化器定义
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = RAdam(optimizer_grouped_parameters, lr=lr, eps=1e-6)
    total_steps = epochs * len(train_loader)
    scaler = GradScaler()

    criterion = torch.nn.CrossEntropyLoss().to(device=device)
    running_loss = 0
    
    losses = []

    global_step = 0
    tic_train = time.time()
    log_steps = 100

    ema = EMA(model, 0.999)
    ema.register()

    for epoch in range(0, epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        step_losses = []
        for i_batch, sampled_batched in enumerate(train_loader):
            q_token_ids = sampled_batched['q_token_ids'].to(device)
            q_token_type_ids = sampled_batched['q_token_type_ids'].to(device)
            q_attention_mask = sampled_batched['q_attention_mask'].to(device)
            p_pos_token_ids = sampled_batched['p_pos_token_ids'].to(device)
            p_pos_token_type_ids = sampled_batched['p_pos_token_type_ids'].to(device)
            p_pos_attention_mask = sampled_batched['p_pos_attention_mask'].to(device)
            p_neg_token_ids = sampled_batched['p_neg_token_ids'].to(device)
            p_neg_token_type_ids = sampled_batched['p_neg_token_type_ids'].to(device)
            p_neg_attention_mask = sampled_batched['p_neg_attention_mask'].to(device)
            with autocast():
                logits = model(q_token_ids, q_token_type_ids, q_attention_mask,
                p_pos_token_ids, p_pos_token_type_ids, p_pos_attention_mask,
                p_neg_token_ids, p_neg_token_type_ids, p_neg_attention_mask)

                batch_size = logits.shape[0]
                # pre_labels = logits.argmax(1)  # [bs]

                all_labels = torch.tensor(np.array(range(0, batch_size), dtype='int64')).to(device)  # [bs]: 0,1,2..,bs-1
                
                loss = criterion(logits, all_labels)  # softmax+log+negative log likelihood loss(会取平均)

            running_loss += loss.item()
            step_losses.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            optimizer.zero_grad()

            if i_batch % 10 == 9:  # 9, 19, 29, .. 每10个batch输出一次平均loss
                losses.append(running_loss / 10)
                running_loss = 0.0

            global_step += 1

            if global_step % log_steps == 0:
                print("fold %d ,global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                        % (1, global_step, epoch, i_batch, np.mean(step_losses),
                            global_step / (time.time() - tic_train),
                            ))
        
        checkpoint_dict[epoch] = np.mean(step_losses)
        checkpoint_dict['last_epoch'] = epoch
        ema.apply_shadow()
        torch.cuda.empty_cache()  # 每个epoch结束之后清空显存，防止显存不足

    torch.save(model.state_dict(), model_file)  # 只是参数