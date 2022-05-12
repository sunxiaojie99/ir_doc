from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from curses.ascii import CR

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

from ..help_class import RAdam, EMA, set_lr
from ..data_utils import setup_seed, load_checkpoint, save_checkpoint
from .cross_data_utils import CrossDataset
from .cross_model import Cross_Train_Model


here = os.path.dirname(os.path.abspath(__file__))


def train(h_params):
    train_set = h_params.train_set
    checkpoints_dir = h_params.checkpoints_dir
    checkpoint_file = h_params.checkpoint_file
    vocab_path = h_params.vocab_path
    pretrained_model_path = h_params.pretrained_model_path
    model_file = os.path.join(checkpoints_dir, 'cross_params.bin')
    log_dir = h_params.log_dir
    device = h_params.device

    seed = h_params.random_seed
    setup_seed(seed)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    lr = h_params.lr
    epochs = h_params.epochs
    batch_size = h_params.batch_size
    max_seq_len = h_params.max_seq_len
    weight_decay = h_params.weight_decay
    warmup_proportion = h_params.warmup_proportion
    num_labels = h_params.num_labels

    debug = h_params.debug

    if debug:
        epochs = 2

    dataset = CrossDataset(
        train_set,
        vocab_path,
        pretrained_model_path,
        max_seq_len=max_seq_len,
        do_lower_case=True,
        shuffle=True,
        debug=debug)
    
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = Cross_Train_Model(h_params, is_predict=False).to(device)

    # 继续之前的训练

    # if os.path.exists(checkpoint_file):
    #     checkpoint_dict = load_checkpoint(checkpoint_file)
    #     last_epoch = checkpoint_dict['last_epoch'] + 1
    #     model.load_state_dict(torch.load(model_file))  # 把之前保存的权重加载到现在的模型中
    # else:
    #     checkpoint_dict = {}
    #     last_epoch = 0
    checkpoint_dict = {}

    # 优化器定义
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = RAdam(optimizer_grouped_parameters, lr=lr, eps=1e-6)
    total_steps = epochs * len(train_loader)
    scaler = GradScaler()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    running_loss = 0
    
    losses = []

    global_step = 0
    tic_train = time.time()
    log_steps = 100

    ema = EMA(model, 0.999)
    ema.register()

    for epoch in tqdm(range(0, epochs)):
        print("Epoch: {}".format(epoch))
        model.train()
        step_losses = []
        labels_true = []
        labels_pred = []
        for i_batch, sampled_batched in enumerate(train_loader):
            token_ids = sampled_batched['token_ids'].to(device)
            token_type_ids = sampled_batched['token_type_ids'].to(device)
            attention_mask = sampled_batched['attention_mask'].to(device)
            label_ids = sampled_batched['label_id'].to(device)
            with autocast():
                logits = model(token_ids, token_type_ids, attention_mask)
                loss = criterion(logits, label_ids)
            
            pred_tag_ids = logits.argmax(1)
            labels_true.extend(label_ids.tolist())
            labels_pred.extend(pred_tag_ids.tolist())

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
        
        epoch_model_file = os.path.join(checkpoints_dir, 'epoch_{}_cross_params.bin'.format(epoch))
        torch.save(model.state_dict(), epoch_model_file)  # 每个epoch都保存一下

        accuracy = metrics.accuracy_score(labels_true, labels_pred)
        f1 = metrics.f1_score(labels_true, labels_pred, average='macro')

        if checkpoint_dict.get('epoch_acc'):
            checkpoint_dict['epoch_acc'][epoch] = accuracy
            checkpoint_dict['epoch_f1'][epoch] = f1
            checkpoint_dict['epoch_loss'][epoch] = np.mean(step_losses)
        else:
            checkpoint_dict['epoch_acc'] = {epoch: accuracy}
            checkpoint_dict['epoch_f1'] = {epoch: f1}
            checkpoint_dict['epoch_loss'] = {epoch: np.mean(step_losses)}


        checkpoint_dict['last_epoch'] = epoch
        save_checkpoint(checkpoint_dict, checkpoint_file)
        ema.apply_shadow()
        torch.cuda.empty_cache()  # 每个epoch结束之后清空显存，防止显存不足
    
    torch.save(model.state_dict(), model_file)  # 只是参数



    
    
