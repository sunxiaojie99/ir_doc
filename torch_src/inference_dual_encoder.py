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
import faiss

from .dual_model import Dual_Train_Model
from .data_utils import InferDataset


def build_engine(para_emb_list, dim):
    faiss_res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(dim)  # 点乘，归一化的向量点乘即cosine相似度（越大越好）
    gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, index)
    # add paragraph embedding
    p_emb_matrix = np.asarray(para_emb_list)
    gpu_index.add(p_emb_matrix.astype('float32'))  # add vectors to the index
    # print(index.ntotal)
    # print ("insert done", file=sys.stderr)
    index = faiss.index_gpu_to_cpu(gpu_index)
    return index


def infer(h_params):
    test_type = h_params.test_type
    test_set = h_params.test_set
    out_file_name = h_params.out_file_name
    test_save = h_params.test_save
    checkpoints_dir = h_params.checkpoints_dir
    model_path = h_params.model_path
    pretrained_model_path = h_params.pretrained_model_path
    vocab_path = h_params.vocab_path
    device = h_params.device
    batch_size = h_params.batch_size
    q_max_seq_len = h_params.q_max_seq_len
    p_max_seq_len = h_params.p_max_seq_len
    hidden_size = h_params.hidden_size
    debug = h_params.debug

    model = Dual_Train_Model(h_params, is_prediction=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 打印模型形状
    # for name, parameters in model.named_parameters():
    # 	print(name, ':', parameters.size())

    model.eval()

    dataset = InferDataset(
        test_set,
        vocab_path,
        pretrained_model_path,
        test_type=test_type,
        q_max_seq_len=q_max_seq_len,
        p_max_seq_len=p_max_seq_len,
        do_lower_case=True,
        debug=debug)

    data_loader = DataLoader(dataset, batch_size=batch_size)

    para_embs = []

    with torch.no_grad():
        for i_batch, sampled_batched in enumerate(data_loader):
            token_ids_q = sampled_batched['token_ids_q'].to(device)
            token_type_ids_q = sampled_batched['token_type_ids_q'].to(device)
            attention_mask_q = sampled_batched['attention_mask_q'].to(device)

            token_ids_p = sampled_batched['token_ids_p'].to(device)
            token_type_ids_p = sampled_batched['token_type_ids_p'].to(device)
            attention_mask_p = sampled_batched['attention_mask_p'].to(device)

            graph_vars = model(token_ids_q, token_type_ids_q, attention_mask_q,
                            token_ids_p, token_type_ids_p, attention_mask_p,
                            token_ids_p, token_type_ids_p, attention_mask_p)
            q_rep = graph_vars['q_rep']
            p_rep = graph_vars['p_rep']

            if test_type == 'query':
                for item in q_rep:
                    para_embs.append(item.cpu().detach().numpy())
            elif test_type == 'passage':
                for item in p_rep:
                    para_embs.append(item.cpu().detach().numpy())

    print('predict embs cnt: {}'.format(len(para_embs)))

    if test_type == 'passage':
        engine = build_engine(para_embs, hidden_size)
        faiss.write_index(engine, out_file_name)
        print("create index done!")
    elif test_type == 'query':
        emb_matrix = np.asarray(para_embs)
        np.save(out_file_name + '.npy', emb_matrix)
        print("save to npy file!")
