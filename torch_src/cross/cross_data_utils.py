import re
import os
import json

import torch
from torch.utils.data import Dataset
import transformers
from tqdm import tqdm
import numpy as np
import random
from ..tokenization import convert_to_unicode, FullTokenizer


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_data(data_file_path, tokenizer, max_seq_len, is_dubug=False):
    """query null para_text label
    return:
    token_ids_q_list, token_ids_p_list
    """

    token_ids_query_list = []
    token_ids_p_list = []
    labels = []

    with open(data_file_path, 'r', encoding='utf8') as f:
        # reader = csv_reader(f)
        if is_dubug:
            lines = f.readlines()[:128]
        else:
            lines = f.readlines()
        for l in tqdm(lines):
            line = l.rstrip('\n').split('\t')
            assert len(line) == 4, line
            # query null para_text label
            query = line[0]
            passage = line[2]
            label = line[3]
            query = convert_to_unicode(query)
            tokens_query = tokenizer.tokenize(query)

            # para
            para = convert_to_unicode(passage)
            tokens_para = tokenizer.tokenize(para)

            truncate_seq_pair(tokens_query, tokens_para, max_seq_len-3)

            token_ids_query_list.append(tokens_query)
            token_ids_p_list.append(tokens_para)
            labels.append(int(label))

    return token_ids_query_list, token_ids_p_list, labels


class CrossDataset(Dataset):
    def __init__(self,
                 data_file_path,
                 vocab_path,
                 pretrained_model_path,
                 max_seq_len=512,
                 do_lower_case=True,
                 debug=False):
        self.max_seq_len = max_seq_len
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            pretrained_model_path)
        self.vocab = self.tokenizer.vocab

        self.token_ids_query_list, self.token_ids_p_list, self.labels = read_data(
            data_file_path, self.tokenizer, max_seq_len, is_dubug=debug)

        if debug:
            print('dug!!!!')
            self.token_ids_query_list = self.token_ids_query_list[:32]
            self.token_ids_p_list = self.token_ids_p_list[:32]
            self.labels = self.labels[:32]
        self.num = len(self.token_ids_query_list)
        print('样本数: ', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 如果是一个tensor类型，变为list

        query = self.token_ids_query_list[idx]
        para = self.token_ids_p_list[idx]
        sample_label = self.labels[idx]
        # 添加[CLS],[SEP], [SEP]
        encoded = self.bert_tokenizer.encode_plus(query, para,
                                                  padding='max_length', truncation=True, max_length=self.max_seq_len)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'label_id': torch.tensor(sample_label)
        }
        return sample

class CrossDataset_Test(Dataset):
    def __init__(self,
                 data_file_path,
                 vocab_path,
                 pretrained_model_path,
                 max_seq_len=512,
                 do_lower_case=True,
                 debug=False):
        self.max_seq_len = max_seq_len
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            pretrained_model_path)
        self.vocab = self.tokenizer.vocab

        self.token_ids_query_list, self.token_ids_p_list, self.labels = read_data(
            data_file_path, self.tokenizer, max_seq_len, is_dubug=debug)

        if debug:
            print('dug!!!!')
            self.token_ids_query_list = self.token_ids_query_list[:32]
            self.token_ids_p_list = self.token_ids_p_list[:32]
            self.labels = self.labels[:32]
        self.num = len(self.token_ids_query_list)
        print('样本数: ', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 如果是一个tensor类型，变为list

        query = self.token_ids_query_list[idx]
        para = self.token_ids_p_list[idx]
        sample_label = self.labels[idx]
        # 添加[CLS],[SEP], [SEP]
        encoded = self.bert_tokenizer.encode_plus(query, para,
                                                  padding='max_length', truncation=True, max_length=self.max_seq_len)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'label_id': torch.tensor(sample_label)
        }
        return sample
