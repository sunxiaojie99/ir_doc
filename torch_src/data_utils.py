import re
import os
import json

import torch
from torch.utils.data import Dataset
import transformers
from tqdm import tqdm
import numpy as np
import random
import time
from .tokenization import convert_to_unicode, FullTokenizer
from datasets import Dataset

here = os.path.dirname(os.path.abspath(__file__))  # 当前文件的目录


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


def csv_reader(fd, delimiter='\t'):
    def gen():
        for i, line in enumerate(fd):
            slots = line.rstrip('\n').split(delimiter)
            if len(slots) == 1:
                yield slots,
            else:
                yield slots
    return gen()


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


def read_data(data_file_path, tokenizer, q_max_seq_len,
              p_max_seq_len, label_map_config=None, is_dubug=False, shuffle=False):
    """query null para_text_pos null para_text_neg null
    return:
    ['token_ids_q', 'text_type_ids_q', 'position_ids_q', \
                 'token_ids_p_pos', 'text_type_ids_p_pos', 'position_ids_p_pos', \
                 'token_ids_p_neg', 'text_type_ids_p_neg', 'position_ids_p_neg',
                 'label_id', 'qid'
                ]
    """
    if label_map_config:
        with open(label_map_config, encoding='utf8') as f:
            label_map = json.load(f)
    else:
        label_map = None

    token_ids_q_list = []
    token_ids_p_pos_list = []
    token_ids_p_neg_list = []
    with open(data_file_path, 'r', encoding='utf-8') as f:
        # reader = csv_reader(f)
        if is_dubug:
            lines = f.readlines()[:128]
        else:
            lines = f.readlines()
        if shuffle:
            np.random.shuffle(lines)
        begin_time = time.time()
        count = 0
        for l in tqdm(lines):
            line = l.rstrip('\n').split('\t')
            assert len(line) == 6, line
            # query\ttitle_pos\tpara_pos\ttitle_neg\tpara_neg\tlabel
            query = line[0]
            p_pos = line[2]
            p_neg = line[4]
            query = convert_to_unicode(query)
            tokens_query = tokenizer.tokenize(query)

            # 只裁剪单独的一个，传进去一个空list即可
            truncate_seq_pair([], tokens_query, q_max_seq_len-2)

            # pos para
            p_pos = convert_to_unicode(p_pos)
            tokens_p_pos = tokenizer.tokenize(p_pos)

            truncate_seq_pair([], tokens_p_pos, p_max_seq_len-3)

            # neg para
            para_neg = convert_to_unicode(p_neg)
            tokens_para_neg = tokenizer.tokenize(para_neg)

            truncate_seq_pair([], tokens_para_neg, p_max_seq_len-3)

            token_ids_q_list.append(tokens_query)
            token_ids_p_pos_list.append(tokens_p_pos)
            token_ids_p_neg_list.append(tokens_para_neg)
            count += 1
        end_time = time.time()
        print('one example time cost:', (end_time-begin_time)/count)

    return token_ids_q_list, token_ids_p_pos_list, token_ids_p_neg_list


def read_dev(data_file_path, tokenizer, q_max_seq_len,
             p_max_seq_len, label_map_config=None, is_dubug=False):
    """
    for query: query_text null null null
    for doc: null null passage_text null
    """
    token_ids_q_list = []
    token_ids_p_list = []
    with open(data_file_path, 'r', encoding='utf8') as f:
        # reader = csv_reader(f)
        if is_dubug:
            lines = f.readlines()[:128]
        else:
            lines = f.readlines()
        
        for l in tqdm(lines):
            line = l.rstrip('\n').split('\t')
            assert len(line) == 4, line
            query = line[0]
            passage = line[2]
            
            query = convert_to_unicode(query)
            tokens_query = tokenizer.tokenize(query)
            # 只裁剪单独的一个，传进去一个空list即可
            truncate_seq_pair([], tokens_query, q_max_seq_len-2)
            token_ids_q_list.append(tokens_query)

            passage = convert_to_unicode(passage)
            tokens_passage = tokenizer.tokenize(passage)
            truncate_seq_pair([], tokens_passage, p_max_seq_len-2)
            token_ids_p_list.append(tokens_passage)

    return token_ids_q_list, token_ids_p_list


def make_inference_dataset(data_file_path, vocab_path, pretrained_model_path,
                 q_max_seq_len=128, p_max_seq_len=512, do_lower_case=True):
    
    with open(data_file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    text_data = {"query": [], "passage": []}
    for l in tqdm(lines):
        line = l.rstrip('\n').split('\t')
        assert len(line) == 4, line
        query = line[0]
        passage = line[2]
        text_data["query"].append(query) 
        text_data["passage"].append(passage) 

    def encode(examples):
        query = examples['query']
        passage = examples['passage']

        query = convert_to_unicode(query)
        tokens_query = tokenizer.tokenize(query)
        truncate_seq_pair([], tokens_query, q_max_seq_len-2)

        passage = convert_to_unicode(passage)
        tokens_passage = tokenizer.tokenize(passage)
        truncate_seq_pair([], tokens_passage, p_max_seq_len-2)
        encoded_q = bert_tokenizer.encode_plus(tokens_query, max_length=q_max_seq_len,
                                                    pad_to_max_length=True, truncation=True)
        encoded_p = bert_tokenizer.encode_plus(tokens_passage, max_length=p_max_seq_len,
                                                    pad_to_max_length=True, truncation=True)

        sample = {
            "token_ids_q": encoded_q['input_ids'],
            "token_type_ids_q": encoded_q['token_type_ids'],
            "attention_mask_q": encoded_q['attention_mask'],
            "token_ids_p": encoded_p['input_ids'],
            "token_type_ids_p": encoded_p['token_type_ids'],
            "attention_mask_p": encoded_p['attention_mask']
        }
        return sample

    dataset = Dataset.from_dict(text_data)
    tokenizer = FullTokenizer(vocab_file=vocab_path, do_lower_case=do_lower_case)
    bert_tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_path)
    dataset = dataset.map(encode, num_proc=5)
    dataset.set_format(type='torch', columns=['token_ids_q', 'token_type_ids_q', 'attention_mask_q', 
                                                'token_ids_p', 'token_type_ids_p', 'attention_mask_p'])
    return dataset


class MyDataset(Dataset):
    def __init__(self,
                 data_file_path,
                 vocab_path,
                 pretrained_model_path,
                 q_max_seq_len=128,
                 p_max_seq_len=512,
                 do_lower_case=True,
                 debug=False,
                 shuffle=False):
        self.q_max_seq_len = q_max_seq_len
        self.p_max_seq_len = p_max_seq_len
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            pretrained_model_path)
        self.vocab = self.tokenizer.vocab

        self.token_ids_q_list, self.token_ids_p_pos_list, self.token_ids_p_neg_list = read_data(
            data_file_path, self.tokenizer, q_max_seq_len, p_max_seq_len, is_dubug=debug, shuffle=shuffle)

        if debug:
            print('dug!!!!')
            self.token_ids_q_list = self.token_ids_q_list
            self.token_ids_p_pos_list = self.token_ids_p_pos_list
            self.token_ids_p_neg_list = self.token_ids_p_neg_list

        self.num = len(self.token_ids_q_list)
        print('样本数: ', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 如果是一个tensor类型，变为list

        # ['微', '信', '分', '享', '链', '接', '打', '开', 'app']
        sample_token_ids_q = self.token_ids_q_list[idx]
        sample_token_ids_p_pos = self.token_ids_p_pos_list[idx]
        sample_token_ids_p_neg = self.token_ids_p_neg_list[idx]
        encoded_q = self.bert_tokenizer.encode_plus(sample_token_ids_q, max_length=self.q_max_seq_len,
                                                    padding='max_length', truncation=True)
        # p self.bert_tokenizer.
        # convert_ids_to_tokens(self.bert_tokenizer.encode_plus(self.token_ids_q_list[0])['input_ids'])
        # ['[CLS]', '微', '信', '分', '享', '链', '接', '打', '开', 'app', '[SEP]']

        encoded_p_pos = self.bert_tokenizer.encode_plus(sample_token_ids_p_pos, max_length=self.p_max_seq_len,
                                                        padding='max_length', truncation=True)
        encoded_p_neg = self.bert_tokenizer.encode_plus(sample_token_ids_p_neg, max_length=self.p_max_seq_len,
                                                        padding='max_length', truncation=True)

        sample = {
            "q_token_ids": torch.tensor(encoded_q['input_ids']),
            "q_token_type_ids": torch.tensor(encoded_q['token_type_ids']),
            "q_attention_mask": torch.tensor(encoded_q['attention_mask']),
            "p_pos_token_ids": torch.tensor(encoded_p_pos['input_ids']),
            "p_pos_token_type_ids": torch.tensor(encoded_p_pos['token_type_ids']),
            "p_pos_attention_mask": torch.tensor(encoded_p_pos['attention_mask']),
            "p_neg_token_ids": torch.tensor(encoded_p_neg['input_ids']),
            "p_neg_token_type_ids": torch.tensor(encoded_p_neg['token_type_ids']),
            "p_neg_attention_mask": torch.tensor(encoded_p_neg['attention_mask']),
        }
        return sample


class InferDataset(Dataset):
    def __init__(self,
                 data_file_path,
                 vocab_path,
                 pretrained_model_path,
                 q_max_seq_len=128,
                 p_max_seq_len=512,
                 do_lower_case=True,
                 test_type='query',
                 debug=False):
        self.q_max_seq_len = q_max_seq_len
        self.p_max_seq_len = p_max_seq_len
        self.tokenizer = FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            pretrained_model_path)
        self.vocab = self.tokenizer.vocab
        self.test_type = test_type

        # if test_type == 'query':
        #     print('=====InferDataset: query')
        #     self.token_ids_list = read_dev_query(
        #         data_file_path, self.tokenizer, q_max_seq_len, p_max_seq_len, is_dubug=debug)
        # elif test_type == 'passage':
        #     print('=====InferDataset: passage')
        #     self.token_ids_list = read_dev_passage(
        #         data_file_path, self.tokenizer, q_max_seq_len, p_max_seq_len, is_dubug=debug)

        self.token_ids_q_list, self.token_ids_p_list = read_dev(
            data_file_path, self.tokenizer, q_max_seq_len, p_max_seq_len, is_dubug=debug)
            
        if debug:
            print('dug!!!!')
            self.token_ids_q_list = self.token_ids_q_list[:64]
            self.token_ids_p_list = self.token_ids_p_list[:64]

        self.num = len(self.token_ids_q_list)
        print('样本数: ', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 如果是一个tensor类型，变为list

        # ['微', '信', '分', '享', '链', '接', '打', '开', 'app']
        sample_token_ids_q = self.token_ids_q_list[idx]
        encoded_q = self.bert_tokenizer.encode_plus(sample_token_ids_q, max_length=self.q_max_seq_len,
                                                    pad_to_max_length=True, truncation=True)
        # p self.bert_tokenizer.
        # convert_ids_to_tokens(self.bert_tokenizer.encode_plus(self.token_ids_q_list[0])['input_ids'])
        # ['[CLS]', '微', '信', '分', '享', '链', '接', '打', '开', 'app', '[SEP]']
        # ['微', '信', '分', '享', '链', '接', '打', '开', 'app']
        sample_token_ids_p = self.token_ids_p_list[idx]
        encoded_p = self.bert_tokenizer.encode_plus(sample_token_ids_p, max_length=self.p_max_seq_len,
                                                    pad_to_max_length=True, truncation=True)

        sample = {
            "token_ids_q": torch.tensor(encoded_q['input_ids']),
            "token_type_ids_q": torch.tensor(encoded_q['token_type_ids']),
            "attention_mask_q": torch.tensor(encoded_q['attention_mask']),
            "token_ids_p": torch.tensor(encoded_p['input_ids']),
            "token_type_ids_p": torch.tensor(encoded_p['token_type_ids']),
            "attention_mask_p": torch.tensor(encoded_p['attention_mask'])
        }
        return sample