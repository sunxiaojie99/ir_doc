import re
import os
import json

import torch
from torch.utils.data import Dataset
import transformers
from tqdm import tqdm
import numpy as np
import random
import tokenization

here = os.path.dirname(os.path.abspath(__file__))  # 当前文件的目录


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True


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
              p_max_seq_len, is_inference=False, label_map_config=None):
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
    with open(data_file_path, 'r', encoding='utf8') as f:
        reader = csv_reader(f)
        for line in reader:
            assert len(line) == 6, line
            # query\ttitle_pos\tpara_pos\ttitle_neg\tpara_neg\tlabel
            query = line[0]
            p_pos = line[2]
            p_neg = line[4]
            query = tokenization.convert_to_unicode(query)
            tokens_query = tokenizer.tokenize(query)

            # 只裁剪单独的一个，传进去一个空list即可
            truncate_seq_pair([], tokens_query, q_max_seq_len-2)

            # pos para
            p_pos = tokenization.convert_to_unicode(p_pos)
            tokens_p_pos = tokenizer.tokenize(p_pos)

            truncate_seq_pair([], tokens_p_pos, p_max_seq_len-3)

            # neg para
            para_neg = tokenization.convert_to_unicode(p_neg)
            tokens_para_neg = tokenizer.tokenize(para_neg)

            truncate_seq_pair([], tokens_para_neg, p_max_seq_len-3)

            token_ids_q_list.append(tokens_query)
            token_ids_p_pos_list.append(tokens_p_pos)
            token_ids_p_neg_list.append(tokens_para_neg)

    return token_ids_q_list, token_ids_p_pos_list, token_ids_p_neg_list


class MyDataset(Dataset):
    def __init__(self,
                 data_file_path,
                 vocab_path,
                 pretrained_model_path,
                 q_max_seq_len=128,
                 p_max_seq_len=512,
                 do_lower_case=True,
                 is_inference=False,
                 debug=False):
        self.q_max_seq_len = q_max_seq_len
        self.p_max_seq_len = p_max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            pretrained_model_path)
        self.vocab = self.tokenizer.vocab
        self.is_inference = is_inference

        self.token_ids_q_list, self.token_ids_p_pos_list, self.token_ids_p_neg_list = read_data(
            data_file_path, self.tokenizer, q_max_seq_len, p_max_seq_len, is_inference=False)

        if debug:
            print('dug!!!!')
            self.token_ids_q_list = self.token_ids_q_list[:50]
            self.token_ids_p_pos_list = self.token_ids_p_pos_list[:50]
            self.token_ids_p_neg_list = self.token_ids_p_neg_list[:50]

        self.num = len(self.token_ids_q_list)
        print('样本数: ', self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 如果是一个tensor类型，变为list

        sample_token_ids_q = self.token_ids_q_list[idx]
        sample_token_ids_p_pos = self.token_ids_p_pos_list[idx]
        sample_token_ids_p_neg = self.token_ids_p_neg_list[idx]
        encoded_q = self.bert_tokenizer.encode_plus(sample_token_ids_q, max_length=self.q_max_seq_len,
                                                    pad_to_max_length=True, truncation=True)

        encoded_p_pos = self.bert_tokenizer.encode_plus(sample_token_ids_p_pos, max_length=self.p_max_seq_len,
                                                        pad_to_max_length=True, truncation=True)
        encoded_p_neg = self.bert_tokenizer.encode_plus(sample_token_ids_p_neg, max_length=self.p_max_seq_len,
                                                        pad_to_max_length=True, truncation=True)
        
        sample = {
            "q_token_ids" : encoded_q['input_ids'],
            "q_token_type_ids" : encoded_q['token_type_ids'],
            "q_attention_mask": encoded_q['attention_mask'],
            "p_pos_token_ids" : encoded_p_pos['input_ids'],
            "p_pos_token_type_ids" : encoded_p_pos['token_type_ids'],
            "p_pos_attention_mask": encoded_p_pos['attention_mask'],
            "p_neg_token_ids" : encoded_p_neg['input_ids'],
            "p_neg_token_type_ids" : encoded_p_neg['token_type_ids'],
            "p_neg_attention_mask": encoded_p_neg['attention_mask'],
        }
        return sample
