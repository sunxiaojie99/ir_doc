import os
import json
from tqdm import tqdm
import random
import numpy as np
import jieba
from pyserini.analysis import Analyzer, get_lucene_analyzer

here = os.path.dirname(os.path.abspath(__file__))

def pre_handle_query(query_file, jieba_query, stop_word_file=None):
    if stop_word_file is not None:
        stopwords = [line.strip() for line in open('',encoding='UTF-8').readlines()]

    f_out = open(jieba_query, 'w', encoding='utf8')

    with open(query_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for l in tqdm(lines):
            q = l.split('\t')
            idx = q[0]
            query = q[1].strip()
            seg_list = jieba.cut_for_search(query, HMM=False)  # 搜索引擎模式
            seg = ' '.join(seg_list)
            f_out.write('{}\t{}\n'.format(idx, seg))
    f_out.close()


def pre_handle_passage(passage_file, jieba_file, stop_word_file=None):
    """
    {"id": "7cc92c636434d8cd8250bc26386368a1", "contents": "如图所示: ?"}

    """
    if stop_word_file is not None:
        stopwords = [line.strip() for line in open('',encoding='UTF-8').readlines()]
    
    f_out = open(jieba_file, 'w', encoding='utf8')

    with open(passage_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            l = eval(line.strip())
            passage_id = l['id']
            passage = l['contents']
            seg_list = jieba.cut_for_search(passage, HMM=False)  # 搜索引擎模式
            seg = ' '.join(seg_list)
            json.dump({"id": passage_id, "contents": seg}, f_out, ensure_ascii=False)
            f_out.write('\n')
    f_out.close()

    
# 1. 处理query
stop_word_file = os.path.join(here, 'baidu_stopwords.txt')
dev_query = os.path.join(here, '../bm25/dual_queries_zh.tsv')
jieba_query = os.path.join(here, 'dual_queries_zh.tsv')
pre_handle_query(dev_query, jieba_query, stop_word_file=None)

# 2. 处理passage
# passage_file = os.path.join(here, '../bm25/collection_jsonl/documents.jsonl')
# jieba_file = os.path.join(here, 'collection_jsonl/documents.jsonl')
# pre_handle_passage(passage_file, jieba_file, None)