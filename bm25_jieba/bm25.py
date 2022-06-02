import os
import json
from tqdm import tqdm
import random
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

def generate_dual_train_data(out_file, dual_bm25_file, dual_bm25_id2text_file ,dual_train_file, file_name_list, passage_index2id, sample_num=10):
    """
    基于bm25的结果，采样生成dual训练的数据集
    dual_bm25_file: bm25file
    dual_bm25_id2text_file: bm25中的count对应的原始文本
    dual_train_file: 每行6个元素，用\t链接， eg, query\ttitle_pos\tpara_pos\ttitle_neg\tpara_neg\tlabel
    """
    with open(dual_bm25_id2text_file, 'r', encoding='utf8') as f:
        bm25_id2text = json.load(f)
    
    with open(passage_index2id, 'r', encoding='utf-8') as f_in:
        passage_lineidx2id = json.load(f_in)
    print('原始文章数：', len(passage_lineidx2id))  # 原始文章数： 8096668

    passage_id2text_map = {}
    count = 0
    for file_name in file_name_list:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for l in tqdm(lines):
                passage_id = passage_lineidx2id[str(count)]  # 获取旧的pid
                count += 1
                line = l.strip().split('\t')
                passage = line[2]
                assert passage_id not in passage_id2text_map
                passage_id2text_map[passage_id] = passage
    
    print('读取原始dual集')
    query2pos_ps = {}  # key:query , value:[p+1, p+2,...]
    query2nes_ps = {}  # 每个query的负样本，防止采重？或者同时出现在2边也值得再学习一遍？
    with open(dual_train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line = l.rstrip('\n').split('\t')
            query = line[0]
            p_pos = line[2]
            p_neg = line[4]
            if query not in query2pos_ps:
                query2pos_ps[query] = []
                query2nes_ps[query] = []
            if p_pos not in query2pos_ps[query]:
                query2pos_ps[query].append(p_pos)
            if p_neg not in query2nes_ps[query]:
                query2nes_ps[query].append(p_neg)
    
    print('生成bm25 neg样本')
    query2bm25neg_dict = {}

    with open(dual_bm25_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            line = l.rstrip('\n').split()
            bm25id = line[0]
            query = bm25_id2text[bm25id]
            para_id = line[2]
            para = passage_id2text_map[para_id]
            rank = line[3]
            score = line[4]
            if query not in query2bm25neg_dict:
                query2bm25neg_dict[query] = []
            assert query in query2pos_ps
            assert query in query2nes_ps
            # 不在pos里 ，也不在neg里
            if para not in query2bm25neg_dict[query] and para not in query2pos_ps[query] and para not in query2nes_ps[query]:
                query2bm25neg_dict[query].append(para)
    
    f_out = open(out_file, 'w', encoding='utf8')
    for q, ps in tqdm(query2bm25neg_dict.items()):
        np_ps = np.array(ps)

        neg_num = len(ps)
        need_ps = []
        all_pos = np.array(query2pos_ps[q])  # 所有的正样本

        if len(all_pos) <=2 :
            need_pos = all_pos
        else:
            sample_idx = random.sample(range(len(all_pos)), 2)  # 每个选2个query搭配
            need_pos = all_pos[sample_idx]


        if neg_num <= sample_num:  # 负样本数量小于需要的，全部采样
            need_neg = np_ps
        else:
            sample_idx = random.sample(range(neg_num), sample_num)
            need_neg = np_ps[sample_idx]

        for pos in need_pos:
            for neg in need_neg: # query\ttitle_pos\tpara_pos\ttitle_neg\tpara_neg\tlabel
                f_out.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(q, '',pos,'', neg,'0' ))
    
    f_out.close()



def generate_bm25_file(bm25_file, dual_bm25_id2id_file, output_file):
    """
    转换bm25格式, qid\tpid\tscore
    """
    with open(dual_bm25_id2id_file, 'r', encoding='utf8') as f:
        bm25_id2qid = json.load(f)
    
    f_out = open(out_put_file, 'w', encoding='utf8')

    with open(bm25_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line = l.rstrip('\n').split()
            bm25id = line[0]
            qid = bm25_id2qid[bm25id]
            para_id = line[2]
            rank = line[3]
            score = line[4]
            f_out.write('{}\t{}\t{}\n'.format(qid, para_id, score))
    
    f_out.close()


def generate_bm25_file_dual(bm25_file=os.path.join(here, 'runs/run.dual.bm25tuned_top200.txt'), 
                            dual_bm25_id2text_file=os.path.join(here, '../bm25/dual_querytext_map.json'), 
                            output_file=os.path.join(here,'dual_bm25_text_pid_map_top200.tsv')):
    """
    转换bm25格式, qid\tpid\tscore
    """
    with open(dual_bm25_id2text_file, 'r', encoding='utf8') as f:
        bm25_id2text = json.load(f)
    
    f_out = open(output_file, 'w', encoding='utf8')

    with open(bm25_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line = l.rstrip('\n').split()
            bm25id = line[0]
            query_text = bm25_id2text[bm25id]
            para_id = line[2]
            rank = line[3]
            score = line[4]
            f_out.write('{}\t{}\t{}\n'.format(query_text, para_id, score))
    
    f_out.close()


option='dev'

print('正在处理:', option)

file_name_list = [
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-00"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-01"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-02"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-03")
]
passage_index2id = os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json")

bm25_file = os.path.join(here, 'runs/run.'+option+'.bm25tuned_top100.txt')
bm25_id2text_file = os.path.join(here, '../bm25/'+option+'_querytext_map.json')
bm25_id2id_file = os.path.join(here, '../bm25/'+option+'_queryid_map.json')

# 1. 利用bm25 生成dual的训练集
# dual_train_file=os.path.join(here, '../dureader-retrieval-baseline-dataset/train/dual.train.tsv')
# out_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/train/bm25_dual.train.tsv')
# generate_dual_train_data(out_file, bm25_file, bm25_id2text_file, dual_train_file, file_name_list, passage_index2id, sample_num=4)


# 2. 转换bm25格式, qid\tpid\tscore
out_put_file = os.path.join(here, option+'_bm25_id_map_top100.tsv')
bm25_file = os.path.join(here, 'runs/run.'+option+'.bm25tuned_top100.txt')
generate_bm25_file(bm25_file, bm25_id2id_file, out_put_file)


# 3. 为dual转换格式，因为dual给的没有qid
# generate_bm25_file_dual()