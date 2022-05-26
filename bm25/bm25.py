import os
import json
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))

def generate_dual_train_data(dual_bm25_file, dual_bm25_id2text_file ,dual_train_file, file_name_list, passage_index2id, num=10):
    """
    基于bm25的结果，采样生成dual训练的数据集
    dual_bm25_file: bm25file
    dual_bm25_id2text_file: bm25中的count对应的原始文本
    dual_train_file: 每行6个元素，用\t链接， eg, query\ttitle_pos\tpara_pos\ttitle_neg\tpara_neg\tlabel
    """
    # with open(dual_bm25_id2text_file, 'r', encoding='utf8') as f:
    #     bm25_id2text = json.load(f)
    
    # query2pos_ps = {}  # key:query , value:[p+1, p+2,...]
    # query2nes_ps = {}  # 每个query的负样本，防止采重？或者同时出现在2边也值得再学习一遍？
    # with open(dual_train_file, 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for l in tqdm(lines):
    #         line = l.rstrip('\n').split('\t')
    #         query = line[0]
    #         p_pos = line[2]
    #         p_neg = line[4]
    #         if query not in query2pos_ps:
    #             query2pos_ps[query] = []
    #             query2nes_ps[query] = []
    #         if p_pos not in query2pos_ps[query]:
    #             query2pos_ps[query].append(p_pos)
    #         if p_neg not in query2nes_ps[query]:
    #             query2nes_ps[query].append(p_neg)
    
    with open(dual_bm25_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in lines:
            line = l.rstrip('\n').split()
            bm25id = line[0]
            para_id = line[2]
            rank = line[3]
            score = line[4]


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


option='dev'

file_name_list = [
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-00"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-01"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-02"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-03")
]
passage_index2id = os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json")

bm25_file = os.path.join(here, 'runs/run.'+option+'.bm25tuned.txt')
bm25_id2text_file = os.path.join(here, option+'_querytext_map.json')
bm25_id2id_file = os.path.join(here, option+'_queryid_map.json')

# 1. 利用bm25 生成dual的训练集
dual_train_file=os.path.join(here, '../dureader-retrieval-baseline-dataset/train/dual.train.tsv')
# generate_dual_train_data(bm25_file, bm25_id2text_file, dual_train_file, file_name_list, passage_index2id, num=10)


# 2. 转换bm25格式, qid\tpid\tscore
out_put_file = os.path.join(here, option+'_bm25_id_map_top50.tsv')
bm25_file = os.path.join(here, 'runs/run.'+option+'.bm25tuned_top50.txt')
generate_bm25_file(bm25_file, bm25_id2id_file, out_put_file)