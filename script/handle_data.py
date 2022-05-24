import os
import json
from tqdm import tqdm
import csv

here = os.path.dirname(os.path.abspath(__file__))

def draw(dic, name): #输入样本数量统计字典
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import namedtuple

    fig, ax = plt.subplots()

    n_groups = len(dic) #列数
    index=[]
    data=[]
    for i in sorted (dic) : 
        index.append(i)
        data.append(dic[i])

    bar_width = 0.2 #每条柱状的宽度
    rects1 = ax.bar(index, data, bar_width,label='length') #绘制柱状图

    ax.legend() #绘制图例（即右上角的方框）

    fig.tight_layout()
    plt.show()
    fig.savefig(name, bbox_inches='tight')


def generate_text_from_id(res_text_file, res_id_file, query_id2text_map_file, file_name_list, passage_index2id):
    """
    res_text_file: 输出文件，每行4个元素，用\t链接，'\t'.join([query, '', para, '0']) eg, query null para_text label
    res_id_file: 读入文件，qid \t pid

    """
    
    query_id2text_map = {}
    with open(query_id2text_map_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            l = eval(line.strip())
            query_id2text_map[l['question_id']] = l['question']
    
    passage_id2text_map = {}

    with open(passage_index2id, 'r', encoding='utf-8') as f_in:
        passage_lineidx2id = json.load(f_in)
    print('原始文章数：', len(passage_lineidx2id))  # 原始文章数： 8096668

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

    print('pass_id_map, ', len(passage_id2text_map))

    f_text = open(res_text_file, 'w', encoding='utf-8')
    with open(res_id_file, 'r') as f:
        for line in f:
            v = line.strip().split('\t')
            qid = v[0]
            pid = v[1]
            q_text = query_id2text_map[qid]
            p_text = passage_id2text_map[pid]
            f_text.write('\t'.join([q_text, '', p_text, '0']) + '\n')

    f_text.close()


def stat_cross(cross_train_file):
    pos_cnt = 0
    neg_cnt = 0
    all_cnt = 0
    print(cross_train_file)
    para_len2count = {}
    query_len2count = {}
    max_para = 0
    min_para = 2000
    with open(cross_train_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line = l.rstrip('\n').split('\t')
            assert len(line) == 4, line
            # query null para_text label
            query = line[0]
            passage = line[2]
            label = line[3]
            if int(label) == 0:
                neg_cnt += 1
            else:
                pos_cnt += 1
            all_cnt += 1
            if len(query) not in query_len2count:
                query_len2count[len(query)] = 1
            else:
                query_len2count[len(query)] += 1
            if len(passage) not in para_len2count:
                para_len2count[len(passage)] = 1
            else:
                para_len2count[len(passage)] += 1
            max_para = max(max_para, len(passage))
            min_para = min(min_para, len(passage))
    print('pos_cnt:{}, neg_cnt:{}, all_cnt:{}'.format(pos_cnt, neg_cnt, all_cnt))

    file = open('passage_stat.csv','w',encoding='utf-8',newline='')
    csv_writer= csv.DictWriter(file,fieldnames=['para_len','cnt'])
    csv_writer.writeheader()
    key=list(para_len2count.keys())
    value =list(para_len2count.values())
    for i in range(len(key)):
        dic = {       #字典类型
            'para_len':key[i],
            'cnt':value[i]
        }
        csv_writer.writerow(dic)   #数据写入csv文件
    file.close()
    


# res_text_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv')
# res_text_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dual_res_top50/test1.retrieval_text.top50.res.tsv')
# res_id_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dual_res_top50/test1.dual.top50.tsv')
# query_id2text_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dureader-retrieval-test1/test1.json')
# file_name_list = [
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-00"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-01"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-02"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-03")
# ]
# passage_index2id = os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json")
# generate_text_from_id(res_text_file, res_id_file, query_id2text_map_file, file_name_list, passage_index2id)


# 统计正负样本比例
cross_train_file=os.path.join(here, '../dureader-retrieval-baseline-dataset/train/cross.train.tsv')
stat_cross(cross_train_file)