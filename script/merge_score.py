import os
import numpy as np
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))

def ensumble(file_list, merge_file):
    """多模型输出文件进行集成（取平均）

    :param file_list: 结果输出列表.
    :param merge_file: 合并输出.
    :return:
    """
    num = len(file_list)
    result_merge = None
    for file in tqdm(file_list):
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            result = []
            for l in lines:
                result.append(float(l.strip()))
        
        result = np.array(result)
        if result_merge is None:
            result_merge = result
        else:
            result_merge += result
    
    result_merge = result_merge / num
    f = open(merge_file, 'w', encoding='utf8')
    for i in result_merge:
        f.write(str(i)+'\n')

def merge_bm25_and_model(bm25_id_map_file, model_id_map_file, model_score_file, out_put_file):
    """merge bm25分数和模型分数"""
    qid_pid2score = {}

    bm25_qid_2_min_score = {}
    with open(bm25_id_map_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line = l.rstrip('\n').split('\t')
            qid = line[0]
            pid = line[1]
            bm_score = float(line[2])
            assert qid + '\t' + pid not in qid_pid2score
            qid_pid2score[qid + '\t' + pid] = {'bm_score': bm_score}

            if qid not in bm25_qid_2_min_score:
                bm25_qid_2_min_score[qid] = bm_score
            else:
                bm25_qid_2_min_score[qid] = min(bm25_qid_2_min_score[qid], bm_score)
    
    scores = []
    model_qid_2_min_score = {}
    with open(model_score_file, 'r') as f:
        for line in f:
            s = float(line.strip())
            scores.append(s)

    idx = 0
    with open(model_id_map_file, 'r') as f:
        for line in f:
            v = line.strip().split('\t')
            qid = v[0]
            pid = v[1]
            model_score = scores[idx]
            idx += 1

            if qid not in model_qid_2_min_score:  # 每个query的最小score
                model_qid_2_min_score[qid] = model_score
            else:
                model_qid_2_min_score[qid] = min(model_qid_2_min_score[qid], model_score)

            if qid + '\t' + pid not in qid_pid2score:  # 不在就创立一个
                qid_pid2score[qid + '\t' + pid] = {'model_score': model_score}
            else:  # 在的话就加上model的分数
                qid_pid2score[qid + '\t' + pid]['model_score'] = model_score
    m_cof = 1
    bm_cof = 0.0006
    print('m_cof:{},bm_cof:{}'.format(m_cof, bm_cof))
    f_out = open(out_put_file, 'w', encoding='utf8')
    
    qid_pid2score = sorted(qid_pid2score.items(), key=lambda x:x[0])
    for qp, scores in qid_pid2score:
        q, p = qp.split('\t')
        if 'bm_score' not in scores:
            if q in bm25_qid_2_min_score:
                bm_score = bm25_qid_2_min_score[q]
            bm_score = 0
        else:
            bm_score = scores['bm_score']
        
        if 'model_score' not in scores:
            model_score = model_qid_2_min_score[q]
            # model_score = 0
        else:
            model_score = scores['model_score']

        score = m_cof * model_score + bm_cof * bm_score
        f_out.write('{}\t{}\n'.format(qp, score))
    
    f_out.close()







# 1. merge 模型分数，取平均
# file_list = [
#     os.path.join(here, '../output/dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv.score.0.0'),
#     os.path.join(here, '../output_baseline/dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv.score.0.0')
# ]
# merge_file = os.path.join(here, '../output/dev.retrieval.top50.res.tsv.score.0.0_merge')
# ensumble(file_list, merge_file)

# 2. merge bm25分数和模型分数
# bm25_id_map_file = os.path.join(here, '../bm25/dev_bm25_id_map_top100.tsv')  # qid\tpid\tscore
# model_id_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv')  # qid\tpid
# model_score_file = os.path.join(here, '../output/dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv.score.0.0') # score
# out_put_file = os.path.join(here, '../output/bm25_model_merge_id_map.tsv')

bm25_id_map_file = os.path.join(here, '../bm25/test1_bm25_id_map_top50.tsv')  # qid\tpid\tscore
model_id_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dual_res_top50/test1.dual.top50.tsv')  # qid\tpid
model_score_file = os.path.join(here, '../output_offical_test1/dureader-retrieval-baseline-dataset/dual_res_top50/test1.retrieval_text.top50.res.tsv.score.0.0') # score
out_put_file = os.path.join(here, '../output_offical_test1/bm25_model_merge_id_map.tsv')

merge_bm25_and_model(bm25_id_map_file, model_id_map_file, model_score_file, out_put_file)