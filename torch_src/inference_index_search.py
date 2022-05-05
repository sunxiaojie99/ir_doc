import sys
import os
import time
import faiss
import math
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def load_qid(file_name):
    qid_list = []
    with open(file_name) as inp:
        for line in inp:
            line = line.strip()
            qid = line.split('\t')[0]
            qid_list.append(qid)
    return qid_list


def read_embed(file_name, dim=768, bs=3000):
    if file_name.endswith('npy'):
        i = 0
        emb_np = np.load(file_name)
        while(i < len(emb_np)):
            vec_list = emb_np[i:i+bs]
            i += bs
            yield vec_list
    else:
        vec_list = []
        with open(file_name) as inp:
            for line in inp:
                data = line.strip()
                vector = [float(item) for item in data.split(' ')]
                assert len(vector) == dim
                vec_list.append(vector)
                if len(vec_list) == bs:
                    yield vec_list
                    vec_list = []
            if vec_list:
                yield vec_list


def search(index, emb_file, qid_list, outfile, top_k, bs=1):
    """
    index: passage的emb index
    emb_file:query的emb
    qid_list:query的文本list
    outfile:输出
    """
    q_idx = 0
    with open(outfile, 'w') as out:
        for batch_vec in read_embed(emb_file, bs=bs):
            q_emb_matrix = np.array(batch_vec)
            res_dist, res_p_id = index.search(
                q_emb_matrix.astype('float32'), top_k)
            # res_dist: 维度是 nq*top_k,代表距离每个query最近的top_k个数据的距离
            # res_p_id: 维度是 nq*top_k,代表距离每个query最近的k个数据的id
            for i in range(len(q_emb_matrix)):
                qid = qid_list[q_idx]
                for j in range(top_k):
                    pid = res_p_id[i][j]
                    score = res_dist[i][j]
                    out.write('%s\t%s\t%s\t%s\n' % (qid, pid, j+1, score))
                q_idx += 1


def enter(topk, bs=1):
    p_index_file = os.path.join(here, '../output/para.index')
    q_emb_file = os.path.join(here, '../output/query.emb'+'.npy')
    q_text_file = os.path.join(
        here, '../dureader-retrieval-baseline-dataset/dev/dev.q.format')

    outfile = os.path.join(here, '../output/res.top{}'.format(topk))
    qid_list = load_qid(q_text_file)

    engine = faiss.read_index(p_index_file)
    print('begin search!')
    search(engine, q_emb_file, qid_list, outfile, topk, bs=bs)
