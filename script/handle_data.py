import os
import json
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))

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



# res_text_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv')
res_text_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dual_res_top50/test1.retrieval_text.top50.res.tsv')
res_id_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dual_res_top50/test1.dual.top50.tsv')
query_id2text_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dureader-retrieval-test1/test1.json')
file_name_list = [
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-00"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-01"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-02"),
    os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-03")
]
passage_index2id = os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json")
generate_text_from_id(res_text_file, res_id_file, query_id2text_map_file, file_name_list, passage_index2id)