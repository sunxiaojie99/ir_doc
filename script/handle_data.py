import os
import json
from tqdm import tqdm
import csv

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




def generate_text_from_recall_res(res_text_file, map_file, res_id_file, query_id2text_map_file, file_name_list, passage_index2id, option='dev'):
    """
    res_text_file: 输出文件，每行4个元素，用\t链接，'\t'.join([query, '', para, '0']) eg, query null para_text label
    res_id_file: 读入文件， json格式，qid作为key，value是一个list，里面装着召回的topk个pid

    """
    
    query_id2text_map = {}
    if option == 'test1':
        with open(query_id2text_map_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                l = eval(line.strip())
                query_id2text_map[l['question_id']] = l['question']
    elif option == 'dev':
        with open(query_id2text_map_file, 'r', encoding='utf-8') as f_in:
            query_text2id_map = json.load(f_in)
        query_id2text_map = {v:k for k,v in query_text2id_map.items()}
    else:
        print('目前只支持dev、test1格式')
        exit()
    
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

    with open(res_id_file, 'r', encoding='utf-8') as f_in:
        qid2pid_list = json.load(f_in)

    f_text = open(res_text_file, 'w', encoding='utf-8')
    f_map = open(map_file, 'w', encoding='utf-8')
    for qid, pids in qid2pid_list.items():
        for pid in pids:
            q_text = query_id2text_map[qid]
            p_text = passage_id2text_map[pid]
            f_map.write('\t'.join([qid, pid]) + '\n')
            f_text.write('\t'.join([q_text, '', p_text, '0']) + '\n')
    
    f_map.close()
    f_text.close()


def stat_cross(cross_train_file):
    pos_cnt = 0
    neg_cnt = 0
    all_cnt = 0
    print(cross_train_file)
    para_len2count = {}
    query_len2count = {}
    pos_para_len2count = {}
    pos_query_len2count = {}
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
            
            if int(label) == 1:
                if len(query) not in pos_query_len2count:
                    pos_query_len2count[len(query)] = 1
                else:
                    pos_query_len2count[len(query)] += 1
                if len(passage) not in pos_para_len2count:
                    pos_para_len2count[len(passage)] = 1
                else:
                    pos_para_len2count[len(passage)] += 1

            max_para = max(max_para, len(passage))
            min_para = min(min_para, len(passage))
    print('pos_cnt:{}, neg_cnt:{}, all_cnt:{}'.format(pos_cnt, neg_cnt, all_cnt))

    file = open('query_stat.csv','w',encoding='utf-8',newline='')
    csv_writer= csv.DictWriter(file,fieldnames=['query_len','cnt'])
    csv_writer.writeheader()
    key=list(query_len2count.keys())
    value =list(query_len2count.values())
    for i in range(len(key)):
        dic = {       #字典类型
            'query_len':key[i],
            'cnt':value[i]
        }
        csv_writer.writerow(dic)   #数据写入csv文件
    file.close()
    

def stat_dual(dual_train_file):
    q2p = {}  # 每个q有多少个<p+,q->
    qp2neg = {}  # 每个<q,p+>有多少个q-
    para_len2count = {}
    query_len2count = {}
    with open(dual_train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for l in tqdm(lines):
            line = l.rstrip('\n').split('\t')
            query = line[0]
            p_pos = line[2]
            p_neg = line[4]
            if query+'@'+p_pos not in qp2neg:
                qp2neg[query+'@'+p_pos] = 1
            else:
                qp2neg[query+'@'+p_pos] += 1
            if query not in q2p:
                q2p[query] = 1
            else:
                q2p[query] += 1
            
            if len(query) not in query_len2count:
                query_len2count[len(query)] = 1
            else:
                query_len2count[len(query)] += 1
            if len(p_pos) not in para_len2count:
                para_len2count[len(p_pos)] = 1
            else:
                para_len2count[len(p_pos)] += 1
            if len(p_neg) not in para_len2count:
                para_len2count[len(p_neg)] = 1
            else:
                para_len2count[len(p_neg)] += 1
    
    q2p_num = {}
    qp2neg_num = {}
    for k, num in q2p.items():
        if num not in q2p_num:
            q2p_num[num] = 1
        else:
            q2p_num[num] += 1
    
    for num in qp2neg.values():
        if num not in qp2neg_num:
            qp2neg_num[num] = 1
        else:
            qp2neg_num[num] += 1
    
    
    sort_q2p = sorted(q2p_num.items(), key=lambda x: x[0])
    print('每个q有多少个<p+,q->:', sort_q2p)
    print('每个<q,p+>有多少个q-:', qp2neg_num)

    file = open('dual_query_stat.csv','w',encoding='utf-8',newline='')
    csv_writer= csv.DictWriter(file,fieldnames=['query_len','cnt'])
    csv_writer.writeheader()
    key=list(query_len2count.keys())
    value =list(query_len2count.values())
    for i in range(len(key)):
        dic = {       #字典类型
            'query_len':key[i],
            'cnt':value[i]
        }
        csv_writer.writerow(dic)   #数据写入csv文件
    file.close()

    file = open('dual_passage_stat.csv','w',encoding='utf-8',newline='')
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


def generate_passage_jsonl(file_name_list, passage_index2id, output_file):
    """生成passage的jsonl格式，用于bm25训练
    {"id": "doc1", "contents": "contents of doc one."}
    """

    with open(passage_index2id, 'r', encoding='utf-8') as f_in:
        passage_lineidx2id = json.load(f_in)
    print('原始文章数：', len(passage_lineidx2id))  # 原始文章数： 8096668

    count = 0

    f_out = open(output_file, 'w', encoding='utf8')
    for file_name in file_name_list:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for l in tqdm(lines):
                passage_id = passage_lineidx2id[str(count)]  # 获取旧的pid
                count += 1
                line = l.strip().split('\t')
                passage = line[2]
                json.dump({"id": passage_id, "contents": passage}, f_out, ensure_ascii=False)
                f_out.write('\n')
    f_out.close()

def generate_query_jsonl(query_file, out_put_file, qid_map_file, count2qtext_map_file, option='dev'):
    query_text2id_map = {}
    if option == 'test1':
        with open(query_id2text_map_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                l = eval(line.strip())
                query_text2id_map[l['question']] = l['question_id']
    elif option == 'dev':
        with open(query_id2text_map_file, 'r', encoding='utf-8') as f_in:
            query_text2id_map = json.load(f_in)
    elif option == 'dual_train':
        with open(query_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for l in tqdm(lines):
                line = l.rstrip('\n').split('\t')
                query = line[0]
                p_pos = line[2]
                p_neg = line[4]
                if query not in query_text2id_map:
                    query_text2id_map[query] = 0
    else:
        print('不支持的格式')
        exit()
    

    file = open(out_put_file,'w',encoding='utf-8',newline='')
    csv_writer= csv.DictWriter(file,fieldnames=['qid','text'], delimiter='\t')
    # csv_writer.writeheader()
    texts=list(query_text2id_map.keys())
    ids =list(query_text2id_map.values())
    count2qid = {}
    count2qtext = {}
    count = 0
    for i in tqdm(range(len(texts))):
        dic = {       #字典类型
            'qid': count,
            'text':texts[i]
        }
        csv_writer.writerow(dic)   #数据写入csv文件
        count2qid[count] = ids[i]
        count2qtext[count] = texts[i]
        count += 1
    file.close()
    with open(qid_map_file, 'w', encoding='utf-8') as f_out:
        json.dump(count2qid, f_out, ensure_ascii=False, indent=2)
    
    with open(count2qtext_map_file, 'w', encoding='utf-8') as f_out:
        json.dump(count2qtext, f_out, ensure_ascii=False, indent=2)


# 1. 根据官方召回结果，生成传入cross阶段的数据

# res_text_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv')
out_res_text_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dual_res_top50/test1.retrieval_text.top50.res.tsv')
# res_id_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dual_res_top50/test1.dual.top50.tsv')
# query_id2text_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dureader-retrieval-test1/test1.json')
# file_name_list = [
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-00"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-01"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-02"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-03")
# ]
# passage_index2id = os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json")
# generate_text_from_id(out_res_text_file, res_id_file, query_id2text_map_file, file_name_list, passage_index2id)

# 2.根据recall模型的结果 dual_res.json 生成cross的输入结果
# res_text_file = os.path.join(here, '../output/dual_res_for_cross.tsv')
# map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top100.res.id_map.tsv')
# res_id_file = os.path.join(here, '../output/dual_res.json')
# query_id2text_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dev/q2qid.dev.json')  # dev设置
# # query_id2text_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dureader-retrieval-test1/test1.json')  # test1设置
# file_name_list = [
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-00"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-01"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-02"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-03")
# ]
# passage_index2id = os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json")
# generate_text_from_recall_res(res_text_file, map_file, res_id_file, query_id2text_map_file, file_name_list, passage_index2id, option='dev')

# 3. 统计cross正负样本比例
# cross_train_file=os.path.join(here, '../dureader-retrieval-baseline-dataset/train/cross.train.tsv')
# stat_cross(cross_train_file)

# 4.统计dual
# dual_train_file=os.path.join(here, '../dureader-retrieval-baseline-dataset/train/dual.train.tsv')
# stat_dual(dual_train_file)

# 5. 生成bm25 docs库
# file_name_list = [
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-00"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-01"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-02"),
#     os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/part-03")
# ]
# passage_index2id = os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json")
# output_file = os.path.join(here, "../bm25/collection_jsonl/documents.jsonl")
# generate_passage_jsonl(file_name_list, passage_index2id, output_file)

# 6. 生成bm25 query文件
# query_id2text_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dev/q2qid.dev.json')  # dev设置
query_id2text_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/dureader-retrieval-test1/test1.json')  # test1设置
# query_id2text_map_file = os.path.join(here, '../dureader-retrieval-baseline-dataset/train/dual.train.tsv')  # dual_train设置
out_put_file=os.path.join(here, "../bm25/test1_queries_zh.tsv")
qid_map_file = os.path.join(here, "../bm25/test1_queryid_map.json")
count2qtext_map_file = os.path.join(here, "../bm25/test1_querytext_map.json")
generate_query_jsonl(query_id2text_map_file, out_put_file, qid_map_file, count2qtext_map_file, option='test1')