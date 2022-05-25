import csv
import sys
import json
from collections import defaultdict

id_score_f = sys.argv[1]

outputf = 'output/cross_res.json'

scores = []
q_ids = []
p_ids = []
q_dic = defaultdict(list)

with open(id_score_f, 'r') as f:
    for line in f:
        v = line.strip().split('\t')
        q=v[0]
        p=v[1]
        s=v[2]
        q_dic[v[0]].append((s, p))

output = []
for q in q_dic:
    rank = 0
    cands = q_dic[q]
    cands.sort(reverse=True)
    for cand in cands:
        rank += 1
        output.append([q, cand[1], rank])
        # if rank > 49: # 保留score前50的passage
        #     break

with open(outputf, 'w') as f:
    res = dict()  # 写出一个字典
    for line in output:
        qid, pid, rank = line
        if qid not in res:
            res[qid] = [0] * 200
        res[qid][int(rank) - 1] = pid # 每个qid的50个pid list
    json.dump(res, f, ensure_ascii=False, indent='\t')
