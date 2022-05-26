import csv
import sys
import json
from collections import defaultdict

score_f = sys.argv[1]
id_f = sys.argv[2]

outputf = 'output/cross_res_dev.json'

scores = []
q_ids = []
p_ids = []
q_dic = defaultdict(list)

with open(score_f, 'r') as f:
    for line in f:
        scores.append(float(line.strip()))

with open(id_f, 'r') as f:
    for line in f:
        v = line.strip().split('\t')
        q_ids.append((v[0]))
        p_ids.append((v[1]))

num = len(scores)

for q, p, s in zip(q_ids[:num], p_ids[:num], scores):
    q_dic[q].append((s, p))

output = []
for q in q_dic:
    rank = 0
    cands = q_dic[q]
    cands.sort(reverse=True)
    for cand in cands:
        rank += 1
        output.append([q, cand[1], rank])
        if rank > 49: # 保留score前50的passage
            break

with open(outputf, 'w') as f:
    res = dict()  # 写出一个字典
    for line in output:
        qid, pid, rank = line
        if qid not in res:
            res[qid] = [0] * 50
        res[qid][int(rank) - 1] = pid # 每个qid的50个pid list
    json.dump(res, f, ensure_ascii=False, indent='\t')
