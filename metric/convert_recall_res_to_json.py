# Convert the retrieval output to standard json format
# loading files: para.map.json -> mapping from row id of para to pid in md5
# loading files: q2qid.dev.json -> mapping from query in Chinese to qid in md5

import hashlib
import json
import sys
from collections import defaultdict


q2id_map = sys.argv[1]  # dev/q2qid.dev.json
p2id_map = sys.argv[2]  # passage2id.map.json: {"0": "67236474b99b2215c296a6942ad6e04c"}
recall_result = sys.argv[3]  # qid, pid, rank, score
topk = int(sys.argv[4])

outputf = 'output/dual_res.json'

# map query to its origianl ID
with open(q2id_map, "r") as fr:
    q2qid = json.load(fr)  # query文本映射原始id

# map para line number to its original ID
with open(p2id_map, "r") as fr:
    pcid2pid = json.load(fr)  # passage的行号 映射实际的pid

qprank = defaultdict(list)  # query的原始id->[pid1, pid2,...,pid50], 召回阶段顺序不重要
with open(recall_result, 'r') as f:
    for line in f.readlines():
        q, pcid, rank, score = line.strip().split('\t')
        qprank[q2qid[q]].append(pcid2pid[pcid])

# check for length
for key in list(qprank.keys()):
    assert len(qprank[key]) == topk, str(len(qprank[key])) + '#' + str(topk)

with open(outputf, 'w', encoding='utf-8') as fp:
    json.dump(qprank, fp, ensure_ascii=False, indent='\t')
