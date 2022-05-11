import csv
import sys
import json
from collections import defaultdict

id_f = sys.argv[1]  # dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv
topk = int(sys.argv[2])
qprank = defaultdict(list)

outputf = 'output/offical_dual_res.json'

with open(id_f, 'r') as f:
    for line in f:
        v = line.strip().split('\t')
        qprank[v[0]].append(v[1])

# check for length
for key in list(qprank.keys()):
    assert len(qprank[key]) == topk, str(len(qprank[key])) + '#' + str(topk)

with open(outputf, 'w', encoding='utf-8') as fp:
    json.dump(qprank, fp, ensure_ascii=False, indent='\t')
