## install
[install-guide](https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation)
```
pip install pyserini
pip install faiss-cpu
```

## step
```
1. 生成pyserini需要的passage 格式, 取消注释script/handle_data.py中的对应部分 5. 生成bm25 docs库，运行
python script/handle_data.py

2. 生成passage检索：
cd bm25
nohup python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input collection_jsonl \
  --language zh \
  --index indexes/lucene-index-ir-passage \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw > bm25.log 2>&1 &

3. 生成需要的query文件格式，取消注释script/handle_data.py中的对应部分 6. 生成bm25 query文件，运行
python script/handle_data.py


# 4. 对dev进行检索

nohup python -m pyserini.search.lucene \
  --index bm25/indexes/lucene-index-ir-passage \
  --topics bm25/dev_queries_zh.tsv \
  --output bm25/runs/run.dev.bm25tuned_top50.txt \
  --language zh \
  --hits 50 \
  --bm25 --k1 0.65 --b 0.7 > bm25_dev.log 2>&1 &


# 4. 对dual_train进行检索
nohup python -m pyserini.search.lucene \
  --index bm25/indexes/lucene-index-ir-passage \
  --topics bm25/dual_queries_zh.tsv \
  --output bm25/runs/run.dual.bm25tuned_top100.txt \
  --language zh \
  --hits 100 \
  --bm25 --k1 0.65 --b 0.7 > bm25_dual.log 2>&1 &

# 4. 对test1进行检索
nohup python -m pyserini.search.lucene \
  --index bm25/indexes/lucene-index-ir-passage \
  --topics bm25/test1_queries_zh.tsv \
  --output bm25/runs/run.test1.bm25tuned_top50.txt \
  --language zh \
  --hits 50 \
  --bm25 --k1 0.65 --b 0.7 > bm25_dual.log 2>&1 &
```

## res
```
--k1 1.2 --b 0.75
top50
待评测的query数量： 2000
在前10找到答案的query数量： 815
在前50找到答案的query数量： 1322
用前10找到答案的做分母的mrr： 0.4624233128834353
{"MRR@10": 0.1884374999999999, "QueriesRanked": 2000, "recall@1": 0.1045, "recall@50": 0.661, "recall@all": 0.661}

--k1 0.82 --b 0.68
top50
待评测的query数量： 2000
在前10找到答案的query数量： 870
在前50找到答案的query数量： 1339
用前10找到答案的做分母的mrr： 0.47510764459040283
{"MRR@10": 0.20667182539682522, "QueriesRanked": 2000, "recall@1": 0.1175, "recall@50": 0.6695, "recall@all": 0.6695}

--k1 0.6 --b 0.68
top50
评测的query数量： 2000
在前10找到答案的query数量： 886
在前50找到答案的query数量： 1343
用前10找到答案的做分母的mrr： 0.49306451323945666
{"MRR@10": 0.2184275793650793, "QueriesRanked": 2000, "recall@1": 0.13, "recall@50": 0.6715, "recall@all": 0.6715}

--k1 0.6 --b 0.5
top50
待评测的query数量： 2000
在前10找到答案的query数量： 897
在前50找到答案的query数量： 1333
用前10找到答案的做分母的mrr： 0.4889362248057894
{"MRR@10": 0.21928789682539657, "QueriesRanked": 2000, "recall@1": 0.129, "recall@50": 0.6665, "recall@all": 0.6665}

```

## ref
- https://github.com/castorini/pyserini/
- https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md
- [bm25公式](https://zhuanlan.zhihu.com/p/79202151),k1越大对应使用更原始的词频信息，b是另一个可调参数（0<b<1），他是用决定使用文档长度来表示信息量的范围：当b为1，是完全使用文档长度来权衡词的权重，当b为0表示不使用文档长度。