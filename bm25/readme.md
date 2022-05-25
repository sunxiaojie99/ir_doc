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
  --index indexes/lucene-index-ir-passage \
  --topics dev_queries_zh.tsv \
  --output runs/run.dev.bm25tuned_top50.txt \
  --hits 50 \
  --bm25 --k1 0.82 --b 0.68 > bm25_dev.log 2>&1 &

# 4. 对dual_train进行检索
nohup python -m pyserini.search.lucene \
  --index indexes/lucene-index-ir-passage \
  --topics dual_queries_zh.tsv \
  --output runs/run.dual.bm25tuned.txt \
  --hits 200 \
  --bm25 --k1 0.82 --b 0.68 > bm25_dual.log 2>&1 &
```

## ref
https://github.com/castorini/pyserini/
https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md