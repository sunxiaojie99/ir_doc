#!/bin/bash

upk1r=0
newk1=2
upb=0
newb=1

# upk1r=0.55
# newk1=0.6
# upb=0.4
# newb=0.5

ver=$upk1r
echo "k1-b-mrr10-recall50" > search_jieba_stop.csv
while [ 1 = "$(echo "$ver <= $newk1" | bc -l)" ]
do
    b=$upb
    while [ 1 = "$(echo "$b <= $newb" | bc -l)" ]
    do
        
        python -m pyserini.search.lucene \
        --index bm25_jieba_stop/indexes/lucene-index-ir-passage \
        --topics bm25_jieba_stop/dev_queries_zh.tsv \
        --output bm25_jieba_stop/runs/run.dev.bm25tuned_top50.txt \
        --language zh \
        --hits 50 \
        --bm25 --k1 $ver --b $b
        python bm25_jieba_stop/bm25.py
        MODEL_OUTPUT="bm25_jieba_stop/dev_bm25_id_map_top50.tsv"
        out_file="output/cross_res_jieba.json"
        python metric/convert_rerank_res_to_json_with_qp.py $MODEL_OUTPUT $out_file
        REFERENCE_FIEL="dureader-retrieval-baseline-dataset/dev/dev.json"
        PREDICTION_FILE="output/cross_res_jieba.json"
        x=`python metric/evaluation_for_bm25.py $REFERENCE_FIEL $PREDICTION_FILE`
        echo "$ver-$b-$x" >> search_jieba_stop.csv
        b=$(echo "$b + 0.1" | bc -l)
    done
    ver=$(echo "$ver + 0.05" | bc -l)
done