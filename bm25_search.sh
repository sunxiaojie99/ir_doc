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
echo "k1-b-mrr10-recall50" > search.csv
while [ 1 = "$(echo "$ver <= $newk1" | bc -l)" ]
do
    b=$upb
    while [ 1 = "$(echo "$b <= $newb" | bc -l)" ]
    do
        
        b=$(echo "$b + 0.1" | bc -l)
        python -m pyserini.search.lucene \
        --index bm25/indexes/lucene-index-ir-passage \
        --topics bm25/dev_queries_zh.tsv \
        --output bm25/runs/run.dev.bm25tuned_top50.txt \
        --language zh \
        --hits 50 \
        --bm25 --k1 $ver --b $b
        python bm25/bm25.py
        MODEL_OUTPUT="bm25/dev_bm25_id_map_top50.tsv"
        python metric/convert_rerank_res_to_json_with_qp.py $MODEL_OUTPUT
        REFERENCE_FIEL="dureader-retrieval-baseline-dataset/dev/dev.json"
        PREDICTION_FILE="output/cross_res.json"
        x=`python metric/evaluation_for_bm25.py $REFERENCE_FIEL $PREDICTION_FILE`
        echo "$ver-$b-$x" >> search.csv
    done
    ver=$(echo "$ver + 0.05" | bc -l)
done