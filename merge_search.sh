#!/bin/bash

begin_m_cof=1
up_m_cof=1
begin_bm_cof=0
up_bm_cof=0.0011

m_cof=$begin_m_cof
echo "m_cof-bm_cof-mrr10-recall50" > merge_search.csv
while [ 1 = "$(echo "$m_cof <= $up_m_cof" | bc -l)" ]
do
    bm_cof=$begin_bm_cof
    while [ 1 = "$(echo "$bm_cof <= $up_bm_cof" | bc -l)" ]
    do
        python script/merge_score.py $m_cof $bm_cof
        MODEL_OUTPUT="output_offical_dev/bm25_model_merge_id_map.tsv"
        out_file="output_offical_dev/cross_res.json"
        python metric/convert_rerank_res_to_json_with_qp.py $MODEL_OUTPUT $out_file
        REFERENCE_FIEL="dureader-retrieval-baseline-dataset/dev/dev.json"
        PREDICTION_FILE="output_offical_dev/cross_res.json"
        x=`python metric/evaluation_for_bm25.py $REFERENCE_FIEL $PREDICTION_FILE`
        echo "$m_cof-$bm_cof-$x" >> merge_search.csv
        bm_cof=$(echo "$bm_cof + 0.00005" | bc -l)
    done
    m_cof=$(echo "$m_cof + 1" | bc -l)
done