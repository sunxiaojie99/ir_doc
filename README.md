# DuReader<sub>retrieval</sub> Dataset


提供了2种数据集:

- **Orginal dataset** in json format, containing queries, their corresponding positive passages, and passage collection.
- **Pre-processed dataset** in tsv format, used for the baseline system, containing extra hard negative samples that selected using our retrieval model. 

# DuReader<sub>retrieval</sub> Baseline System
In this repository, we release a baseline system for DuReader<sub>retrieval</sub> dataset. The baseline system is based on [RocketQA](https://arxiv.org/pdf/2010.08191.pdf) and [ERNIE 1.0](https://arxiv.org/abs/1904.09223), and is implemented with [PaddlePaddle](https://www.paddlepaddle.org.cn/) framework. To run the baseline system, please follow the instructions below.

## Environment Requirements
The baseline system has been tested on

 - CentOS 6.3
 - PaddlePaddle 2.2 
 - Python 3.7.0
 - Faiss 1.7.1
 - Cuda 10.1
 - CuDnn 7.6
 
To install PaddlePaddle, please see the [PaddlePaddle Homepage](http://paddlepaddle.org/) for more information.


## Download
Before run the baseline system, please download the pre-processed dataset and the pretrained and fine-tuned model parameters (ERNIE 1.0 base):

```
sh script/download.sh
```
The dataset will be saved into `dureader-retrieval-baseline-dataset/`, the pretrained and fine-tuned model parameters will be saved into `pretrained-models/` and `finetuned-models/`, respectively.  The descriptions of the data structure can be found in `dureader-retrieval-baseline-dataset/readme.md`. 

**Note**: in addition to the positive samples from the origianl dataset, we also provide hard negative samples that produced by our dense retrieval model in the training data. Users may use their own strategies for hard negative sample selection. 


## Run Baseline
The baseline system contatins two steps: 

- Step 1: a dual-encoder for passage retrieval; 
- Step 2: a cross-encoder for passage re-ranking.

For more details about the model structure, please refer to [RocketQA](https://arxiv.org/pdf/2010.08191.pdf) (Qu et al., 2021). 

### Step 1 - Dual-encoder (for retrieval)
#### Training 训练
微调检索模型
To fine-tune a retrieval model, please run the following command:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
TRAIN_SET="dureader-retrieval-baseline-dataset/train/dual.train.demo.tsv"
MODEL_PATH="pretrained-models/ernie_base_1.0_twin_CN/params"
sh script/run_dual_encoder_train.sh $TRAIN_SET $MODEL_PATH 10 4 
```
我使用的版本
```
export CUDA_VISIBLE_DEVICES=0,1
TRAIN_SET="dureader-retrieval-baseline-dataset/train/dual.train.tsv"
MODEL_PATH="pretrained-models/ernie_base_1.0_twin_CN/params"
sh script/run_dual_encoder_train.sh $TRAIN_SET $MODEL_PATH 10 2
```
This will train on the demo data for 10 epochs with 4 gpu cars. The training log will be saved into `log/`. At the end of training, model parameters will be saved into `output/`. To start the training on the full dataset, please set `TRAIN_SET=dureader-retrieval-baseline-dataset/train/dual.train.tsv`.

**Note**: We strongly recommend to use more gpus for training. The performance increases with the effective batch size, which is related to the number of gpus. 如果使用单gpu训练， please turn off the option `use_cross_batch` in `script/run_dual_encoder_train.sh`. 

> [faiss 教程](https://zhuanlan.zhihu.com/p/320653340)

#### Prediction 预测
用微调过的参数在验证集上验证
To predict with fine-tuned parameters, (e.g. on the devlopment set), please run the following command:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
TEST_SET="dureader-retrieval-baseline-dataset/dev/dev.q.format"
MODEL_PATH="finetuned-models/dual_params/"
DATA_PATH="dureader-retrieval-baseline-dataset/passage-collection"
TOP_K=50
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K
```
The fine-tuned parameters under `MODEL_PATH ` will be loaded for prediction. The prediction on the development set will take a few hours on 4*V100 cards. 预测结果 will be saved into `output/`. 

We provide a script to convert the model output to the standard json format for evaluation. To preform the conversion:
提供了一个脚本把模型输出转化为提交的json格式

```
QUERY2ID="dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"
PARA2ID="dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"
MODEL_OUTPUT="output/dev.res.top50"
python metric/convert_recall_res_to_json.py $QUERY2ID $PARA2ID $MODEL_OUTPUT
```
Where `MODEL_OUTPUT` represents the output file from the dual-encoder, `QUERY2ID `, `PARA2ID ` are the mapping files which maps the query and passages to their original IDs. The output json file will be saved in `output/dual_res.json`.


**Note**: 我们把文章集合为了数据并行划分了4份，We divide the passage collection into 4 parts for data parallel. For users who use different number of GPUs, please update the data files (i.e. `dureader-retrieval-baseline-dataset/passage-collection/part-0x`) and the corresponding configurations.

### Step 2 - Cross-encoder (for re-ranking)
#### Training
To fine-tune a re-ranking model, please run the following command:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
TRAIN_SET=dureader-retrieval-baseline-dataset/train/cross.train.demo.tsv
MODEL_PATH=pretrained-models/ernie_base_1.0_CN/params
sh script/run_cross_encoder_train.sh $TRAIN_SET $MODEL_PATH 3 4
```

```
export CUDA_VISIBLE_DEVICES=0
TRAIN_SET=dureader-retrieval-baseline-dataset/train/cross.train.tsv
MODEL_PATH=output_baseline/step_104248
sh script/run_cross_encoder_train.sh $TRAIN_SET $MODEL_PATH 3 1
nohup sh script/run_cross_encoder_train.sh $TRAIN_SET $MODEL_PATH 3 1 > process_cross_train_paddle.log 2>&1 &

MODEL_PATH=pretrained-models/ernie_base_1.0_CN/params
nohup sh script/run_cross_encoder_train.sh $TRAIN_SET $MODEL_PATH 1 1 > process_corss_train_paddle.log 2>&1 &
```
This will train on the demo data for 3 epochs with 4 gpu cars (a few minutes on 4*V100). The training log will be saved into `log/`. The model parameters will be saved into `output/`. To start the training on the full dataset, please set `TRAIN_SET=dureader-reltrieval-baseline-dataset/train/cross.train.tsv`


#### Prediction
To predict with fine-tuned parameters, (e.g. on the devlopment set), please run the following command:

```
export CUDA_VISIBLE_DEVICES=0
TEST_SET=dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv
MODEL_PATH=output/step_34750
nohup sh script/run_cross_encoder_inference.sh $TEST_SET $MODEL_PATH > process_corss_infer_paddle_dev.log 2>&1 &
```
TRAIN_SET 是第一个阶段为每个query得到的top-50（更多也可以，会根据精排模型得分，保留前50）检索到的文章，MODEL_PATH是微调后的模型地址

Where `TEST_SET` is the top-50 retrieved passages for each query from step 1, `MODEL_PATH` is the path to fined-tuned model parameters. The predicted answers will be saved into `output/`. 

We provide a script to convert the model output to the standard json format for evaluation. To preform the conversion:

```
MODEL_OUTPUT="output/dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv.score.0.0"
MODEL_OUTPUT="output/dev.retrieval.top50.res.tsv.score.0.0_merge"
ID_MAP="dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv"
python metric/convert_rerank_res_to_json.py $MODEL_OUTPUT $ID_MAP 
```
MODEL_OUTPUT是cross-encoder的输出文件，ID_MAP是query和passage的id。

Where `MODEL_OUTPUT` represents the output file from the cross-encoder, `ID_MAP` is the mapping file which maps the query and passages to their original IDs. The output json file will be saved in `output/cross_res.json`.

## Evaluation
提供了评估脚本进行评估.

`MRR@10`, `Recall@1` and `Recall@50` are used as evaluation metrics. Here we provide a script `evaluation.py` for evaluation.

To evluate, run

```
REFERENCE_FIEL="dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/cross_res.json"
python metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE
```
REFERENCE_FIEL原始的验证集合带答案，PREDICTION_FILE模型的输出

Where `REFERENCE_FIEL` is the origianl dataset file, and `PREDICTION_FILE ` is the model prediction that should be a valid JSON file of `(qid, [list-of -top50-pid])` pairs, for example:

```
{
   "edb58f525bd14724d6f490722fa8a657":[
      "5bc347ff17d488f1704e2893c9e8ecfa",
      "6e67389d07da8ce02ed97167d23baf9d",
      "06031941b9613d2fde5cb309bbefaf88",
      ...
      "58c00697311c9ad6eb384b6fca7bd12d",
      "e06eb2750f5ed163eb85b6ef02b7c608",
      "aa425d7dcb409592527a22ce5eccd4d5"
   ],
   "a451acd1e9836b04b16664e9f0c290e5":[
      "71c8004cc562f2a75181b2d3d370a45a",
      "ef3d34ea63b3de9db612bd7b7ffd143a",
      "2c839510f35d5495251c6b3c057bd300",
      ...
      "6b9777840bb537a433add0b9f553fd42",
      "4203161e38b9b5e67ff16fc777f614be",
      "ae651b80efbb10f786380a6afdc1dcbe",
   ]
}
```

After runing the evaluation script, you will get the evaluation results with the following format:

```
{"MRR@10": 0.7284081349206347, "QueriesRanked": 2000, "recall@1": 0.641, "recall@50": 0.9175}
```

## Baseline Performance
The performance of our baseline model on the development set are shown below:

| Model |  MRR@10 | recall@1 | recall@50 |
| --- | --- | --- | --- |
| dual-encoder (retrieval) | 60.45 | 49.75 | 91.75|
| cross-encoder (re-ranking) | 72.84 | 64.10 | 91.75|

# Copyright and License
This repository is provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/RocketQA/blob/main/LICENSE).
