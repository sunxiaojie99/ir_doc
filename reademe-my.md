# 召回模型 run
1.运行召回模型，得到召回模型
- 输入训练数据格式：`query null para_text_pos null para_text_neg null`
- 输出文件目录：`output/dual_params.bin`
> [huggingface ernie链接](https://huggingface.co/nghuyong/ernie-gram-zh) , 随便找了一个链接，也可以看看有没有其他合适的
```
# 1.下载数据集
sh script/download.sh

# 2.下载预训练模型，到文件夹 torch_pretrained_models 下
eg, torch_pretrained_models/chinese-bert-wwm

# 3.更改相应的参数，torch_src/dual_hparams.py, 可能需要更改的：
- pretrained_model_path
- vocab_path
- epoch
- batch_size


# 4. gpu单卡运行，如果不是卡1，还需要改一下do_dual_train.py中的 os.environ['CUDA_VISIBLE_DEVICES'] = '1'设置

# 5.开始运行
CUDA_VISIBLE_DEVICES=1 nohup python3 -u do_dual_train.py > process_do_dual_train.log 2>&1 &

# 注：cpu 上debug
python do_dual_train.py --debug
```


2.针对dev的query在所有文档库里检索topk的文章，文件输出至 `output/res.topk`，输出格式为qid, pid, rank, score
```
# 0. 进行passage过滤，合并passage，会生成一个all_doc文件
python torch_src/handle_passage_pre.py

# 1. 更改相应的参数，torch_src/inference_de_hparams.py, 可能需要更改的：
- pretrained_model_path
- vocab_path
- batch_size

# 2. 更改do_dual_infer.py 中的 enter(topk=50, bs=1000)，其实topk代表召回阶段为每个query召回k篇文档，后续送入精排阶段。bs代表在用 faiss 检索的时候，一次组织多少个query进行检索，太大可能会段错误

# 3. gpu单卡运行，如果不是卡1，还需要改一下do_dual_infer.py中的 os.environ['CUDA_VISIBLE_DEVICES'] = '1'设置

# 4.开始运行
CUDA_VISIBLE_DEVICES=1 nohup python3 -u do_dual_infer.py > process_do_dual_infer.log 2>&1 &

# 注：cpu debug
python do_dual_infer.py --debug
```

3.利用脚本转化第二步的输出 `output/res.topk`, 输出`output/dual_res.json`, json格式，qid作为key，value是一个list，里面装着召回的50个pid
```
QUERY2ID="dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"

PARA2ID="dureader-retrieval-baseline-dataset/passage-collection/new_passage2id.map.json"

# 如果上一步topk=50，这里的后缀就是50
MODEL_OUTPUT="output/res.top50"

# 50 代表上一步我们划分的是 top几，这里用于数量检查
python metric/convert_recall_res_to_json.py $QUERY2ID $PARA2ID $MODEL_OUTPUT 50
```

4.【可忽略！测试eval脚本用的】转换官方召回50到提交格式
```
ID_MAP="dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv"
python metric/convert_offical_recall_res_to_json.py $ID_MAP 50
```

# 精排模型 run
1.训练精排模型
- 输入:dureader-retrieval-baseline-dataset/train/cross.train.tsv
- 输入数据格式：`query null para_text label`
- 输出:
```
# 1. 更改相应的参数，torch_src/cross/cross_hparams.py, 可能需要更改的：
- pretrained_model_path
- vocab_path
- batch_size
- lr

# 2.gpu单卡运行，如果不是卡0，还需要改一下do_cross_train.py中的 os.environ['CUDA_VISIBLE_DEVICES'] = '0'设置

# 3. 开始运行
CUDA_VISIBLE_DEVICES=0 nohup python3 -u do_cross_train.py > process_do_cross_train.log 2>&1 &

# 注：debug模型，不带CUDA_VISIBLE_DEVICES=0即为cpu运行
CUDA_VISIBLE_DEVICES=0 python do_cross_train.py --debug
```

2.针对召回模型得到的topk，进行重排序
- 输入1: 第一个阶段为每个query得到的top-50（更多也可以，会根据精排模型得分，保留前50）检索到的文章
- 输入2: MODEL_PATH是微调后的模型地址
```
# 1. 更改相应的参数，torch_src/cross/inference_cross_hparams.py, 可能需要更改的：
- test_set
- save_path
- pretrained_model_path
- vocab_path
- batch_size

# 2. gpu单卡运行，如果不是卡0，还需要改一下 do_cross_infer.py中的 os.environ['CUDA_VISIBLE_DEVICES'] = '0'设置

# 4.开始运行
CUDA_VISIBLE_DEVICES=0 nohup python3 -u do_cross_infer.py > process_do_cross_infer.log 2>&1 &

# 注：cpu debug
python do_cross_infer.py --debug
```

3.转换输出格式，输出文件`output/cross_res.json`
```
MODEL_OUTPUT="output/cross_infer_top50.score"
ID_MAP="dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv"
python metric/convert_rerank_res_to_json.py $MODEL_OUTPUT $ID_MAP 
```

# eval

1.使用`MRR@10`, `Recall@1` 和 `Recall@50` 用作评估指标，在dev上，REFERENCE_FIEL是官方带答案的文件，PREDICTION_FILE模型的输出，可以是召回模型得到的`output/dual_res.json`，也可以是精排模型得到的`output/cross_res.json`
```
REFERENCE_FIEL="dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/cross_res.json"
python metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE
```

# dataset
8096668 个文章,800多w个文章

召回的训练集，889580，80多w条
精排的训练集，1111975，100多w条

# result

待评测的query数量： 2000
在前10找到答案的query数量： 1673
在前50找到答案的query数量： 1835
用前10找到答案的做分母的mrr： 0.7295092932570516
{"MRR@10": 0.6102345238095237, "QueriesRanked": 2000, "recall@1": 0.4955, "recall@50": 0.9175}

# ref
- https://github.com/jingtaozhan/RepBERT-Index/blob/master/dataset.py
- [pytorch 矩阵乘法](https://zhuanlan.zhihu.com/p/100069938)
- [paddle/pytorch对照表](https://www.i4k.xyz/article/qq_32097577/112383360)
- [faiss 教程](https://zhuanlan.zhihu.com/p/320653340)
- [比赛链接](https://aistudio.baidu.com/aistudio/competition/detail/157/0/introduction)
- [损失函数技术总结及Pytorch使用示例](https://zhuanlan.zhihu.com/p/383997503)