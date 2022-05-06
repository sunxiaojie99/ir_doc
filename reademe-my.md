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
CUDA_VISIBLE_DEVICES=1 nohup python3 do_dual_train.py > process_do_dual_train.log 2>&1 &

# 注：cpu 上debug
python do_dual_train.py --debug
```


2.针对dev的query在所有文档库里检索topk的文章，文件输出至 `output/res.topk`，输出格式为qid, pid, rank, score
```
# 0. 合并passage，将 `script/handle.py` 文件移动到 dureader-retrieval-baseline-dataset/passage-collection/目录下，会生成一个all_doc文件
```
cd dureader-retrieval-baseline-dataset/passage-collection/
python handle.py
```

# 1. 更改相应的参数，torch_src/inference_de_hparams.py, 可能需要更改的：
- pretrained_model_path
- vocab_path
- batch_size

# 2. 更改do_dual_infer.py 中的 enter(topk=50, bs=1000)，其实topk代表召回阶段为每个query召回k篇文档，后续送入精排阶段。bs代表在用 faiss 检索的时候，一次组织多少个query进行检索，太大可能会段错误

# 3. gpu单卡运行，如果不是卡1，还需要改一下do_dual_infer.py中的 os.environ['CUDA_VISIBLE_DEVICES'] = '1'设置

# 4.开始运行
CUDA_VISIBLE_DEVICES=1 nohup python3 do_dual_infer.py > process_do_dual_infer.log 2>&1 &

# 注：cpu debug
python do_dual_infer.py --debug
```

3.利用脚本转化第二步的输出 `output/res.topk`, 输出`output/dual_res.json`, json格式，qid作为key，value是一个list，里面装着召回的50个pid
```
QUERY2ID="dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"

PARA2ID="dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"

# 如果上一步topk=50，这里的后缀就是50
MODEL_OUTPUT="output/res.top50"

# 50 代表上一步我们划分的是 top几，这里用于数量检查
python metric/convert_recall_res_to_json.py $QUERY2ID $PARA2ID $MODEL_OUTPUT 50
```

# 精排模型 run
1.训练精排模型
- 输入:dureader-retrieval-baseline-dataset/train/cross.train.tsv
- 输入数据格式：`query null para_text label`
- 输出:
```
python do_cross_train.py --debug
```

2.针对召回模型得到的topk，进行重排序
- 输入1: 第一个阶段为每个query得到的top-50（更多也可以，会根据精排模型得分，保留前50）检索到的文章
- 输入2: MODEL_PATH是微调后的模型地址
```
```

# dataset
8096668 个文章,800多w个文章

# ref
- https://github.com/jingtaozhan/RepBERT-Index/blob/master/dataset.py
- [pytorch 矩阵乘法](https://zhuanlan.zhihu.com/p/100069938)
- [paddle/pytorch对照表](https://www.i4k.xyz/article/qq_32097577/112383360)
- [faiss 教程](https://zhuanlan.zhihu.com/p/320653340)
- [比赛链接](https://aistudio.baidu.com/aistudio/competition/detail/157/0/introduction)