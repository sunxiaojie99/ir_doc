# 召回模型 run
1.运行召回模型，输出文件：`output/dual_params.bin`
- 输入数据格式：`query null para_text_pos null para_text_neg null`
```
python do_dual_train.py --debug
```


2.针对dev的query检索topk的文章，文件输出至 `output/res.topk`，格式为qid, pid, rank, score
```
# enter(topk=10, bs=1)，于这里修改topk
python do_dual_infer.py
```

3.利用脚本转化第二步的输出 `output/res.topk`, 输出`output/dual_res.json`, json格式，qid作为key，value是一个list，里面装着召回的50个pid
```
# 10 代表上一步我们划分的是 top几，这里用于数量检查
QUERY2ID="dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"
PARA2ID="dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"
MODEL_OUTPUT="output/res.top10"
python metric/convert_recall_res_to_json.py $QUERY2ID $PARA2ID $MODEL_OUTPUT 10
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