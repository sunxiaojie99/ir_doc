## train/
### dual.train.tsv 双塔召回训练数据
- content: training samples for dual-encoder
- format: `query null para_text_pos null para_text_neg null` (`\t` seperated, `null` represents invalid column. `para_text_pos` and `para_text_neg` represent passage text of positive sample and negative sample, respectively)

最后都是0
```
微信分享链接打开app		一般用自带浏览器可以调用起app没问题。微信里面能调出app的,是和腾讯有合作的应用,其他会被过滤掉。有一个公司的产品,叫 魔窗,免费可以接入的		随便找个能够分享外链的网站，不要网盘什么的，最好找免费的主机或者免费空间，能拿到绝对路径就ok，出来像这样：http://23,44,23,21/upload/my.apk；http://www.liantu.com/举个例子， 望采纳。	0
```


### cross.train.tsv 用于交叉模型的训练
- content: training samples for cross-encoder
- format: `query null para_text label` (`\t` seperated, `null` represents invalid column.)

看上去1代表pos，0代表neg（判断2个文本是否相关）
```
微信分享链接打开app		i微信里面能调出app的,是和腾讯有合作的应用,其他会被过滤掉。有一个公司的产品,叫 魔窗,免费可以接入的	1
微信分享链接打开app		百度经验:jingyan.baidu.com主要思路就是用一个可以在电脑上面打开手机软件的模拟器,来打开微信。	0
```

### dual.train.demo.tsv
 - content: demo training samples or dual-encoder
 - format: see `dual.train.tsv`

### cross.train.demo.tsv
 - content: demo training samples or cross-encoder
 - format: see `cross.train.tsv`

**Note**: In the training data of dual and cross encoder, the positive samples are from the original dataset, and the negative samples are selected using our dense retrieval model. User may use their own strategies for hard negative sample selection (See RocketQA for more details, Qu et al., 2021). 



## passage-collection/
### part-0x
- content: passage collection (contains 8M passages, divided into 4 parts for data parallel)
- format: `null null passage_text null` (`\t` seperated, `null` represents invalid column)

### passage2id.map.json
- content: passage collection (contains 8M passages, divided into 4 parts for data parallel)
- format: json, a map of `passage_line_index` and `passage_id`



## dev/
### dev.json 验证集样本
- content: development samples 带groundtrue
- format: json
- {"question_id": "", "question": "", "answer_paragraphs": [{"paragraph_id": "", "paragraph_text": ""}, ...

### dev.q.format 验证集的query列表
- content: list of queries from the development set
- format: `query_text null null null` (`\t` seperated, `null` represents invalid column)

### q2qid.dev.json query_text到文本的映射
- content: list of queries from the development set
- format: json, a map of `query_text` and `query_id`



## auxiliary/
### dev.retrieval.top50.res.tsv
- content: Top-50 retrieved passages from the dual-encoder for development set. Used as the input of the re-ranking model during prediction. 

在验证集上用双塔encoder召回的top-50文章
- format: `query_text null passage_text null` (`\t` seperated, `null` represents invalid column)

### dev.retrieval.top50.res.id_map.tsv

dev.retrieval.top50.res.tsv中query和passage原始的id
- content: the origianl id of queries and passages in `dev.retrieval.top50.res.tsv`
- format: `query_id passage_id` (`\t` seperated)