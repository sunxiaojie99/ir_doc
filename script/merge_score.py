import os
import numpy as np
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))

def ensumble(file_list, merge_file):
    """多模型输出文件进行集成（取平均）

    :param file_list: 结果输出列表.
    :param merge_file: 合并输出.
    :return:
    """
    num = len(file_list)
    result_merge = None
    for file in tqdm(file_list):
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            result = []
            for l in lines:
                result.append(float(l.strip()))
        
        result = np.array(result)
        if result_merge is None:
            result_merge = result
        else:
            result_merge += result
    
    result_merge = result_merge / num
    f = open(merge_file, 'w', encoding='utf8')
    for i in result_merge:
        f.write(str(i)+'\n')

file_list = [
    os.path.join(here, '../output/dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv.score.0.0'),
    os.path.join(here, '../output_baseline/dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv.score.0.0')
]
merge_file = os.path.join(here, '../output/dev.retrieval.top50.res.tsv.score.0.0_merge')
ensumble(file_list, merge_file)
