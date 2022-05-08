import os
import json
from regex import P
from tqdm import tqdm
from tokenization import _is_whitespace, _is_control
import json

here = os.path.dirname(os.path.abspath(__file__))


def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def handle(data_dir, map_file, out_file, new_map_file):
    """
    passage文件中，有的passage比如376574行，全部由控制字符组成，需要提前过滤掉.
    eg: "\x07\x07\x07\x05\x07"
    """

    file_name_list = os.listdir(data_dir)

    with open(map_file, 'r', encoding='utf-8') as f_in:
        passage_lineidx2id = json.load(f_in)
    print('原始文章数：', len(passage_lineidx2id))  # 原始文章数： 8096668

    f_w = open(out_file, 'w', encoding='utf-8')

    new_passage_lineidx2id = {}
    count = 0  # 原始行👌
    new_count = 0
    print(file_name_list)
    for file_name in file_name_list:
        if "part" in file_name:
            file = os.path.join(data_dir, file_name)
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for l in tqdm(lines):
                    passage_id = passage_lineidx2id[str(count)]  # 获取旧的pid
                    count += 1
                    line = l.strip().split('\t')
                    passage = line[2]
                    clean_text = _clean_text(passage)
                    if clean_text == "":
                        continue
                    
                    f_w.write(l)
                    new_passage_lineidx2id[str(new_count)] = passage_id  # 存入pid
                    new_count += 1

    f_w.close()
    print('新的文章数：', new_count)  # 新的文章数： 8096660

    with open(new_map_file, 'w', encoding='utf-8') as f_out:
        json.dump(new_passage_lineidx2id, f_out, ensure_ascii=False, indent=2)
    


def merge(data_dir, out_file):
    """"合并多个passage文件到一个"""
    file_name_list = os.listdir(data_dir)
    count = 0
    f_w = open(out_file, 'w', encoding='utf-8')
    for file_name in tqdm(file_name_list):
        if "part" in file_name:
            file = os.path.join(data_dir, file_name)
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for l in lines:
                    f_w.write(l)
                    count += 1
    f_w.close()
    print(count)


handle(os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/"),
       os.path.join(
           here, "../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"),
       os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/all_doc"),
       os.path.join(
           here, "../dureader-retrieval-baseline-dataset/passage-collection/new_passage2id.map.json"))
# merge(os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/"),
#       os.path.join(here, "../dureader-retrieval-baseline-dataset/passage-collection/all_doc"))
