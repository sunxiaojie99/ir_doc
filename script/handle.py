import os
import json
from regex import P
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))


def handle(data_dir, map_file):
	"""读取多个文件"""
	file_name_list = os.listdir(data_dir)
	passage_list = []
	with open(map_file, 'r', encoding='utf-8') as f_in:
		passage_lineidx2id = json.load(f_in)
	print(len(passage_lineidx2id))  # 8096668 个文章
	count = 0
	for file_name in file_name_list:
		if "part" in file_name:
			file = os.path.join(data_dir, file_name)
			with open(file, "r", encoding="utf-8") as f:
				lines = f.readlines()
				for l in tqdm(lines):
					text = l.strip().split("\t")[2]
					passage_list.append(text)
					print(passage_lineidx2id[str(count)])
					count += 1
	print(len(passage_list))  # 4*2024167=8096668

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


# handle(os.path.join(here, "../passage-collection"), os.path.join(here, "../passage-collection/passage2id.map.json"))
merge(os.path.join(here, "../passage-collection"), os.path.join(here, "../passage-collection/all_doc"))
