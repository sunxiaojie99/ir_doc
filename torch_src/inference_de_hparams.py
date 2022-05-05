import os
import argparse
import torch
import json

here = os.path.dirname(os.path.abspath(__file__))

def gen_params(test_type):
	# test_type = 'query'  # 'passage'

	if test_type == 'query':
		test_set = os.path.join(here, '../dureader-retrieval-baseline-dataset/dev/dev.q.format')
		out_file_name=os.path.join(here, '../output/query.emb')
	elif test_type == 'passage':
		test_set = os.path.join(here, '../dureader-retrieval-baseline-dataset/passage-collection/all_doc')
		out_file_name=os.path.join(here, '../output/para.index')

	test_save=os.path.join(here, '../output/test_out.tsv')

	checkpoints_dir=os.path.join(here, '../output')

	model_path = os.path.join(checkpoints_dir, 'dual_params.bin')

	pretrained_model_path = os.path.join(here, '../torch_pretrained_models/chinese-bert-wwm') # https://huggingface.co/nghuyong/ernie-gram-zh
	vocab_path = os.path.join(here, '../torch_pretrained_models/chinese-bert-wwm/vocab.txt')
	hidden_size=json.load(open(os.path.join(pretrained_model_path, 'config.json'), 'r', encoding='utf-8'))['hidden_size']

	device = "cuda" if torch.cuda.is_available() else "cpu"

	batch_size=8
	q_max_seq_len=32
	p_max_seq_len=384

	parser = argparse.ArgumentParser()

	parser.add_argument('--test_type', type=str, default=test_type)
	parser.add_argument("--test_set", type=str, default=test_set)
	parser.add_argument("--out_file_name", type=str, default=out_file_name)
	parser.add_argument("--test_save", type=str, default=test_save)
	parser.add_argument("--checkpoints_dir", type=str, default=checkpoints_dir)
	parser.add_argument("--model_path", type=str, default=model_path)
	parser.add_argument('--pretrained_model_path', type=str, default=pretrained_model_path)
	parser.add_argument('--vocab_path', type=str, default=vocab_path)
	parser.add_argument("--device", type=str, default=device)

	parser.add_argument("--batch_size", type=int, default=batch_size)
	parser.add_argument("--hidden_size", type=int, default=hidden_size)
	parser.add_argument("--q_max_seq_len", type=int, default=q_max_seq_len)
	parser.add_argument("--p_max_seq_len", type=int, default=p_max_seq_len)

	parser.add_argument("--debug",
						action='store_true', help='default false')  # 默认为false

	hparams = parser.parse_args()

	return hparams

