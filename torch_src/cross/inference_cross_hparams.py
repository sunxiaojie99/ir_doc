import os
import argparse
import torch
import json

here = os.path.dirname(os.path.abspath(__file__))

test_set = os.path.join(here, '../../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv')
# test_set = os.path.join(here, '../../dureader-retrieval-baseline-dataset/train/cross.train.tsv')
checkpoints_dir=os.path.join(here, '../../output_torch_ernie1.0_focal_a1.3')
checkpoint_file=os.path.join(checkpoints_dir, 'cross_checkpoint.json')
pretrained_model_path = os.path.join(here, '../../torch_pretrained_models/ernie_1.0_torch') # https://huggingface.co/nghuyong/ernie-gram-zh
vocab_path = os.path.join(here, '../../torch_pretrained_models/ernie_1.0_torch/vocab.txt')
log_dir = os.path.join(here, '../../log_dir')
save_path = os.path.join(checkpoints_dir, 'cross_infer_top50_epoch2.score')
all_score_save_path = os.path.join(checkpoints_dir, 'all_cross_infer_top50.score')

model_path = os.path.join(checkpoints_dir, 'epoch_2_cross_params.bin')

device = "cuda" if torch.cuda.is_available() else "cpu"
hidden_size=json.load(open(os.path.join(pretrained_model_path, 'config.json'), 'r', encoding='utf-8'))['hidden_size']
lr=1e-5
epoch = 2
batch_size=384
max_seq_len=510
weight_decay=0.0
warmup_proportion=0.1
random_seed=1
num_labels=2  # 2分类

parser = argparse.ArgumentParser()

parser.add_argument('--test_set', type=str, default=test_set)
parser.add_argument("--checkpoints_dir", type=str, default=checkpoints_dir)
parser.add_argument("--checkpoint_file", type=str, default=checkpoint_file)
parser.add_argument("--model_path", type=str, default=model_path)
parser.add_argument("--pretrained_model_path", type=str, default=pretrained_model_path)
parser.add_argument("--vocab_path", type=str, default=vocab_path)
parser.add_argument('--log_dir', type=str, default=log_dir)
parser.add_argument("--device", type=str, default=device)
parser.add_argument("--save_path", type=str, default=save_path)
parser.add_argument("--all_score_save_path", type=str, default=all_score_save_path)

parser.add_argument("--lr", type=float, default=lr)

parser.add_argument("--epoch", type=int, default=epoch)
parser.add_argument("--hidden_size", type=int, default=hidden_size)
parser.add_argument("--batch_size", type=int, default=batch_size)
parser.add_argument("--max_seq_len", type=int, default=max_seq_len)
parser.add_argument("--weight_decay", type=int, default=weight_decay)
parser.add_argument("--warmup_proportion", type=int, default=warmup_proportion)
parser.add_argument("--random_seed", type=int, default=random_seed)
parser.add_argument("--num_labels", type=int, default=num_labels)

parser.add_argument("--debug",
                    action='store_true', help='default false')  # 默认为false

hparams = parser.parse_args()