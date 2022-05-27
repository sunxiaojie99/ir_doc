import os
import argparse
import torch
import json

here = os.path.dirname(os.path.abspath(__file__))

train_set = os.path.join(here, '../../dureader-retrieval-baseline-dataset/train/cross.train.tsv')
checkpoints_dir=os.path.join(here, '../../output_torch_ernie1.0')
checkpoint_file=os.path.join(checkpoints_dir, 'cross_checkpoint.json')
pretrained_model_path = os.path.join(here, '../../torch_pretrained_models/ernie_1.0_torch') # https://huggingface.co/nghuyong/ernie-gram-zh
vocab_path = os.path.join(here, '../../torch_pretrained_models/ernie_1.0_torch/vocab.txt')
log_dir = os.path.join(here, '../../log_dir')

device = "cuda" if torch.cuda.is_available() else "cpu"
hidden_size=json.load(open(os.path.join(pretrained_model_path, 'config.json'), 'r', encoding='utf-8'))['hidden_size']
lr=1e-5
epochs = 3
batch_size=64
max_seq_len=384
weight_decay=0.0
warmup_proportion=0.1
random_seed=1
num_labels=2  # 2分类

parser = argparse.ArgumentParser()

parser.add_argument('--train_set', type=str, default=train_set)
parser.add_argument("--checkpoints_dir", type=str, default=checkpoints_dir)
parser.add_argument("--checkpoint_file", type=str, default=checkpoint_file)
parser.add_argument("--pretrained_model_path", type=str, default=pretrained_model_path)
parser.add_argument("--vocab_path", type=str, default=vocab_path)
parser.add_argument('--log_dir', type=str, default=log_dir)
parser.add_argument("--device", type=str, default=device)

parser.add_argument("--lr", type=float, default=lr)

parser.add_argument("--epochs", type=int, default=epochs)
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