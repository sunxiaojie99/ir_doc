import os

from sympy import im
from torch_src.inference_de_hparams import gen_params
from torch_src.inference_dual_encoder import infer
from torch_src.inference_index_search import enter

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
here = os.path.dirname(os.path.abspath(__file__))


def main():
	"""CUDA_VISIBLE_DEVICES=1 nohup python3 do_train.py > process_5fold_bce_context_3_ernie_gram.log 2>&1 &"""
	hparams = gen_params('query')
	hparams.debug = True

	print('Arguments:')
	for arg in vars(hparams):
		print('    {}: {}'.format(arg, getattr(hparams, arg)))

	infer(hparams)

	hparams = gen_params('passage')
	hparams.debug = True
	infer(hparams)

	enter(topk=10, bs=1)

if __name__ == '__main__':
    main()