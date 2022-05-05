import os
from torch_src.cross.cross_hparams import hparams
from torch_src.cross.cross_train import train

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
here = os.path.dirname(os.path.abspath(__file__))


def main():
	"""CUDA_VISIBLE_DEVICES=1 nohup python3 do_train.py > process_5fold_bce_context_3_ernie_gram.log 2>&1 &"""
	hparams.debug = True
	print('Arguments:')
	for arg in vars(hparams):
		print('    {}: {}'.format(arg, getattr(hparams, arg)))
	train(hparams)


if __name__ == '__main__':
    main()