import os
from torch_src.cross.cross_hparams import hparams
from torch_src.cross.cross_train import train

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def main():
    # hparams.debug = True
    print('Arguments:')
    for arg in vars(hparams):
        print('    {}: {}'.format(arg, getattr(hparams, arg)))
    train(hparams)


if __name__ == '__main__':
    main()
