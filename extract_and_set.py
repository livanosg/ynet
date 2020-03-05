import os
from config import setup_paths, paths
from setup_datasets.chaos import unzip_chaos
from setup_datasets.ircad import setup_ircadb
from setup_datasets.split_evaluation import split_eval


def setup_datasets():
    setup_ircadb()
    unzip_chaos()
    split_eval()
    os.makedirs(paths['train'], exist_ok=True)
    os.makedirs(paths['eval'], exist_ok=True)
    os.replace(setup_paths['chaos_train'], paths['train'] + '/Train_Sets_1')
    os.replace(setup_paths['chaos_eval'], paths['eval'] + '/Eval_Sets_1')

    os.replace(setup_paths['irdcab_train'],  paths['train'] + '/Train_Sets_2')
    os.replace(setup_paths['irdcab_eval'],  paths['eval'] + '/Eval_Sets_2')


if __name__ == '__main__':
    setup_datasets()
