import os
from config import dataset_root
from shutil import rmtree
from glob import glob


def rmv2(dataset):
    ground_2_paths = glob('{}/**'.format(dataset_root), recursive=True)
    ground_2_paths = [path for path in ground_2_paths if os.path.isdir(path) and '/{}'.format(dataset) in path and 'Ground_2' in path]
    print(len(ground_2_paths))
    for i in ground_2_paths:
        # rmtree(i)
        print(i)


if __name__ == '__main__':
    rmv2('MR')
