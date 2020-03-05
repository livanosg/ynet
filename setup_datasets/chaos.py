import os
import zipfile
from config import dataset_root, setup_paths


def unzip_chaos():
    zip_path = [setup_paths['chaos_tain_zip'], setup_paths['chaos_test_zip']]
    for zip in zip_path:
        print('Extracting {} ...'.format(zip))
        with zipfile.ZipFile(zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_root)
        print('Done!')
    os.replace(setup_paths['chaos_train'].replace('Train_Sets_1', 'Train_Sets'), setup_paths['chaos_train'])
