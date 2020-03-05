from os.path import dirname, abspath

root_dir = dirname(abspath(__file__))
dataset_root = root_dir + '/Datasets'

setup_paths = {'irdcab_train': dataset_root + '/Train_Sets_2',
               'irdcab_eval': dataset_root + '/Eval_Sets_2',
               'chaos_train': dataset_root + '/Train_Sets_1',
               'chaos_eval': dataset_root + '/Eval_Sets_1',
               'chaos_test': dataset_root + '/Test_Sets_1',
               'chaos_tain_zip': dataset_root + '/CHAOS_Train_Sets.zip',
               'chaos_test_zip': dataset_root + '/CHAOS_Test_Sets.zip',
               'irdcab_root': dataset_root + '/3Dircadb1',
               'irdcab_zip': dataset_root + '/3Dircadb1.zip'}

paths = {'train': dataset_root + '/Train_Sets',
         'eval': dataset_root + '/Eval_Sets',
         'infer': dataset_root + '/Test_Sets',
         'save': root_dir + '/saves',
         'save_pred': root_dir + '/predictions'}


