from os.path import dirname, abspath
from os import makedirs
from tensorflow.version import VERSION

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


def save_logs(args, log_data):
    """ Log configuration and information of model"""
    logs = [120 * '#',
            'TensorFlow Version: {}'.format(VERSION),
            'Mode: {}'.format(args.mode),
            120 * '#',
            'Working Directory: {}'.format(root_dir),
            'Model Options',
            120 * '#',
            'You have chosen {} data'.format(args.modality),
            'You have chosen {} classes'.format(args.classes),
            120 * '#',
            'Augmentation mode: {}'.format(args.augm_set),
            'Augmentation probability: {}%'.format(args.augm_prob * 100),
            'Batch size: {}'.format(args.batch_size),
            120 * '#',
            'Training options',
            120 * '#',
            'Training epochs: {}'.format(args.epochs),
            'Train set contains {} images'.format(log_data['train_size']),
            'Steps per epoch = {} steps'.format(log_data['steps_per_epoch']),
            'Total training steps = {} steps'.format(log_data['max_training_steps']),
            'Learning Rate: {}'.format(args.lr),
            'Number of learning rate decays in a training session: {}'.format(args.decays_per_train),
            'Decay rate of learning rate: {}'.format(args.decay_rate),
            'Early stopping after {} epochs with no metric increase.'.format(args.early_stop),
            'Dropout rate: {}'.format(args.dropout),
            120 * '#',
            'Evaluation options',
            120 * '#',
            'Evaluation set contains {} examples'.format(log_data['eval_size']),
            'Total evaluation steps = {}.'.format(log_data['eval_steps']),
            120 * '#',
            'Estimator configuration',
            120 * '#',
            'Random seed: {}'.format(str(args.seed)),
            120 * '#']
    makedirs(log_data['model_path'], exist_ok=True)
    file = open(log_data['model_path'] + '/train_info.txt', "w+")
    for i in logs:
        file.write(i)
        file.write('\n')
    file.close()
