import os
import random
from glob import glob
from config import dataset_root, paths


def split_eval():
    for modality in ['/CT', '/MR']:
        print(modality[1:].center(10, ' ').center(100, '*'))
        if os.path.exists(paths['eval']):
            set_eval_path = paths['eval']
        else:
            set_eval_path = dataset_root

        if os.path.exists(paths['train']):
            set_train_path = paths['train']
        else:
            set_train_path = dataset_root
        for eval_sets in ['/Eval_Sets_1', '/Eval_Sets_2']:
            if os.path.exists(set_eval_path + eval_sets + modality):
                print('Removing patients from {} folder'.format(eval_sets[1:]).center(100, '-'))
                eval_set = glob(set_eval_path + eval_sets + modality + '/**')
                for eval_patient in eval_set:
                    print('Moving: {}'.format(eval_patient).center(60, ' ').center(100, '|'))
                    print('to: {}'.format(eval_patient.replace('Eval_Sets', 'Train_Sets')).center(60, ' ').center(100, '|'))
                    os.makedirs(os.path.dirname(eval_patient.replace('Eval_Sets', 'Train_Sets')), exist_ok=True)
                    os.replace(eval_patient, eval_patient.replace('Eval_Sets', 'Train_Sets'))
        for train_sets in ['/Train_Sets_1', '/Train_Sets_2']:
            print('Moving patients from {} folder'.format(train_sets.split('/')[-1]).center(100, '-'))
            patient_list = glob(set_train_path + train_sets + modality + '/**')
            if patient_list:
                eval_patients = random.sample(patient_list, k=2)
                for patient in eval_patients:
                    print('Moving: {}'.format(patient).center(60, ' ').center(100, '|'))
                    print('to: {}'.format(patient.replace('Train_Sets', 'Eval_Sets')).center(60, ' ').center(100, '|'))
                    os.makedirs(os.path.dirname(patient.replace('Train_Sets', 'Eval_Sets')), exist_ok=True)
                    os.replace(patient, patient.replace('Train_Sets', 'Eval_Sets'))
