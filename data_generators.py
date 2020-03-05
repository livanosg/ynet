import cv2
import numpy as np
from glob import glob
from pydicom import dcmread
from cv2 import imread
from augmentations import augmentations, augmentations2
from config import paths


# noinspection PyUnboundLocalVariable
def get_paths(dataset, branch, modality):
    if branch == 1:
        folder = 'Ground'
    else:
        folder = 'Ground_2'
    ct_dcm_paths = paths[dataset] + '/**/CT/**/**.dcm'
    ct_grd_paths = paths[dataset] + '/**/CT/**/{}/*.png'.format(folder)
    mr_dcm_in = paths[dataset] + '/**/MR/**/InPhase/*.dcm'
    mr_dcm_out = paths[dataset] + '/**/MR/**/OutPhase/*.dcm'
    mr_dcm_t2 = paths[dataset] + '/**/MR/**/T2SPIR/**/*.dcm'
    mr_grd_t1 = paths[dataset] + '/**/MR/**/T1DUAL/{}/*.png'.format(folder)
    mr_grd_t2 = paths[dataset] + '/**/MR/**/T2SPIR/{}/*.png'.format(folder)

    ct_dicom_list = sorted(glob(ct_dcm_paths, recursive=True))
    ct_ground_list = sorted(glob(ct_grd_paths, recursive=True))

    mr_dcm_in_list = sorted(glob(mr_dcm_in, recursive=True))
    mr_dcm_out_list = sorted(glob(mr_dcm_out, recursive=True))
    mr_dcm_t2_list = sorted(glob(mr_dcm_t2, recursive=True))
    mr_grd_t1_list = sorted(glob(mr_grd_t1, recursive=True))
    mr_grd_t2_list = sorted(glob(mr_grd_t2, recursive=True))
    mr_dicom_list = mr_dcm_in_list + mr_dcm_out_list + mr_dcm_t2_list
    mr_ground_list = mr_grd_t1_list + mr_grd_t1_list + mr_grd_t2_list

    if modality == 'CT':
        dicom_list = ct_dicom_list
        ground_list = ct_ground_list
    elif modality == 'MR':
        dicom_list = mr_dicom_list
        ground_list = mr_ground_list
    else:
        dicom_list = ct_dicom_list + mr_dicom_list
        ground_list = ct_ground_list + mr_ground_list

    if dataset in ('train', 'eval'):
        assert len(dicom_list) == len(ground_list)  # Check lists length
        zipped_list = list(zip(dicom_list, ground_list))
        return zipped_list
    if dataset == 'infer':
        return dicom_list


# noinspection PyUnboundLocalVariable
def test_gen(params):
    data_paths = get_paths(dataset='infer', branch=params['branch'], modality=params['modality'])
    for dcm_path in data_paths:
        dicom = dcmread(dcm_path).pixel_array
        dicom = (dicom - np.mean(dicom)) / np.std(dicom)
        yield dicom, dcm_path


def make_labels_gen(params):
    train_data_paths = get_paths(dataset='train', branch=1, modality=params['modality'])
    eval_data_paths = get_paths(dataset='eval', branch=1, modality=params['modality'])
    data_paths = train_data_paths + eval_data_paths
    if params['shuffle']:
        np.random.shuffle(data_paths)
    for dcm_path, label_path in data_paths:
        print(dcm_path, label_path)
        dicom = dcmread(dcm_path).pixel_array
        dicom = (dicom - np.mean(dicom)) / np.std(dicom)
        yield dicom, label_path


def data_gen(dataset, params, only_paths=False):
    data_paths = get_paths(dataset=dataset, branch=params['branch'], modality=params['modality'])
    if params['shuffle']:
        np.random.shuffle(data_paths)
    if only_paths:
        for dicom_path, label_path in data_paths:
            yield dicom_path, label_path
    else:
        for dicom_path, label_path in data_paths:
            prob = np.random.random()
            dicom, label_1 = dcmread(dicom_path).pixel_array, imread(label_path, 0)
            if params['branch'] == 2:
                label_2 = imread(label_path.replace('Ground_2', 'Ground'), 0)
            if dataset == 'train' and params['augm_set'] is not None and prob < params['augm_prob']:  # Data augmentation
                if params['branch'] == 1:
                    dicom, label_1 = augmentations(dcm_image=dicom, grd_image=label_1, augm_set=params['augm_set'])
                else:
                    dicom, label_1, label_2 = augmentations2(dcm_image=dicom, label_1=label_1, label_2=label_2, augm_set=params['augm_set'])
            if params['modality'] in ('MR', 'ALL'):
                if params['modality'] == 'MR':
                    resize = 320 - dicom.shape[0]
                else:
                    resize = 512 - dicom.shape[0]
                dicom = np.pad(dicom, [int(resize / 2)], mode='constant', constant_values=np.min(dicom))
                label_1 = np.pad(label_1, [int(resize / 2)], mode='constant', constant_values=np.min(label_1))
                if params['branch'] == 2:
                    label_2 = np.pad(label_2, [int(resize / 2)], mode='constant', constant_values=np.min(label_2))
            dicom = (dicom - np.mean(dicom)) / np.std(dicom)
            if params['branch'] == 1:
                if 'CT' in label_path:
                    label_1[label_1 == 255] = 1
                if 'MR' in label_path:
                    label_1[label_1 != 63] = 0
                    label_1[label_1 == 63] = 1
                yield dicom, label_1
            else:
                if 'CT' in label_path:
                    label_2[label_2 == 255] = 1
                if 'MR' in label_path:
                    label_2[label_2 != 63] = 0
                    label_2[label_2 == 63] = 1
                label_1 = 2 * (label_1 / 255)
                label_1 = label_1.astype(np.int8)
                yield dicom, label_1, label_2


if __name__ == '__main__':
    params = {'modality': 'CT', 'shuffle': False, 'branch': 2}
    label_gen = data_gen(dataset='eval', params=params)
    cv2.namedWindow('Test', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('Test_1', cv2.WINDOW_FREERATIO)
    for i, j, k in label_gen:
        print('loop label_1', np.unique(j))
        print('loop label_2', np.unique(k))
        if len(np.unique(j)) > 2:
            cv2.imshow('Test_1', j)
            cv2.waitKey()
