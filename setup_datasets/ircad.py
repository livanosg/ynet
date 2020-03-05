import os
import cv2
import zipfile
import numpy as np
import pydicom as pd
from glob import glob
from config import setup_paths
from shutil import rmtree


def unzip_ircad():
    conf_pat = {'file': 'PATIENT_DICOM.zip',
                'zip_sub': 'PATIENT_DICOM/',
                'to': 'DICOM_anon/'}
    conf_label = {'file': 'MASKS_DICOM.zip',
                  'zip_sub': 'MASKS_DICOM/liver/',
                  'to': 'Ground_dcm/'}
    with zipfile.ZipFile(setup_paths['irdcab_zip'], 'r') as zip_ref:
        zip_ref.extractall(setup_paths['irdcab_root'])
    for conf in [conf_pat, conf_label]:
        for dirpath, dirnames, filenames in os.walk(setup_paths['irdcab_root']):
            for file in filenames:
                if file == conf['file']:
                    filepath = os.path.join(dirpath, file)
                    print('Extracting {}'.format(filepath))
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        liver_list = [liver_path for liver_path in zip_ref.namelist() if conf['zip_sub'] in liver_path]
                        for liver_path in liver_list:
                            filepath = os.path.join(dirpath, liver_path)
                            zip_ref.extract(member=zip_ref.getinfo(liver_path), path=dirpath)
                            if os.path.isfile(filepath):
                                os.makedirs(os.path.dirname(filepath.replace(conf['zip_sub'], conf['to']) + '.dcm'),
                                            exist_ok=True)
                                os.replace(filepath, filepath.replace(conf['zip_sub'], conf['to']) + '.dcm')
                        os.removedirs(os.path.dirname(filepath))


def rename_ircad():
    print('Splitting dataset from main folder...')
    patient = sorted(glob(setup_paths['irdcab_root'] + '/**'))
    for i, patient_id in enumerate(patient):
        os.replace(patient_id, patient_id.replace(os.path.basename(patient_id), str(i + 1)))
    patient_2 = sorted(glob(setup_paths['irdcab_root'] + '/**/**'))
    for i in patient_2:
        if 'DICOM_anon' in i or 'Ground_dcm' in i:
            os.makedirs(os.path.dirname(i.replace(setup_paths['irdcab_root'], setup_paths['irdcab_train'] + '/CT')), exist_ok=True)
            os.replace(i, i.replace(setup_paths['irdcab_root'], setup_paths['irdcab_train'] + '/CT'))
    print('Removing Extracted Folder...')
    rmtree(setup_paths['irdcab_root'])
    print('Done!')


def make_png_ircad():
    ground_dcm = sorted(glob(setup_paths['irdcab_train'] + '/**/Ground_dcm/**.dcm', recursive=True))
    print('Converting dcm labels to png...')
    for ground_dcm_path in ground_dcm:
        ground_pxl = pd.dcmread(ground_dcm_path).pixel_array
        png_path = ground_dcm_path.replace('Ground_dcm', 'Ground').replace('.dcm', '.png')
        max_val = np.max(ground_pxl)
        if max_val > 0:
            ground_pxl[ground_pxl == max_val] = 255
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        cv2.imwrite(png_path, ground_pxl)
    print('Removing dcm labels...')
    ground_dcm_folder_list = sorted(glob(setup_paths['irdcab_train'] + '/**/Ground_dcm', recursive=True))
    for ground_dcm_folder in ground_dcm_folder_list:
        rmtree(ground_dcm_folder)
    print('Done!')


def setup_ircadb():
    unzip_ircad()
    rename_ircad()
    make_png_ircad()
