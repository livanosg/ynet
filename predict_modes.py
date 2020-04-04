import os
import cv2
import numpy as np
import tensorflow as tf
import config
import input_fns
import model_fns


def run_chaos_test(args):
    model_fn_params = {'dropout': 0., 'classes': args.classes, 'distribution': args.nodist}
    if not args.nodist:
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # If op cannot be executed on GPU ==> CPU.
        session_config.gpu_options.allow_growth = True  # Allow full memory usage of GPU.

    load_path = config.paths['save'] + '/' + args.load_model
    ynet = tf.estimator.Estimator(model_fn=model_fns.ynet_model_fn, model_dir=load_path, params=model_fn_params)
    pred_input_fn = input_fns.Input_function('chaos-test', args)
    predicted = ynet.predict(input_fn=lambda: pred_input_fn.get_tf_generator(),
                             predict_keys=['final_prediction', 'path'],
                             yield_single_examples=True)
    for idx, output in enumerate(predicted):
        path = output['path'].decode("utf-8")
        if args.modality == 'ALL':
            new_path = path.replace('Test_Sets', 'Test_Sets/Task1')
        elif args.modality == 'CT':
            new_path = path.replace('Test_Sets', 'Test_Sets/Task2')
        else:
            new_path = path.replace('Test_Sets', 'Test_Sets/Task3')
        if 'CT' in new_path:
            intensity = 255
        else:  # 'MR' in new_path:
            intensity = 63
        results = output['final_prediction'].astype(np.uint8) * intensity
        new_path = new_path.replace('DICOM_anon', 'Results')
        new_path = new_path.replace('.dcm', '.png')
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        cv2.imwrite(new_path, results)


def predict(args):
    model_fn_params = {'dropout': 0., 'classes': args.classes, 'distribution': args.nodist}

    if not args.nodist:
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # If op cannot be executed on GPU ==> CPU.
        session_config.gpu_options.allow_growth = True  # Allow full memory usage of GPU.

    load_path = config.paths['save'] + '/' + args.load_model
    ynet = tf.estimator.Estimator(model_fn=model_fns.ynet_model_fn, model_dir=load_path, params=model_fn_params)
    pred_input_fn = input_fns.Input_function('pred', args)

    predicted = ynet.predict(input_fn=lambda: pred_input_fn.get_tf_generator(),
                             predict_keys=['final_prediction', 'path'], yield_single_examples=True)
    for idx, output in enumerate(predicted):
        path = output['path'].decode("utf-8")
        new_path = path.replace('DICOM_anon', 'Results')
        new_path = new_path.replace('.dcm', '.png')
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        results = output['final_prediction'].astype(np.uint8) * 255
        cv2.imwrite(new_path, results)
