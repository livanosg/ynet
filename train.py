import os
import numpy as np
import tensorflow as tf
from os import environ, makedirs
from os.path import dirname
from math import ceil
from cv2 import imwrite
from tensorflow.compat.v1.logging import info
from tensorflow.estimator.experimental import stop_if_no_decrease_hook
from data_generators import data_gen
from input_fns import train_eval_input_fn, pred_input_fn
from model_fn import ynet_model_fn
from config import paths, root_dir
from logs_script import save_logs


def estimator_mod(args):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    # Distribution Strategy
    environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit " + root_dir
    # TODO Implement on multi-nodes SLURM
    global_batch = args.batch_size
    if args.nodist:
        strategy = None
    else:
        strategy = tf.distribute.MirroredStrategy()
        global_batch = args.batch_size * strategy.num_replicas_in_sync
    # If op cannot be executed on GPU ==> assign to CPU.
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # Avoid error message if there is no gpu available.
    session_config.gpu_options.allow_growth = True  # Allow full memory usage of GPU.
    # Setting up working environment

    if args.resume:
        model_path = paths['save'] + '/' + args.load_model
        eval_path = model_path + '/eval'
        warm_start = None
    else:
        if args.load_model:
            warm_start_from = paths['save'] + '/' + args.load_model
            if args.branch == 1:
                warm_start = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=warm_start_from,
                                                            vars_to_warm_start=".*Model.*")
            else:
                warm_start = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=warm_start_from,
                                                            vars_to_warm_start=".*Model/Branch_1.*")
        else:
            warm_start = None
        trial = 0
        while os.path.exists(paths['save'] + '/{}_trial_{}'.format(args.modality, trial)):
            trial += 1
        model_path = paths['save'] + '/{}_trial_{}'.format(args.modality, trial)
        eval_path = model_path + '/eval'

    input_fn_params = {'classes': args.classes,
                       'modality': args.modality,
                       'augm_prob': args.augm_prob,
                       'batch_size': global_batch,
                       'shuffle': True}

    # Eval and predict options
    if args.mode == 'pred':
        input_fn_params['shuffle'] = False
        input_fn_params['batch_size'] = args.batch_size = 1

    train_size = len(list(data_gen(dataset='train', params=input_fn_params, only_paths=True)))
    eval_size = len(list(data_gen(dataset='eval', params=input_fn_params, only_paths=True)))

    if args.mode in ('lr', 'test'):
        train_size = 10
        eval_size = 5

    steps_per_epoch = ceil(train_size / global_batch)
    max_training_steps = args.epochs * steps_per_epoch
    save_summary_steps = steps_per_epoch
    model_fn_params = {'branch': args.branch,
                       'dropout': args.dropout,
                       'classes': args.classes,
                       'lr': args.lr,
                       'decay_rate': args.decay_rate,
                       'steps_per_epoch': steps_per_epoch,
                       'decay_steps': ceil(args.epochs * steps_per_epoch / (args.decays_per_train + 1)),
                       'eval_path': eval_path,
                       'eval_steps': eval_size,
                       'distribution': args.nodist,
                       'load_model': args.load_model,
                       'resume': args.resume}
    if args.mode == 'lr':
        model_fn_params['lr'] = 0.00005
        model_fn_params['decay_rate'] = 0.95
        model_fn_params['decay_steps'] = 30
        save_summary_steps = global_batch

        # Global batch size for a step ==> _PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync  # TODO use it to define learning rate
    # https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_estimator_limited_support
    # Your input_fn is called once per worker, thus giving one dataset per worker. Then one batch from that dataset
    # is fed to one replica on that worker, thereby consuming N batches for N replicas on 1 worker. In other words,
    # the dataset returned by the input_fn should provide batches of size PER_REPLICA_BATCH_SIZE. And the global
    # batch size for a step can be obtained as PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync.

    configuration = tf.estimator.RunConfig(tf_random_seed=args.seed,
                                           save_summary_steps=save_summary_steps,
                                           keep_checkpoint_max=args.early_stop + 2,
                                           save_checkpoints_steps=steps_per_epoch,
                                           log_step_count_steps=ceil(steps_per_epoch / 2),
                                           train_distribute=strategy,
                                           # eval_distribute=strategy, ==> breaks distributed training
                                           session_config=session_config)
    liver_seg = tf.estimator.Estimator(model_fn=ynet_model_fn,
                                       model_dir=model_path,
                                       params=model_fn_params,
                                       config=configuration,
                                       warm_start_from=warm_start)

    # Early Stopping Strategy hook *Bugged for MultiWorkerMirror
    steps_without_increase = steps_per_epoch * args.early_stop
    early_stopping = stop_if_no_decrease_hook(liver_seg, metric_name='loss', max_steps_without_decrease=steps_without_increase)
    # Profiling hook *Bugged for MultiWorkerMirror, Configure training profiling ==> while only training,
    # Profiler steps < steps in liver_seg.train(. . .)
    profiler_hook = tf.estimator.ProfilerHook(save_steps=steps_per_epoch * 2, output_dir=model_path, show_memory=True)

    if args.mode in ('train-and-eval', 'test'):
        log_data = {'train_size': train_size, 'steps_per_epoch': steps_per_epoch,
                    'max_training_steps': max_training_steps, 'eval_size': eval_size,
                    'eval_steps': eval_size, 'model_path': model_path}

        save_logs(args, log_data)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_eval_input_fn(mode='train', params=input_fn_params),
                                            hooks=[profiler_hook, early_stopping], max_steps=max_training_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: train_eval_input_fn(mode='eval', params=input_fn_params),
                                          steps=eval_size, start_delay_secs=1, throttle_secs=0)  # EVAL_STEPS => set or evaluator hangs or dont repeat in input_fn
        tf.estimator.train_and_evaluate(liver_seg, train_spec=train_spec, eval_spec=eval_spec)
        info('Train and Evaluation Mode Finished!\n'
             'Metrics and checkpoints are saved at:\n'
             '{}\n ----------'.format(model_path))

    if args.mode in ('train', 'lr'):
        liver_seg.train(input_fn=lambda: train_eval_input_fn(mode='train', params=input_fn_params),
                        steps=max_training_steps, hooks=[early_stopping])

    if args.mode == tf.estimator.ModeKeys.EVAL:
        results = liver_seg.evaluate(input_fn=lambda: train_eval_input_fn(mode='eval', params=input_fn_params))
        print(results)

    if args.mode == tf.estimator.ModeKeys.PREDICT:  # Prediction mode used for test data of CHAOS challenge
        input_fn_params['shuffle'] = False
        predicted = liver_seg.predict(input_fn=lambda: pred_input_fn(params=input_fn_params),
                                      predict_keys=['final_prediction', 'path'], yield_single_examples=True)
        for idx, output in enumerate(predicted):
            path = output['path'].decode("utf-8")
            new_path = path.replace('DICOM_anon', 'Results')
            new_path = new_path.replace('.dcm', '.png')
            makedirs(dirname(new_path), exist_ok=True)
            results = output['final_prediction'].astype(np.uint8) * 255
            imwrite(new_path, results)

    if args.mode == 'chaos-test':  # Prediction mode used for test data of CHAOS challenge
        input_fn_params['shuffle'] = False
        predicted = liver_seg.predict(input_fn=lambda: pred_input_fn(params=input_fn_params),
                                      predict_keys=['output_1', 'path'], yield_single_examples=True)
        for idx, output in enumerate(predicted):
            if args.modality == 'ALL':
                path = output['path'].decode("utf-8")
                new_path = path.replace('Test_Sets', 'Test_Sets/Task1')
            elif args.modality == 'CT':
                new_path = output['path'].replace('Test_Sets', 'Test_Sets/Task2')
            else:
                new_path = output['path'].replace('Test_Sets', 'Test_Sets/Task3')
            if 'CT' in new_path:
                intensity = 255
            else:  # 'MR' in new_path:
                intensity = 63
            results = output['predicted'].astype(np.uint8) * intensity
            new_path = new_path.replace('DICOM_anon', 'Results')
            new_path = new_path.replace('.dcm', '.png')
            makedirs(dirname(new_path), exist_ok=True)
            imwrite(new_path, results)

    # if args.mode == 'make-labels':  # Prediction mode used for test data of CHAOS challenge
    #     input_fn_params['shuffle'] = True
    #     predicted = liver_seg.predict(input_fn=lambda: make_labels_input_fn(params=input_fn_params), predict_keys=['image', 'output_1', 'path'], yield_single_examples=True)
    #     for idx, output in enumerate(predicted):
    #         output_1 = output['output_1'].astype(np.int32)
    #         label_path = output['path'].decode("utf-8")
    #         output['image'] = np.squeeze(output['image'])
    #         output['image'] = (output['image'] - np.min(output['image'])) / (np.max(output['image']) - np.min(output['image']))
    #         new_label_path = label_path.replace('Ground', 'Ground_2')
    #         label = cv2.imread(label_path, 0).astype(np.int32)
    #         info('Saving label: {}'.format(new_label_path))
    #         if 'CT' in label_path:
    #             label[label > 0] = 1
    #         if 'MR' in label_path:
    #             label[label != 63] = 0
    #             label[label == 63] = 1
    #         label_2 = label + output_1 * 2
    #         label_2[label_2 == 3] = 0
    #         label_2 = label_2 / 2
    #         label_2 = label_2 * 255
    #         makedirs(dirname(new_label_path), exist_ok=True)
    #         imwrite(new_label_path, label_2)
    #     info('Making labels Finished!')
    # exit()
