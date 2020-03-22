import numpy as np
import tensorflow as tf
from os import environ, makedirs
from os.path import dirname
from math import ceil
from cv2 import imwrite
from tensorflow.compat.v1.logging import info
from tensorflow.estimator.experimental import stop_if_no_decrease_hook
from data_generators import data_gen
from help_fn import get_model_paths
from input_fns import train_eval_input_fn, pred_input_fn
from model_fn import ynet_model_fn
from config import paths, root_dir
from logs_script import save_logs


def estimator_mod(args):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    # Distribution Strategy

    environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit " + root_dir
    environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    warm_start = None
    session_config = None
    strategy = None

    if not args.nodist:
        strategy = tf.distribute.MirroredStrategy()
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # If op cannot be executed on GPU ==> CPU.
        session_config.gpu_options.allow_growth = True  # Allow full memory usage of GPU.

    if args.load_model:
        load_path = paths['save'] + '/' + args.load_model
        if args.resume:
            model_path = load_path
            eval_path = load_path + '/eval'
        else:
            if args.branch == 1:
                warm_start = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=load_path,
                                                            vars_to_warm_start=".*Model.*")
            else:
                warm_start = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=load_path,
                                                            vars_to_warm_start=".*Model/Branch_1.*")
            model_path, eval_path = get_model_paths(args)
    else:
        model_path, eval_path = get_model_paths(args)

    input_fn_params = {'classes': args.classes,
                       'modality': args.modality,
                       'augm_prob': args.augm_prob,
                       'batch_size': args.batch_size,
                       'shuffle': True}

    if args.mode not in ('lr', 'test'):
        train_size = len(list(data_gen(dataset='train', params=input_fn_params, only_paths=True)))
        eval_size = len(list(data_gen(dataset='eval', params=input_fn_params, only_paths=True)))
    else:
        train_size = 10
        eval_size = 5
    steps_per_epoch = ceil(train_size / args.batch_size)
    max_training_steps = args.epochs * steps_per_epoch

    # Eval and predict options
    if args.mode == 'pred':
        input_fn_params['shuffle'] = False
        input_fn_params['batch_size'] = 1

    if args.mode == 'lr':
        save_summary_steps = args.batch_size
    else:
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
                       'distribution': args.nodist}

    # TODO use it to define learning rate
    # Global batch size for a step ==> _PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync
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

    ynet = tf.estimator.Estimator(model_fn=ynet_model_fn, model_dir=model_path, params=model_fn_params,
                                  config=configuration, warm_start_from=warm_start)

    # Early Stopping Strategy hook *Bugged for MultiWorkerMirror
    early_stop_steps = steps_per_epoch * args.early_stop
    early_stopping = stop_if_no_decrease_hook(ynet, metric_name='loss', max_steps_without_decrease=early_stop_steps)
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
                                          steps=eval_size, start_delay_secs=1, throttle_secs=0)
        # EVAL_STEPS => set or evaluator hangs or dont repeat in input_fn
        tf.estimator.train_and_evaluate(ynet, train_spec=train_spec, eval_spec=eval_spec)
        info('Train and Evaluation Mode Finished!\nMetrics and checkpoints are saved at: {}'.format(model_path))

    if args.mode in ('train', 'lr'):
        ynet.train(input_fn=lambda: train_eval_input_fn(mode='train', params=input_fn_params),
                   steps=max_training_steps, hooks=[early_stopping])

    if args.mode == tf.estimator.ModeKeys.EVAL:
        results = ynet.evaluate(input_fn=lambda: train_eval_input_fn(mode='eval', params=input_fn_params))
        print(results)

    if args.mode == tf.estimator.ModeKeys.PREDICT:  # Prediction mode used for test data of CHAOS challenge
        input_fn_params['shuffle'] = False
        predicted = ynet.predict(input_fn=lambda: pred_input_fn(params=input_fn_params),
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
        predicted = ynet.predict(input_fn=lambda: pred_input_fn(params=input_fn_params),
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
