import math
import os

import tensorflow as tf
import tensorflow.compat.v1.logging
import tensorflow.estimator.experimental

import config
import help_fn
import input_fns
import logs_script
import model_fns


def training_fn(args):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit " + config.root_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    warm_start = None
    strategy = None
    session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # If op cannot be executed on GPU ==> CPU.
    session_config.gpu_options.allow_growth = True  # Allow full memory usage of GPU.
    if not args.nodist:
        strategy = tf.distribute.MirroredStrategy()
    if args.load_model:
        load_path = config.paths['save'] + '/' + args.load_model
        if args.resume:
            model_path = load_path
            eval_path = load_path + '/eval'
        else:
            warm_start = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=load_path,
                                                        vars_to_warm_start=".*Model.*")
            model_path, eval_path = help_fn.get_model_paths(args)
    else:
        model_path, eval_path = help_fn.get_model_paths(args)

    train_fn = input_fns.Input_function('train', args)
    eval_fn = input_fns.Input_function('eval', args)

    if args.mode in ('lr', 'test'):
        train_size = 10
        eval_size = 5
    else:
        train_size = len(train_fn)
        eval_size = len(eval_fn)

    steps_per_epoch = math.ceil(train_size / args.batch_size)
    max_training_steps = args.epochs * steps_per_epoch

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
                       'decay_steps': math.ceil(args.epochs * steps_per_epoch / (args.decays_per_train + 1)),
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
                                           log_step_count_steps=math.ceil(steps_per_epoch / 2),
                                           train_distribute=strategy,
                                           session_config=session_config)

    ynet = tf.estimator.Estimator(model_fn=model_fns.ynet_model_fn, model_dir=model_path, params=model_fn_params,
                                  config=configuration, warm_start_from=warm_start)

    # Early Stopping Strategy hook *Bugged for MultiWorkerMirror
    early_stop_steps = steps_per_epoch * args.early_stop
    early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(ynet, metric_name='loss',
                                                                        max_steps_without_decrease=early_stop_steps)
    # Profiling hook *Bugged for MultiWorkerMirror, Configure training profiling ==> while only training,
    # Profiler steps < steps in liver_seg.train(. . .)
    # profiler_hook = tf.estimator.ProfilerHook(save_steps=steps_per_epoch * 2, output_dir=model_path, show_memory=True)

    if args.mode in ('train-and-eval', 'test'):
        log_data = {'train_size': train_size, 'steps_per_epoch': steps_per_epoch,
                    'max_training_steps': max_training_steps, 'eval_size': eval_size,
                    'eval_steps': eval_size, 'model_path': model_path}
        logs_script.save_logs(args, log_data)
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_fn.get_tf_generator(),
                                            hooks=[early_stopping], max_steps=max_training_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_fn.get_tf_generator(),
                                          steps=math.ceil(eval_size/args.batch_size), start_delay_secs=1, throttle_secs=0)
        # EVAL_STEPS => set or evaluator hangs or dont repeat in input_fn
        tf.estimator.train_and_evaluate(ynet, train_spec=train_spec, eval_spec=eval_spec)
        tensorflow.compat.v1.logging.info('Train and Evaluation Mode Finished!\n\
                                           Metrics and checkpoints are saved at: {}'.format(model_path))
    if args.mode in ('train', 'lr'):
        ynet.train(input_fn=lambda: train_fn.get_tf_generator(), steps=max_training_steps, hooks=[early_stopping])
    if args.mode == 'eval':
        ynet.evaluate(input_fn=lambda: eval_fn.get_tf_generator(), steps=math.ceil(eval_size/args.batch_size))
