from contextlib import nullcontext

import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer as Adam
from tensorflow.python import summary
from tensorflow.compat.v1 import estimator
from archit import ynet
from help_fn import cyclic_learning_rate, f1
from loss_fn import custom_loss


# noinspection PyUnboundLocalVariable
def ynet_model_fn(features, labels, mode, params):
    if params['distribution']:
        device_1 = device_2 = nullcontext()
    else:
        device_1 = tf.device('/GPU:0')
        device_2 = tf.device('/GPU:1')

    loss, train_op, = None, None
    eval_metric_ops, training_hooks, evaluation_hooks = None, None, None
    predictions_dict = None
    output_1, output_2 = ynet(input_tensor=features['image'], params=params)
    with device_1:
        with tf.name_scope('arg_max_outputs'):
            output_1_arg = tf.math.argmax(output_1, axis=-1)
            output_2_arg = tf.math.argmax(output_2, axis=-1)
        with tf.name_scope('Final_Output_Calculations'):
            final_output = tf.compat.v2.where(tf.equal(output_2_arg, tf.ones_like(output_2_arg)), output_2_arg,
                                              output_1_arg)
            final_output = tf.compat.v2.where(tf.equal(output_2_arg, 2), tf.zeros_like(output_2_arg), final_output)
            one_hot_final_output = tf.one_hot(indices=final_output, depth=2)
        if mode in (estimator.ModeKeys.TRAIN, estimator.ModeKeys.EVAL):
            label_1_arg = tf.math.argmax(labels['label'], -1)
            with tf.name_scope('Second_Branch_Label_Calculations'):
                label_2_arg = tf.where(tf.equal(label_1_arg, output_1_arg), tf.zeros_like(label_1_arg),
                                       tf.ones_like(label_1_arg))
                label_2_arg = tf.where(tf.greater(output_1_arg, label_1_arg), tf.ones_like(label_1_arg) * 2,
                                       label_2_arg)
                label_2_one_hot = tf.one_hot(indices=tf.cast(label_2_arg, tf.int32),
                                             depth=(params['classes'] ** 2) - params['classes'] + 1)

            with tf.name_scope('Dice_Score_Calculation'):
                dice_output_1 = f1(labels=labels['label'], predictions=output_1)
                dice_output_2 = f1(labels=label_2_one_hot, predictions=output_2)
                dice_final = f1(labels=labels['label'], predictions=one_hot_final_output)

            with tf.name_scope('Branch_{}_training'.format(params['branch'])):
                with tf.name_scope('{}'.format(mode)):
                    input_img = tf.math.divide(features['image'] - tf.reduce_max(features['image'], [0, 1, 2]),
                                               tf.reduce_max(features['image'], [0, 1, 2]) - tf.reduce_min(
                                                   features['image'], [0, 1, 2]))
                    final_output_img = tf.expand_dims(tf.cast(final_output * 255, dtype=tf.uint8), axis=-1)
                    output_1_img = tf.expand_dims(tf.cast(output_1_arg * 255, dtype=tf.uint8), axis=-1)
                    label_1_img = tf.expand_dims(tf.cast(label_1_arg * 255, dtype=tf.uint8), axis=-1)
                    output_2_img = tf.concat([tf.expand_dims((1 - output_1[:, :, :, 1]), -1),
                                             tf.expand_dims((output_1[:, :, :, 1]), -1),
                                             tf.expand_dims(tf.zeros_like(output_1[:, :, :, 1]), -1)], -1)
                    summary.image('1_Medical_Image', input_img, max_outputs=1)
                    summary.image('2_Output_1_label', label_1_img, max_outputs=1)
                    summary.image('3_Output_1', output_1_img, max_outputs=1)
                    summary.image('4_Output_1_preds', output_2_img, max_outputs=1)
                    summary.image('5_Final', final_output_img, max_outputs=1)
                    summary.image('6_Output_2', tf.one_hot(output_2_arg, depth=3), max_outputs=1)
                    summary.image('7_Output_2_preds', output_2, max_outputs=1)
                    summary.image('8_Output_2_label', label_2_one_hot, max_outputs=1)
            if params['branch'] == 1:
                with tf.name_scope('Loss_Calculation'):
                    loss = custom_loss(predictions=output_1, labels=labels['label'])
            else:
                with tf.name_scope('Loss_Calculation'):
                    loss = custom_loss(predictions=output_2, labels=label_2_one_hot)
    with device_2:
        if mode == estimator.ModeKeys.TRAIN:
            with tf.name_scope('Learning_Rate'):
                global_step = tf.compat.v1.train.get_or_create_global_step()
                # learning_rate = tf.compat.v1.train.exponential_decay(params['lr'], global_step=global_step,
                #                                                      decay_steps=params['decay_steps'],
                #                                                      decay_rate=params['decay_rate'], staircase=False)
                learning_rate = cyclic_learning_rate(global_step, learning_rate=params['lr'], max_lr=5 * params['lr'],
                                                     step_size=params['steps_per_epoch'] * 5, gamma=0.999994,
                                                     mode='exp_range', name=None)
            with tf.name_scope('Optimizer'):
                var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                       'Model/Branch_{}/'.format(params['branch']))
                optimizer = Adam(learning_rate=learning_rate)
                grads = optimizer.compute_gradients(loss=loss, var_list=var_list)
                train_op = optimizer.apply_gradients(grads_and_vars=grads, global_step=global_step)
            with tf.name_scope('Metrics'):
                summary.scalar('1_Output_1_DSC', dice_output_1[1])
                summary.scalar('2_Final_DSC', dice_final[1])
                summary.scalar('Learning_Rate', learning_rate)
                summary.scalar('3_Output_2_DSC', dice_output_2[1])

        if mode == estimator.ModeKeys.EVAL:
            eval_metric_ops = {'Metrics/1_Output_1_DSC': dice_output_1, 'Metrics/2_Final_DSC': dice_final,
                               'Metrics/3_Output_2_DSC': dice_output_2}

            eval_summary_hook = tf.estimator.SummarySaverHook(output_dir=params['eval_path'],
                                                              summary_op=summary.merge_all(),
                                                              save_steps=params['eval_steps'])
            evaluation_hooks = [eval_summary_hook]

    if mode == estimator.ModeKeys.PREDICT:
        predictions_dict = {'image': features['image'],
                            'output_1': output_1_arg,
                            'output_2': output_2_arg,
                            'final_prediction': final_output,
                            'path': features['path']}

    return estimator.EstimatorSpec(mode,
                                   predictions=predictions_dict,
                                   loss=loss,
                                   train_op=train_op,
                                   eval_metric_ops=eval_metric_ops,
                                   training_hooks=training_hooks,
                                   evaluation_hooks=evaluation_hooks)
