from contextlib import nullcontext

import tensorflow as tf
from tensorflow.compat.v1 import train, estimator
from tensorflow import summary

import help_fn
import loss_fn
from archit import ynet, incept_ynet


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
    logits_1, logits_2 = incept_ynet(input_tensor=tf.concat([features['image']]*3, axis=-1), params=params)
    preds_1 = tf.math.softmax(logits_1, -1)
    preds_2 = tf.math.softmax(logits_2, -1)
    with device_1:
        with tf.name_scope('arg_max_outputs'):
            output_1_categ = tf.math.argmax(preds_1, axis=-1)
            output_2_categ = tf.math.argmax(preds_2, axis=-1)
        with tf.name_scope('Final_Output_Calculations'):
            # TN = 0, FP = 0, TP = 1, FN = 1
            final_output = tf.compat.v2.where(tf.math.logical_or(tf.equal(output_2_categ, 1), tf.equal(output_2_categ, 3)),
                                              tf.ones_like(output_2_categ), tf.zeros_like(output_2_categ))
            one_hot_final_output = tf.one_hot(indices=final_output, depth=params['classes'])
        if mode in (estimator.ModeKeys.TRAIN, estimator.ModeKeys.EVAL):
            with tf.name_scope('Label2_Calculations'):
                label_1_categ = tf.math.argmax(labels['label'], -1)
                # TN = 0, FN = 1, FP = 2, TP = 3
                label_2_categ = label_1_categ + (output_1_categ * params['classes'])
                label_2_one_hot = tf.one_hot(indices=tf.cast(label_2_categ, tf.int32), depth=(params['classes'] ** 2))

            with tf.name_scope('Dice_Score_Calculation'):
                dice_output_1 = help_fn.f1(labels=labels['label'], predictions=preds_1)
                dice_output_2 = help_fn.f1(labels=label_2_one_hot, predictions=preds_2)
                dice_final = help_fn.f1(labels=labels['label'], predictions=one_hot_final_output)

            with tf.name_scope('Branch_{}_training'.format(params['branch'])):
                with tf.name_scope('{}'.format(mode)):
                    input_max = tf.reduce_max(features['image'], [0, 1, 2])
                    input_min = tf.reduce_min(features['image'], [0, 1, 2])
                    input_img = tf.math.divide(tf.math.subtract(features['image'], input_max),
                                               tf.math.subtract(input_max, input_min))
                    output_1_img = tf.expand_dims(tf.cast(output_1_categ * 255, dtype=tf.uint8), axis=-1)  # 1 channel
                    output_1_preds_img = tf.expand_dims(preds_1[:, :, :, 1], axis=-1)  # 1 channel
                    label_1_img = tf.expand_dims(tf.cast(label_1_categ * 255, dtype=tf.uint8), axis=-1)  # 1 channel
                    output_2_img = tf.cast(tf.one_hot(output_2_categ, depth=params['classes'] ** 2)[:, :, :, 1:] * 255,
                                           dtype=tf.uint8)  # 3 channels
                    output_2_preds_img = preds_2[:, :, :, 1:]  # 3 channels
                    label_2_img = tf.cast(label_2_one_hot[:, :, :, 1:] * 255, dtype=tf.uint8)
                    final_output_img = tf.expand_dims(tf.cast(final_output * 255, dtype=tf.uint8), axis=-1)  # 1 channel

                    summary.image('1_Medical_Image', input_img, max_outputs=1)
                    summary.image('2_Output_1_label', label_1_img, max_outputs=1)
                    summary.image('3_Output_1', output_1_img, max_outputs=1)
                    summary.image('4_Output_1_preds', output_1_preds_img, max_outputs=1)
                    summary.image('5_Final', final_output_img, max_outputs=1)
                    summary.image('6_Output_2', output_2_img, max_outputs=1)
                    summary.image('7_Output_2_preds', output_2_preds_img, max_outputs=1)
                    summary.image('8_Output_2_label', label_2_img, max_outputs=1)
            if params['branch'] == 1:
                with tf.name_scope('Loss_Calculation'):
                    loss = loss_fn.custom_loss(logits=logits_1, labels=labels['label'])
            else:
                with tf.name_scope('Loss_Calculation'):
                    loss = loss_fn.custom_loss(logits=logits_2, labels=label_2_one_hot)
    with device_2:
        if mode == estimator.ModeKeys.TRAIN:
            with tf.name_scope('Learning_Rate'):
                global_step = train.get_or_create_global_step()
                learning_rate = help_fn.cyclic_learning_rate(global_step, learning_rate=params['lr'],
                                                             max_lr=2.5 * params['lr'],
                                                             step_size=params['steps_per_epoch'] * 5, gamma=0.999991,
                                                             mode='exp_range', name=None)
            with tf.name_scope('Optimizer'):
                var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                       'Model/Up{}'.format(params['branch']))
                train_op = train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,
                                                                                     var_list=var_list,
                                                                                     global_step=global_step)
            with tf.name_scope('Metrics'):
                summary.scalar('1_Output_1_DSC', dice_output_1[1])
                summary.scalar('2_Final_DSC', dice_final[1])
                summary.scalar('Learning_Rate', learning_rate)
                summary.scalar('3_Output_2_DSC', dice_output_2[1])

        if mode == estimator.ModeKeys.EVAL:
            eval_metric_ops = {'Metrics/1_Output_1_DSC': dice_output_1, 'Metrics/2_Final_DSC': dice_final,
                               'Metrics/3_Output_2_DSC': dice_output_2}

            eval_summary_hook = estimator.SummarySaverHook(output_dir=params['eval_path'], summary_op=summary.merge_all(),
                                                           save_steps=params['eval_steps'])
            evaluation_hooks = [eval_summary_hook]
    if mode == estimator.ModeKeys.PREDICT:
        predictions_dict = {'image': features['image'],
                            'preds_1': output_1_categ,
                            'preds_2': output_2_categ,
                            'final_prediction': final_output,
                            'path': features['path']}

    return estimator.EstimatorSpec(mode=mode, predictions=predictions_dict, loss=loss, train_op=train_op,
                                   eval_metric_ops=eval_metric_ops, training_hooks=training_hooks,
                                   evaluation_hooks=evaluation_hooks)
