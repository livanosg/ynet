from contextlib import nullcontext

import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer as Adam
from tensorflow.python import summary
from loss_fn import custom_loss
from tensorflow.compat.v1 import estimator
from archit import ynet


def ynet_model_fn(features, labels, mode, params):

    loss, train_op, = None, None
    eval_metric_ops, training_hooks, evaluation_hooks = None, None, None
    predictions_dict = None
    output_1, output_2 = ynet(input_tensor=features['image'], params=params)
    with tf.name_scope('arg_max_outputs'):
        output_1_arg = tf.math.argmax(output_1, axis=-1)
        output_2_arg = tf.math.argmax(output_2, axis=-1)
    with tf.name_scope('Final_Output_Calculations'):
        final_output = tf.where(tf.equal(output_2_arg, 1), output_2_arg, output_1_arg)
        final_output = tf.where(tf.equal(output_2_arg, 2), tf.zeros_like(final_output), final_output)
        one_hot_final_output = tf.one_hot(indices=final_output, depth=2)

    if mode in (estimator.ModeKeys.TRAIN, estimator.ModeKeys.EVAL):
        with tf.name_scope('Second_Branch_Label_Calculations'):
            label_1 = tf.arg_max(labels['label'], -1)
            label_2 = label_1 + output_1_arg * 2  # FN == 1, FP == 2
            label_2 = tf.stop_gradient(label_2)
            label_2 = tf.where(tf.equal(label_2, 3), tf.zeros_like(label_2), label_2)
            one_hot_label_2 = tf.one_hot(indices=label_2, depth=params['classes'] ** 2 - params['classes'] + 1)
        with tf.name_scope('Loss_Calculation'):
            loss_1 = custom_loss(predictions=output_1, labels=labels['label'])
            loss_2 = custom_loss(predictions=output_2, labels=one_hot_label_2)
        if params['branch'] == 1:
            loss = loss_1 + (0 * loss_2)
        else:
            loss = (0 * loss_1) + loss_2
        with tf.name_scope('Dice_Score_Calculation'):
            dice_output_1 = tf.contrib.metrics.f1_score(labels=labels['label'], predictions=output_1)
            dice_output_2 = tf.contrib.metrics.f1_score(labels=one_hot_label_2, predictions=output_2)
            dice_final = tf.contrib.metrics.f1_score(labels=labels['label'], predictions=one_hot_final_output)

    if mode in (estimator.ModeKeys.TRAIN, estimator.ModeKeys.EVAL):
        with tf.name_scope('Branch_{}_training'.format(params['branch'])):
            with tf.name_scope('{}'.format(mode)):  # The Inputs and outputs of the algorithm
                input_img = tf.math.divide(features['image'] - tf.reduce_max(features['image'], [0, 1, 2]),
                                           tf.reduce_max(features['image'], [0, 1, 2]) - tf.reduce_min(features['image'], [0, 1, 2]))
                input_img = tf.cast(input_img * 255, tf.uint8)

                output_1_img = tf.expand_dims(tf.cast(output_1_arg * 255, dtype=tf.uint8), axis=-1)
                label_1_img = tf.expand_dims(tf.cast(label_1 * 255, dtype=tf.uint8), axis=-1)

                output_2_img = tf.expand_dims(tf.cast(output_2_arg * 127 + 1, dtype=tf.uint8), axis=-1)
                label_2_img = tf.expand_dims(tf.cast(label_2, dtype=tf.uint8) * 127 + 1, axis=-1)

                final_output_img = tf.expand_dims(tf.cast(final_output * 255, dtype=tf.uint8), axis=-1)
                summary.image('1_Medical_Image', input_img, max_outputs=1)
                summary.image('2_Output_1_label', label_1_img, max_outputs=1)
                summary.image('3_Output_1', output_1_img, max_outputs=1)
                summary.image('4_Final', final_output_img, max_outputs=1)
                summary.image('5_Output_2', output_2_img, max_outputs=1)
                summary.image('6_Output_2_label', label_2_img, max_outputs=1)
    if mode == estimator.ModeKeys.TRAIN:
        with tf.name_scope('Learning_Rate'):
            global_step = tf.compat.v1.train.get_or_create_global_step()
            learning_rate = tf.compat.v1.train.exponential_decay(params['lr'], global_step=global_step,
                                                                 decay_steps=params['decay_steps'],
                                                                 decay_rate=params['decay_rate'], staircase=False)
        with tf.name_scope('Optimizer_conf'):
            if params['branch'] == 1:
                var_list = None
            else:
                var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'Model/Branch_2/')
            train_op = Adam(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step, var_list=var_list)
            # final_train_op = tf.group(train_op_1, train_op_2)
    if mode == estimator.ModeKeys.TRAIN:
        with tf.name_scope('Metrics'):
            summary.scalar('1_Output_1_DSC', dice_output_1[1])
            summary.scalar('2_Final_DSC', dice_final[1])
            summary.scalar('3_Output_2_DSC', dice_output_2[1])
            summary.scalar('Learning_Rate', learning_rate)
    if mode == estimator.ModeKeys.EVAL:
        eval_metric_ops = {'Metrics/1_Output_1_DSC': dice_output_1,
                           'Metrics/2_Final_DSC': dice_final,
                           'Metrics/3_Output_2_DSC': dice_output_2}

        with tf.name_scope('Evaluation_Summary_Hook'):
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
