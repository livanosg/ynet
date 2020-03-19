import sys

import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer as Adam
from tensorflow.python import summary
from tensorflow.compat.v1 import estimator
from archit import ynet
from help_fn import cyclic_learning_rate
from loss_fn import custom_loss


def ynet_model_fn(features, labels, mode, params):
    loss, train_op, = None, None
    eval_metric_ops, training_hooks, evaluation_hooks = None, None, None
    predictions_dict = None
    output_1, output_2 = ynet(input_tensor=features['image'], params=params)
    # print_op = tf.print(output_1[0:150:160, 150:160, 1], output_stream=sys.stdout)
    with tf.name_scope('arg_max_outputs'):
        # with tf.control_dependencies([print_op]):
        output_1_arg = tf.math.argmax(output_1, axis=-1)
        output_2_arg = tf.math.argmax(output_2, axis=-1)
    with tf.name_scope('Final_Output_Calculations'):
        final_output = tf.where(tf.equal(output_2_arg, 1), output_2_arg, output_1_arg)
        final_output = tf.where(tf.equal(output_2_arg, 2), tf.zeros_like(output_2_arg), final_output)
        one_hot_final_output = tf.one_hot(indices=final_output, depth=2)
    if mode in (estimator.ModeKeys.TRAIN, estimator.ModeKeys.EVAL):
        with tf.name_scope('Second_Branch_Label_Calculations'):
            label_1_arg = tf.arg_max(labels['label'], -1)
            label_2_arg = tf.where_v2(tf.equal(label_1_arg, output_1_arg), tf.zeros_like(label_1_arg), tf.ones_like(label_1_arg))
            label_2_arg = tf.where_v2(tf.greater(output_1_arg, label_1_arg), tf.ones_like(label_1_arg) * 2, label_2_arg)
            print_op_1 = tf.print(label_2_arg[0, 150, 150], output_stream=sys.stdout)
            # label_2 = label_1 + output_1_arg * 2  # FN == 1, FP == 2
            # label_2 = tf.where(tf.equal(label_2, 3), tf.zeros_like(label_2), label_2)
            with tf.control_dependencies([print_op_1]):
                label_2_one_hot = tf.one_hot(indices=tf.cast(label_2_arg, tf.int32), depth=params['classes'] ** 2 - params['classes'] + 1)
        with tf.name_scope('Loss_Calculation'):
            if params['branch'] == 1:
                loss = custom_loss(predictions=output_1, labels=labels['label'])
            else:
                loss = custom_loss(predictions=output_2, labels=label_2_one_hot)
        with tf.name_scope('Dice_Score_Calculation'):
            dice_output_1 = tf.contrib.metrics.f1_score(labels=labels['label'], predictions=output_1)
            dice_output_2 = tf.contrib.metrics.f1_score(labels=label_2_one_hot, predictions=output_2)
            dice_final = tf.contrib.metrics.f1_score(labels=labels['label'], predictions=one_hot_final_output)

        with tf.name_scope('Branch_{}_training'.format(params['branch'])):
            with tf.name_scope('{}'.format(mode)):  # The Inputs and outputs of the algorithm
                input_img = tf.math.divide(features['image'] - tf.reduce_max(features['image'], [0, 1, 2]),
                                           tf.reduce_max(features['image'], [0, 1, 2]) - tf.reduce_min(features['image'], [0, 1, 2]))
                output_1_img = tf.expand_dims(tf.cast(output_1_arg * 255, dtype=tf.uint8), axis=-1)
                label_1_img = tf.expand_dims(tf.cast(label_1_arg * 255, dtype=tf.uint8), axis=-1)

                output_2_img = tf.expand_dims(tf.cast(output_2_arg * 127 + 1, dtype=tf.uint8), axis=-1)
                label_2_img = tf.expand_dims(tf.cast(label_2_arg, dtype=tf.uint8) * 127 + 1, axis=-1)

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
            # learning_rate = tf.compat.v1.train.exponential_decay(params['lr'], global_step=global_step,
            #                                                      decay_steps=params['decay_steps'],
            #                                                      decay_rate=params['decay_rate'], staircase=False)
            learning_rate = cyclic_learning_rate(global_step, learning_rate=params['lr'], max_lr=5 * params['lr'],
                                                 step_size=params['steps_per_epoch'] * 5, gamma=0.9999, mode='exp_range', name=None)
        with tf.name_scope('Optimizer'):
            if params['branch'] == 1:
                var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'Model/Branch_1/')
            else:
                var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'Model/Branch_2/')
            optimizer = Adam(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss=loss, var_list=var_list)
            train_op = optimizer.apply_gradients(grads_and_vars=grads, global_step=global_step)
            # final_train_op = tf.group(train_op_1, train_op_2)
        with tf.name_scope('Metrics'):
            summary.scalar('1_Output_1_DSC', dice_output_1[1])
            summary.scalar('2_Final_DSC', dice_final[1])
            summary.scalar('3_Output_2_DSC', dice_output_2[1])
            summary.scalar('Learning_Rate', learning_rate)

    if mode == estimator.ModeKeys.EVAL:
        eval_metric_ops = {'Metrics/1_Output_1_DSC': dice_output_1,
                           'Metrics/2_Final_DSC': dice_final,
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
