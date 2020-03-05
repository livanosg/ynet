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
    output_1, output_2 = ynet(input_tensor=features['dicom'], params=params)
    with tf.name_scope('arg_max_outputs'):
        output_1_arg = tf.math.argmax(output_1, axis=-1)
        output_2_arg = tf.math.argmax(output_2, axis=-1)
    with tf.name_scope('Final_Output_Calculations'):
        final_output = tf.where(tf.equal(output_1_arg, 1), output_1_arg, tf.zeros_like(output_1_arg))
        final_output = tf.where(tf.equal(output_2_arg, 1), output_2_arg, final_output)
        final_output = tf.where(tf.equal(output_2_arg, 2), tf.zeros_like(final_output), final_output)
    with tf.name_scope('Prediction_Mode_Outputs'):
        if mode == estimator.ModeKeys.PREDICT:
            predictions_dict = {'dicom': features['dicom'],
                                'output_1': output_1_arg,
                                'output_2': output_2_arg,
                                'final_prediction': final_output,
                                'path': features['path']}

    if mode in (estimator.ModeKeys.TRAIN, estimator.ModeKeys.EVAL):
        with tf.name_scope('Loss_Calculation'):
            if params['branch'] == 1:
                loss_1 = custom_loss(predictions=output_1, labels=labels['label_1'])
                loss_2 = custom_loss(predictions=output_2, labels=tf.zeros_like(output_2))
                loss = loss_1 + 0 * loss_2
            else:
                loss_1 = custom_loss(predictions=output_1, labels=tf.zeros_like(output_1))
                loss_2 = custom_loss(predictions=output_2, labels=labels['label_1'])
                loss = 0 * loss_1 + loss_2
        with tf.name_scope('Dice_Score_Calculation'):
            if params['branch'] == 1:
                dice_branch = tf.contrib.metrics.f1_score(labels=labels['label_1'][:, :, :, 1], predictions=output_1[:, :, :, 1])  # todo name_scope
                dice_final = tf.contrib.metrics.f1_score(labels=labels['label_1'][:, :, :, 1], predictions=final_output)
            else:
                dice_branch = tf.contrib.metrics.f1_score(labels=labels['label_1'], predictions=output_2)
                dice_out_1 = tf.contrib.metrics.f1_score(labels=labels['label_2'], predictions=output_1[:, :, :, 1])
                dice_final = tf.contrib.metrics.f1_score(labels=labels['label_2'], predictions=final_output)

        with tf.name_scope('{}_images'.format(mode)):  # The Inputs and outputs of the algorithm
            with tf.name_scope('Branch_{}_training'.format(params['branch'])):
                summary.image('Input_Image', features['dicom'], max_outputs=1)
                summary.image('Branch_{}_label'.format(params['branch']), tf.expand_dims(tf.cast(tf.math.argmax(labels['label_1'], axis=-1), dtype=tf.float32), axis=-1), max_outputs=1)
                summary.image('Output_1', tf.expand_dims(tf.cast(output_1_arg, dtype=tf.float32), axis=-1), max_outputs=1)
                summary.image('Output_2', tf.expand_dims(tf.cast(output_2_arg, dtype=tf.float32), axis=-1), max_outputs=1)
                summary.image('Final', tf.expand_dims(tf.cast(final_output, dtype=tf.float32), axis=-1), max_outputs=1)
                if params['branch'] == 2:
                    summary.image('Branch_1_label', tf.expand_dims(tf.cast(labels['label_2'], dtype=tf.float32), axis=-1), max_outputs=1)

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

        with tf.name_scope('Metrics'):
            summary.scalar('Branch_{}_DSC'.format(params['branch']), dice_branch[1])
            summary.scalar('Final_DSC', dice_final[1])
            summary.scalar('Learning_Rate', learning_rate)
            if params['branch'] == 2:
                summary.scalar('Branch_1_DSC', dice_out_1[1])
    if mode == estimator.ModeKeys.EVAL:
        if params['branch'] == 1:
            eval_metric_ops = {'Metrics/Branch_{}_DSC'.format(params['branch']): dice_branch,
                               'Metrics/Final_DSC'.format(params['branch']): dice_final}

        else:
            eval_metric_ops = {'Metrics/Branch_2_DSC': dice_branch,
                               'Metrics/Branch_1_DSC': dice_out_1,
                               'Metrics/Final_DSC'.format(params['branch']): dice_final}

        with tf.name_scope('Evaluation_Summary_Hook'):
            eval_summary_hook = tf.estimator.SummarySaverHook(output_dir=params['eval_path'],
                                                              summary_op=summary.merge_all(),
                                                              save_steps=params['eval_steps'])
            evaluation_hooks = [eval_summary_hook]
    return estimator.EstimatorSpec(mode,
                                   predictions=predictions_dict,
                                   loss=loss,
                                   train_op=train_op,
                                   eval_metric_ops=eval_metric_ops,
                                   training_hooks=training_hooks,
                                   evaluation_hooks=evaluation_hooks)
