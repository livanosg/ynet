import numpy as np
import tensorflow as tf
eps = 1e-8


def weighted_crossentropy(predictions, labels): #todo fix weights shape
    """weighted softmax_cross_entropy"""
    with tf.name_scope('Weighted_Crossentropy'):
        class_freq = tf.reduce_sum(labels, axis=[1, 2], keepdims=True)
        weights = tf.math.divide(tf.reduce_sum(class_freq, axis=-1, keepdims=True), tf.clip_by_value(class_freq, eps, tf.constant(np.inf)))
        log = tf.math.multiply(tf.math.log(tf.clip_by_value(predictions, eps, 1.)), labels)
        print_op = tf.print(log)
        with tf.control_dependencies([print_op]):
            wce = tf.math.multiply(weights, log)
        return tf.reduce_mean(tf.math.pow(tf.negative(wce), 0.3))


def weighted_log_dice_loss(labels, predictions, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in probs form"""

    class_freq = tf.reduce_sum(labels, [1, 2])
    # [b, h, w, classes]
    numerator = tf.reduce_sum(labels * predictions, axis=[1, 2])
    denominator = tf.reduce_sum(labels + predictions, axis=[1, 2])
    weights = 1 / class_freq**2
    weights = tf.where(tf.math.is_finite(weights), weights, tf.ones_like(weights) * eps)

    dice = tf.math.divide(tf.math.add(tf.math.multiply(weights * numerator, 2.), eps), tf.math.add(weights * denominator, eps))
    dice = tf.reduce_mean(dice, axis=-1)

    dice = -tf.math.log(tf.clip_by_value(dice, eps, 1.))
    print_op = tf.print(dice)
    with tf.control_dependencies([print_op]):
        loss = tf.reduce_mean(tf.math.pow(dice, 0.3))
    return loss


def custom_loss(predictions, labels):
    with tf.name_scope('Custom_loss'):
        # loss = weighted_log_dice_loss(predictions=predictions, labels=labels)
        loss = weighted_crossentropy(predictions=predictions, labels=labels)
        # loss = tf.math.multiply(.3, dice_loss) + tf.math.multiply(0.7, xe_loss)
        return loss
