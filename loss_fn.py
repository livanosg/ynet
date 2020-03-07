from tensorflow.keras.backend import epsilon
import tensorflow as tf


def weighted_crossentropy(predictions, labels):
    """weighted softmax_cross_entropy"""
    with tf.name_scope('Weighted_Crossentropy'):
        class_frequencies = tf.reduce_sum(labels, axis=[0, 1, 2], keepdims=True)
        weights = tf.div(tf.reduce_sum(class_frequencies)-class_frequencies, tf.add(class_frequencies, epsilon()))
        wce = tf.reduce_sum(weights * labels * tf.math.log(predictions + epsilon()))
        wce = tf.negative(tf.div(wce, tf.reduce_sum(class_frequencies)))
        return wce


def weighted_log_dice_loss(predictions, labels):
    """Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations.
    Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M. Jorge Cardoso"""

    with tf.name_scope('Generalized_Dice_Loss'):
        class_frequencies = tf.reduce_sum(labels, axis=[0, 1, 2])
        weights = tf.math.divide(1., tf.add(tf.pow(class_frequencies, 2), epsilon()))  # epsilon()
        numerator = tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(predictions, labels), axis=(0, 1, 2)), weights)) + epsilon()
        denominator = tf.reduce_sum(tf.multiply(tf.reduce_sum(labels + predictions, axis=(0, 1, 2)), weights))
        loss = tf.negative(tf.math.log(tf.multiply(tf.div(numerator, denominator), 2.0)))
    return loss


def custom_loss(predictions, labels):
    with tf.name_scope('Custom_loss'):
        dice_loss = weighted_log_dice_loss(predictions=predictions, labels=labels)
        xe_loss = weighted_crossentropy(predictions=predictions, labels=labels)
        loss = tf.math.multiply(.3, dice_loss) + tf.math.multiply(0.7, xe_loss)
        return loss
