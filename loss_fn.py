import tensorflow as tf
eps = 1e-15


def weighted_crossentropy(predictions, labels):
    """weighted softmax_cross_entropy"""
    with tf.name_scope('Weighted_Crossentropy'):
        class_frequencies = tf.reduce_sum(labels, axis=[0, 1, 2], keepdims=True)
        weights = tf.div(tf.reduce_sum(class_frequencies)-class_frequencies, tf.add(class_frequencies, eps))
        wce = tf.reduce_sum(weights * labels * tf.math.log(predictions + eps))
        wce = tf.negative(tf.div(wce, tf.reduce_sum(class_frequencies)))
        return wce

def weighted_log_dice_loss(predictions, labels):
    """Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations.
    Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M. Jorge Cardoso"""

    with tf.name_scope('Generalized_Dice_Loss'):
        class_frequencies = tf.reduce_sum(labels, axis=[0, 1, 2])
        weights = tf.math.divide(1., tf.pow(tf.add(class_frequencies, 1), 2))
        numerator = tf.math.multiply(2., tf.math.add(tf.reduce_sum(tf.math.multiply(labels, predictions), axis=[0, 1, 2]), 1e-7))
        # numerator = tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.multiply(labels, predictions) + 1, axis=(0, 1, 2)), weights)) + epsilon()
        denominator = tf.math.add(tf.reduce_sum(tf.math.add(labels, predictions), axis=[0, 1, 2]), 1e-7)
        dice = tf.math.multiply(weights, tf.math.divide(numerator, denominator))
        # denominator = tf.reduce_sum(tf.multiply(tf.reduce_sum(labels + predictions, axis=(0, 1, 2)), weights))
    return tf.reduce_mean(tf.math.pow(tf.negative(tf.math.log(dice)), 0.3))

def custom_loss(predictions, labels):
    with tf.name_scope('Custom_loss'):
        dice_loss = weighted_log_dice_loss(predictions=predictions, labels=labels)
        xe_loss = weighted_crossentropy(predictions=predictions, labels=labels)
        loss = tf.math.multiply(.3, dice_loss) + tf.math.multiply(0.7, xe_loss)
        return loss
