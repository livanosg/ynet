import tensorflow as tf
eps = 1e-7


def weighted_crossentropy(predictions, labels):  # todo fix weights shape
    """weighted softmax_cross_entropy"""
    with tf.name_scope('Weighted_Crossentropy'):
        class_freq = tf.reduce_sum(labels, axis=[0, 1, 2], keepdims=True)
        weights = tf.math.divide(tf.reduce_sum(class_freq, axis=-1, keepdims=True), tf.math.maximum(eps, class_freq))
        log = weights * tf.reduce_sum(labels * tf.math.log(tf.math.maximum(eps, predictions)), axis=[0, 1, 2])
        log_loss = tf.math.pow(-log, 0.3)
        return tf.reduce_mean(log_loss)


def weighted_log_dice_loss(labels, predictions):
    """both tensors are [b, h, w, classes] and y_pred is in probs form"""

    class_freq = tf.reduce_sum(labels, axis=[0, 1, 2])  # class freqs for each image batch [b, h, w, classes] => [class_freq]
    class_freq_eps = tf.math.maximum(class_freq, eps)
    weights = 1 / class_freq_eps**2
    numerator = tf.reduce_sum(labels * predictions, axis=[0, 1, 2])  # label * pred = TP of each class
    numerator = weights * numerator
    numerator = tf.where(tf.equal(class_freq, 0), tf.ones_like(numerator), numerator)
    denominator = tf.reduce_sum(labels + predictions, axis=[0, 1, 2])  # label + pred = 2 * TP + FP + FN of each class
    denominator = weights * denominator
    denominator = tf.where(tf.equal(class_freq, 0), tf.ones_like(denominator) * 2, denominator)
    dice = 2. * (numerator/denominator)
    with tf.control_dependencies([tf.print(dice)]):
        dice = -tf.math.log(dice)
    # dice = tf.where(tf.is_finite(-tf.math.log(dice)),  -tf.math.log(dice), -tf.math.log(dice + 1))
    with tf.control_dependencies([tf.print(dice)]):
        dice = tf.math.reduce_mean(dice)
    loss = tf.math.pow(dice, 0.3)
    return loss


def custom_loss(predictions, labels):
    with tf.name_scope('Custom_loss'):
        loss = weighted_log_dice_loss(predictions=predictions, labels=labels)
        # wce_loss = weighted_crossentropy(predictions=predictions, labels=labels)
        # loss = tf.math.multiply(.3, dice_loss) + tf.math.multiply(0.7, wce_loss)
        return loss
