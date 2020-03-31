import tensorflow as tf

eps = 1e-8


def weighted_crossentropy(logits, labels):  # todo fix weights shape
    """weighted softmax_cross_entropy"""
    with tf.name_scope('Weighted_Crossentropy'):
        class_freq = tf.reduce_sum(labels, axis=[0, 1, 2], keepdims=True)
        class_freq = tf.math.maximum(class_freq, 1)
        weights = tf.math.pow(tf.math.divide(tf.reduce_sum(class_freq), class_freq), 0.5)
        weights = tf.multiply(weights, labels)
        weights = tf.reduce_max(weights, -1)
        loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=weights)
    return loss


def log_dice_loss(logits, labels):
    """both tensors are [b, h, w, classes] and y_pred is in probs form"""
    with tf.name_scope('Weighted_Generalized_Dice_Log_Loss'):
        predictions = tf.math.softmax(logits, -1)
        class_freq = tf.reduce_sum(labels, axis=[0, 1, 2])  # class freqs for each image batch [b, h, w, classes] => [class_freq]
        class_freq = tf.math.maximum(class_freq, 1)
        weights = 1 / (class_freq ** 2)

        numerator = tf.reduce_sum(labels * predictions, axis=[0, 1, 2])  # label * pred = TP of each class
        denominator = tf.reduce_sum(labels + predictions, axis=[0, 1, 2])  # label + pred = 2 * TP + FP + FN of each class
        dice = (2 * weights * (numerator + 1)) / (weights * (denominator + 1))
        loss = tf.math.reduce_mean(- tf.math.log(dice))
    return loss


def custom_loss(logits, labels):
    with tf.name_scope('Custom_loss'):
        dice_loss = log_dice_loss(logits=logits, labels=labels)
        wce_loss = weighted_crossentropy(logits=logits, labels=labels)
        loss = tf.math.multiply(.3, dice_loss) + tf.math.multiply(0.7, wce_loss)
    return loss
