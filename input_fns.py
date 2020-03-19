import tensorflow as tf
from tensorflow.compat.v1.logging import info
from data_generators import data_gen, test_gen


def train_eval_input_fn(mode, params):
    """ input_fn for tf.estimator for TRAIN, EVAL and PREDICT modes.
    Inputs
    mode -> one of tf.estimator modes defined from tf.estimator.ModeKeys
    params -> arguments passed to data_generator and batch size"""
    with tf.name_scope('input_fn'):
        info(' Setting up {} dataset iterator...'.format(mode))
        # Don't declare generator to a variable or else Dataset.from_generator cannot instantiate the generator
        data_set = tf.data.Dataset.from_generator(generator=lambda: data_gen(mode, params),
                                                  output_types=(tf.float32, tf.int32),
                                                  output_shapes=(
                                                  tf.TensorShape([None, None]), tf.TensorShape([None, None])))
        data_set = data_set.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=params['classes'], dtype=tf.float32)))
        data_set = data_set.map(lambda x, y: (tf.expand_dims(tf.cast(x, tf.float32), -1), y))
        data_set = data_set.map(lambda x, y: ({'image': x}, {'label': y}))
        if mode == 'train':
            data_set = data_set.batch(params['batch_size'])
        if mode == 'eval':
            data_set = data_set.batch(1)
        if mode == tf.estimator.ModeKeys.TRAIN:
            data_set = data_set.repeat()
        data_set = data_set.prefetch(buffer_size=-1)
    return data_set


def pred_input_fn(params):
    data_set = tf.data.Dataset.from_generator(generator=lambda: test_gen(params=params),
                                              output_types=(tf.float32, tf.string),
                                              output_shapes=(tf.TensorShape([None, None]), tf.TensorShape(None)))
    data_set = data_set.map(lambda x, y: {'image': tf.expand_dims(x, -1), 'path': tf.cast(y, tf.string)})
    data_set = data_set.batch(params['batch_size'])
    data_set = data_set.prefetch(buffer_size=-1)
    return data_set

#  https://github.com/tensorflow/tensorflow/issues/13463
#  I figured out that you can momentarily get rid of the corrupted error by
#  cleaning the linux memory cache with the command sudo sh -c "sync; echo 1 > /proc/sys/vm/drop_caches".
#  This might indicate that the records are getting corrupted in memory or during the read from drivers.
