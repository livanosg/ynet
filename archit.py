from contextlib import nullcontext
import tensorflow as tf
from tensorflow import name_scope, shape, slice, concat
from tensorflow.compat.v1 import variable_scope
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, ReLU
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization


def conv_bn_dr(input_tensor, filters, dropout=0., padding='same'):
    conv = Conv2D(filters=filters, kernel_size=(3, 3), padding=padding, kernel_initializer=glorot_normal)(input_tensor)
    conv = BatchNormalization()(conv)
    conv = ReLU(negative_slope=0.1, )(conv)
    conv = Dropout(rate=dropout, )(conv)
    return conv


def convolution_layer(input_tensor, filters, dropout=0., padding='same'):
    conv = conv_bn_dr(input_tensor=input_tensor, filters=filters, dropout=dropout, padding=padding)
    conv = conv_bn_dr(input_tensor=conv, filters=filters, dropout=dropout, padding=padding)
    return conv


def down_layer(input_tensor, filters, dropout=0.):
    connection = convolution_layer(input_tensor=input_tensor, filters=filters, dropout=dropout)
    output = MaxPooling2D(pool_size=(2, 2), strides=2)(connection)
    return output, connection


def trans_conv(input_tensor, filters, batch_norm=False, dropout=0.):
    up_conv = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same',
                              kernel_initializer=glorot_normal)(input_tensor)
    if batch_norm:
        up_conv = BatchNormalization()(up_conv)
    up_conv = ReLU(negative_slope=0.1)(up_conv)
    if dropout:
        up_conv = Dropout(dropout)(up_conv)
    return up_conv


def upconv_layer(input_tensor, connection, filters, dropout=0.):
    trans = trans_conv(input_tensor=input_tensor, filters=filters, dropout=dropout)
    up_conv = Concatenate()([connection, trans])
    return convolution_layer(input_tensor=up_conv, filters=filters // 2, dropout=dropout)


def merge_layer(input_tensor, connection, filters, dropout=0.):
    up_conv = Concatenate()([connection, input_tensor])
    return convolution_layer(input_tensor=up_conv, filters=filters, dropout=dropout)


# noinspection PyShadowingNames
def crop_con(up_layer, down_layer):
    up_shape = shape(up_layer)
    down_shape = shape(down_layer)
    # offsets for the top left corner of the crop
    offsets = [0, (down_shape[1] - up_shape[1]) // 2, (down_shape[2] - up_shape[2]) // 2, 0]
    size = [-1, up_shape[1], up_shape[2], 1]
    down_cropped = slice(down_layer, offsets, size)
    return concat([down_cropped, up_layer], -1)  # Concatenate at number of feature maps axis.


def ynet(input_tensor, params):
    dropout = params['dropout']
    classes = params['classes']
    if params['distribution']:
        device_1 = device_2 = nullcontext()
    else:
        device_1 = tf.device('/GPU:0')
        device_2 = tf.device('/GPU:1')

    with variable_scope('Model'):
        with device_1:
            with name_scope('Branch1_D1'):
                branch_1, connection_1 = down_layer(input_tensor=input_tensor, filters=32, dropout=dropout)
            with name_scope('Branch1_D2'):
                branch_1, connection_2 = down_layer(input_tensor=branch_1, filters=64, dropout=dropout)
            with name_scope('Branch1_D3'):
                branch_1, connection_3 = down_layer(input_tensor=branch_1, filters=128, dropout=dropout)
            with name_scope('Branch1_D4'):
                branch_1, connection_4 = down_layer(input_tensor=branch_1, filters=256, dropout=dropout)
            with name_scope('Branch1_Bridge'):
                bridge = convolution_layer(input_tensor=branch_1, filters=512, dropout=dropout)
            with name_scope('Branch1_U1'):
                branch_1_1 = upconv_layer(input_tensor=bridge, connection=connection_4, filters=256,
                                          dropout=dropout)
            with name_scope('Branch1_U2'):
                branch_1_2 = upconv_layer(input_tensor=branch_1_1, connection=connection_3, filters=128,
                                          dropout=dropout)
            with name_scope('Branch1_U3'):
                branch_1_3 = upconv_layer(input_tensor=branch_1_2, connection=connection_2, filters=64,
                                          dropout=dropout)
            with name_scope('Branch1_U4'):
                branch_1_4 = upconv_layer(input_tensor=branch_1_3, connection=connection_1, filters=32,
                                          dropout=dropout)
            with name_scope('Branch1_Output'):
                output_1 = Conv2D(filters=classes, kernel_size=1, padding='same')(branch_1_4)
        with device_2:
            with name_scope('Branch2_U1'):
                branch_2 = upconv_layer(input_tensor=bridge, connection=branch_1_1, filters=256, dropout=dropout)
            with name_scope('Branch2_U2'):
                branch_2 = upconv_layer(input_tensor=branch_2, connection=branch_1_2, filters=128, dropout=dropout)
            with name_scope('Branch2_U3'):
                branch_2 = upconv_layer(input_tensor=branch_2, connection=branch_1_3, filters=64, dropout=dropout)
            with name_scope('Branch2_U4'):
                branch_2 = upconv_layer(input_tensor=branch_2, connection=branch_1_4, filters=32, dropout=dropout)
            with name_scope('Branch2_Merger'):
                branch_2 = merge_layer(input_tensor=branch_2, connection=output_1, filters=16, dropout=dropout)
            with name_scope('Branch2_Output'):
                output_2 = Conv2D(filters=classes ** 2, kernel_size=1, padding='same',
                                  kernel_initializer=glorot_normal)(branch_2)
        return output_1, output_2


# noinspection PyShadowingNames
def incept_ynet(input_tensor, params):
    if params['modality'] in ('ALL', 'CT'):
        input_shape = [512, 512, 3]
    else:
        input_shape = [320, 320, 3]
    dropout = params['dropout']
    classes = params['classes']
    with variable_scope('Model'):
        with name_scope('Inception_v3'):
            inception_v3 = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                                          input_tensor=input_tensor,
                                                                          input_shape=input_shape,
                                                                          pooling=None)
            inception_v3.trainable = False
            lr_1 = inception_v3.get_layer('activation').output  # [None, 255, 255, 32]
            lr_4 = inception_v3.get_layer('activation_4').output  # [None, 124, 124, 192]
            mx_0 = inception_v3.get_layer('mixed0').output  # [None, 61, 61,  256]
            mx_1 = inception_v3.get_layer('mixed1').output  # [None, 61, 61,  288]
            mx_2 = inception_v3.get_layer('mixed2').output  # [None, 61, 61,  288]
            mx_3 = inception_v3.get_layer('mixed3').output  # [None, 30, 30,  768]
            mx_4 = inception_v3.get_layer('mixed4').output  # [None, 30, 30,  768]
            mx_5 = inception_v3.get_layer('mixed5').output  # [None, 30, 30,  768]
            mx_6 = inception_v3.get_layer('mixed6').output  # [None, 30, 30,  768]
            mx_7 = inception_v3.get_layer('mixed7').output  # [None, 30, 30,  768]
            mx_8 = inception_v3.get_layer('mixed8').output  # [None, 14, 14, 1280]
            mx_9 = inception_v3.get_layer('mixed9').output  # [None, 14, 14, 2048]
            mx_10 = inception_v3.get_layer('mixed10').output  # [None,  14,  14, 2048]
            concat_1 = tf.keras.layers.concatenate([mx_8, mx_9, mx_10])  # [None,  14,  14, 5376]
        with name_scope('Up1_1'):
            up_0 = Conv2DTranspose(256, kernel_size=2, strides=1, activation='relu', padding='valid')(concat_1)  # [None, 15, 15, 256]
            up_0 = BatchNormalization()(up_0)
            up_0 = ReLU(negative_slope=0.1, )(up_0)
            up_0 = Dropout(rate=dropout)(up_0)
            up_0 = Conv2DTranspose(256, kernel_size=2, strides=2, activation='relu', padding='valid')(up_0)  # [None, 30, 30, 256]
            up_0 = BatchNormalization()(up_0)
            up_0 = ReLU(negative_slope=0.1, )(up_0)
            up_0 = Dropout(rate=dropout)(up_0)
            up_0 = convolution_layer(up_0, filters=256, dropout=dropout, padding='same')  # [None, 30, 30, 256]
        with name_scope('Up1_2'):
            concat_2 = tf.keras.layers.concatenate([up_0, mx_3, mx_4, mx_5, mx_6, mx_7])  # [None, 30, 30, 4096]

            up_1 = Conv2DTranspose(256, kernel_size=2, strides=2, activation='relu', padding='valid')(concat_2)  # [None, 60, 60, 256]
            up_1 = BatchNormalization()(up_1)
            up_1 = ReLU(negative_slope=0.1, )(up_1)
            up_1 = Dropout(rate=dropout)(up_1)
            up_1 = Conv2DTranspose(128, kernel_size=2, strides=1, activation='relu', padding='valid')(up_1)  # [None, 61, 61, 128]
            up_1 = BatchNormalization()(up_1)
            up_1 = ReLU(negative_slope=0.1, )(up_1)
            up_1 = Dropout(rate=dropout)(up_1)
            up_1 = convolution_layer(up_1, filters=128, dropout=dropout, padding='same')  # [None, 61, 61, 128]

        with name_scope('Up1_3'):
            concat_3 = tf.keras.layers.concatenate([up_1, mx_0, mx_1, mx_2])  # [None, 61, 61, 960]

            up_2 = Conv2DTranspose(64, kernel_size=2, strides=1, activation='relu', padding='valid')(concat_3)  # [None, 62, 62, 64]
            up_2 = BatchNormalization()(up_2)
            up_2 = ReLU(negative_slope=0.1, )(up_2)
            up_2 = Dropout(rate=dropout)(up_2)

            up_2 = Conv2DTranspose(64, kernel_size=2, strides=2, activation='relu', padding='valid')(up_2)  # [None, 124, 124, 64]
            up_2 = BatchNormalization()(up_2)
            up_2 = ReLU(negative_slope=0.1, )(up_2)
            up_2 = Dropout(rate=dropout)(up_2)
            up_2 = convolution_layer(up_2, filters=64, dropout=dropout, padding='same')  # [None, 124, 124, 64]

        with name_scope('Up1_4'):
            concat_4 = tf.keras.layers.concatenate([up_2, lr_4])  # [None, 124, 124, 256]

            up_3 = Conv2DTranspose(64, kernel_size=5, strides=1, activation='relu', padding='valid')(concat_4)  # [None, 128, 128, 64]
            up_3 = BatchNormalization()(up_3)
            up_3 = ReLU(negative_slope=0.1, )(up_3)
            up_3 = Dropout(rate=dropout)(up_3)

            up_3 = Conv2DTranspose(64, kernel_size=2, strides=2, activation='relu', padding='valid')(up_3)  # [None, 256, 256, 128]
            up_3 = BatchNormalization()(up_3)
            up_3 = ReLU(negative_slope=0.1, )(up_3)
            up_3 = Dropout(rate=dropout)(up_3)

            up_3 = convolution_layer(up_3, filters=64, dropout=dropout, padding='same')  # [None, 256, 256, 64]
            up_3 = Conv2D(filters=32, kernel_size=2, padding='valid', kernel_initializer=glorot_normal)(up_3)  # [None, 255, 255, 32]
            up_3 = BatchNormalization()(up_3) # [None, 255, 255,  32]
            up_3 = ReLU(negative_slope=0.1, )(up_3)  # [None, 255, 255, 32]
            up_3 = Dropout(rate=dropout)(up_3)  # [None, 255, 255, 32]

        with name_scope('Up1_5'):
            concat_5 = tf.keras.layers.concatenate([up_3, lr_1])  # [None, 255, 255, 64]

            up_4 = Conv2DTranspose(32, kernel_size=2, strides=1, activation='relu', padding='valid')(concat_5)  # [None, 256, 256, 32]
            up_4 = BatchNormalization()(up_4)
            up_4 = ReLU(negative_slope=0.1, )(up_4)
            up_4 = Dropout(rate=dropout)(up_4)

            up_4 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu')(up_4)  # [None, 512, 512, 32]
            up_4 = BatchNormalization()(up_4)
            up_4 = ReLU(negative_slope=0.1, )(up_4)
            up_4 = Dropout(rate=dropout)(up_4)
            up_4 = convolution_layer(up_4, filters=32, dropout=dropout, padding='same')  # [None, 512, 512, 32]
        with name_scope('Up1_Output'):
            output_1 = Conv2D(filters=2, kernel_size=1, padding='same')(up_4)  # [None, 512, 512, 2]

        with name_scope('Up2_1'):
            up_2_0 = Conv2DTranspose(256, kernel_size=2, strides=1, activation='relu', padding='valid')(concat_1)  # [None,  30,  30,  128]
            up_2_0 = BatchNormalization()(up_2_0)
            up_2_0 = ReLU(negative_slope=0.1, )(up_2_0)
            up_2_0 = Dropout(rate=dropout)(up_2_0)

            up_2_0 = Conv2DTranspose(256, kernel_size=2, strides=2, activation='relu', padding='valid')(up_2_0)  # [None,  30,  30,  128]
            up_2_0 = BatchNormalization()(up_2_0)
            up_2_0 = ReLU(negative_slope=0.1, )(up_2_0)
            up_2_0 = Dropout(rate=dropout)(up_2_0)

            up_2_0 = convolution_layer(up_2_0, filters=256, dropout=dropout, padding='same')
        with name_scope('Up2_2'):
            concat_2_1 = tf.keras.layers.concatenate([up_2_0, concat_2])  # [None,  30,  30, 3968]

            up_2_1 = Conv2DTranspose(256, kernel_size=2, strides=2, activation='relu', padding='valid')(concat_2_1)  # [None,  61,  61,  128]
            up_2_1 = BatchNormalization()(up_2_1)
            up_2_1 = ReLU(negative_slope=0.1, )(up_2_1)
            up_2_1 = Dropout(rate=dropout)(up_2_1)

            up_2_1 = Conv2DTranspose(128, kernel_size=2, strides=1, activation='relu', padding='valid')(up_2_1)  # [None,  61,  61,  128]
            up_2_1 = BatchNormalization()(up_2_1)
            up_2_1 = ReLU(negative_slope=0.1, )(up_2_1)
            up_2_1 = Dropout(rate=dropout)(up_2_1)

            up_2_1 = convolution_layer(up_2_1, filters=128, dropout=dropout, padding='same')
        with name_scope('Up2_3'):
            concat_2_2 = tf.keras.layers.concatenate([up_2_1, concat_3])  # [None,  61,  61,  960]

            up_2_2 = Conv2DTranspose(64, kernel_size=2, strides=1, activation='relu', padding='valid')(concat_2_2)  # [None, 122, 122,   64]
            up_2_2 = BatchNormalization()(up_2_2)
            up_2_2 = ReLU(negative_slope=0.1, )(up_2_2)
            up_2_2 = Dropout(rate=dropout)(up_2_2)

            up_2_2 = Conv2DTranspose(64, kernel_size=2, strides=2, activation='relu', padding='valid')(up_2_2)  # [None, 124, 124,   64]
            up_2_2 = BatchNormalization()(up_2_2)
            up_2_2 = ReLU(negative_slope=0.1, )(up_2_2)
            up_2_2 = Dropout(rate=dropout)(up_2_2)

            up_2_2 = convolution_layer(up_2_2, filters=64, dropout=dropout, padding='same')
        with name_scope('Up2_4'):
            concat_2_3 = tf.keras.layers.concatenate([up_2_2, concat_4])  # [None, 124, 124,  256]

            up_2_3 = Conv2DTranspose(64, kernel_size=5, strides=1, activation='relu', padding='valid')(concat_2_3)  # [None,  128, 128, 64]
            up_2_3 = BatchNormalization()(up_2_3)
            up_2_3 = ReLU(negative_slope=0.1, )(up_2_3)
            up_2_3 = Dropout(rate=dropout)(up_2_3)

            up_2_3 = Conv2DTranspose(64, kernel_size=2, strides=2, activation='relu', padding='valid')(up_2_3)  # [None,  30,  30,  128]
            up_2_3 = BatchNormalization()(up_2_3)
            up_2_3 = ReLU(negative_slope=0.1, )(up_2_3)
            up_2_3 = Dropout(rate=dropout)(up_2_3)

            up_2_3 = convolution_layer(up_2_3, filters=64, dropout=dropout, padding='same')
            up_2_3 = Conv2D(filters=32, kernel_size=2, padding='valid', kernel_initializer=glorot_normal)(up_2_3)
            up_2_3 = BatchNormalization()(up_2_3)
            up_2_3 = ReLU(negative_slope=0.1, )(up_2_3)
            up_2_3 = Dropout(rate=dropout)(up_2_3)
        with name_scope('Up2_5'):
            concat_2_4 = tf.keras.layers.concatenate([up_2_3, concat_5])
            up_2_4 = Conv2DTranspose(32, kernel_size=2, strides=1, activation='relu', padding='valid')(concat_2_4)  # [None, 122, 122,   64]
            up_2_4 = BatchNormalization()(up_2_4)
            up_2_4 = ReLU(negative_slope=0.1, )(up_2_4)
            up_2_4 = Dropout(rate=dropout)(up_2_4)
            up_2_4 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu')(up_2_4)  # [None, 512, 512,   16]
            up_2_4 = BatchNormalization()(up_2_4)
            up_2_4 = ReLU(negative_slope=0.1, )(up_2_4)
            up_2_4 = Dropout(rate=dropout)(up_2_4)
            up_2_4 = convolution_layer(up_2_4, filters=32, dropout=dropout, padding='same')
        with name_scope('Up2_Output'):
            up_2_4 = tf.keras.layers.concatenate([up_2_4, output_1])  # [None,  30,  30, 3968]
            up_2_4 = convolution_layer(up_2_4, filters=32, dropout=dropout, padding='same')
            output_2 = Conv2D(filters=classes ** 2, kernel_size=1, padding='same', kernel_initializer=glorot_normal)(up_2_4)

    return output_1, output_2
