from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, ReLU, Softmax
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow import name_scope, shape, slice, concat
from tensorflow import variable_scope


def conv_bn_dr(input_tensor, filters, dropout=0.):
    conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
    conv = BatchNormalization()(conv)
    conv = ReLU(negative_slope=0.1, )(conv)
    conv = Dropout(rate=dropout, )(conv)
    return conv


def convolution_layer(input_tensor, filters, dropout=0.):
    conv = conv_bn_dr(input_tensor=input_tensor, filters=filters, dropout=dropout)
    conv = conv_bn_dr(input_tensor=conv, filters=filters, dropout=dropout)
    return conv


def down_layer(input_tensor, filters, dropout=0.):
    connection = convolution_layer(input_tensor=input_tensor, filters=filters, dropout=dropout)
    output = MaxPooling2D(pool_size=(2, 2), strides=2)(connection)
    return output, connection


def trans_conv(input_tensor, filters, batch_norm=False, dropout=0.):
    up_conv = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same')(input_tensor)
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
    with variable_scope('Model'):
        with variable_scope('Branch_1'):
            with name_scope('Down_1'):
                output, connection_1 = down_layer(input_tensor=input_tensor, filters=64, dropout=dropout)
            with name_scope('Down_2'):
                output, connection_2 = down_layer(input_tensor=output, filters=128, dropout=dropout)
            with name_scope('Down_3'):
                output, connection_3 = down_layer(input_tensor=output, filters=256, dropout=dropout)
            with name_scope('Down_4'):
                output, connection_4 = down_layer(input_tensor=output, filters=512, dropout=dropout)
            with name_scope('Bridge'):
                output = convolution_layer(input_tensor=output, filters=1024, dropout=dropout)
            with name_scope('Up1_1'):
                branch_1_1 = upconv_layer(input_tensor=output, connection=connection_4, filters=512, dropout=dropout)
            with name_scope('Up1_2'):
                branch_1_2 = upconv_layer(input_tensor=branch_1_1, connection=connection_3, filters=256, dropout=dropout)
            with name_scope('Up1_3'):
                branch_1_3 = upconv_layer(input_tensor=branch_1_2, connection=connection_2, filters=128, dropout=dropout)
            with name_scope('Up1_4'):
                branch_1_4 = upconv_layer(input_tensor=branch_1_3, connection=connection_1, filters=64, dropout=dropout)
            with name_scope('Output1'):
                logits = Conv2D(filters=classes, kernel_size=1, padding='same')(branch_1_4)
                predictions1 = Softmax(axis=-1)(logits)
        with variable_scope('Branch_2'):
            with name_scope('Up2_1'):
                branch_2 = upconv_layer(input_tensor=output, connection=branch_1_1, filters=512, dropout=dropout)
            with name_scope('Up2_2'):
                branch_2 = upconv_layer(input_tensor=branch_2, connection=branch_1_2, filters=256, dropout=dropout)
            with name_scope('Up2_3'):
                branch_2 = upconv_layer(input_tensor=branch_2, connection=branch_1_3, filters=128, dropout=dropout)
            with name_scope('Up2_4'):
                branch_2 = upconv_layer(input_tensor=branch_2, connection=branch_1_4, filters=64, dropout=dropout)
            with name_scope('Output2'):
                branch_2 = Conv2D(filters=classes ** 2 - classes + 1, kernel_size=1, padding='same')(branch_2)
                predictions2 = Softmax(axis=-1)(branch_2)
    return predictions1, predictions2
