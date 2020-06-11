import Augmentor as augmentor
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add
from tensorflow.keras import activations


def conv_block(input_, filters, kernels, strides=(2, 2), alpha=0.2):

    elu = lambda x: activations.elu(x, alpha=alpha)

    x = input_

    for filters_count, kernel in zip(filters, kernels):
        x = Conv2D(filters=filters_count, kernel_size=kernel, strides=strides, padding='same',
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation(activation=elu)(x)

    shortcut = Conv2D(filters=filters.copy().pop(), kernel_size=kernels.copy().pop(), strides=strides, padding='same',
                      kernel_initializer='he_uniform')(input_)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation(activation=elu)(shortcut)

    conv_block_res = add([x, shortcut])
    conv_block_res_active = Activation(activation=elu)(conv_block_res)

    return conv_block_res_active


def ident_block(input_, filters, kernels, strides=(1, 1), alpha=0.2):

    elu = lambda x: activations.elu(x, alpha=alpha)

    x = input_

    for filters_count, kernel in zip(filters, kernels):
        x = Conv2D(filters=filters_count, kernel_size=kernel, strides=strides, padding='same',
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation(activation=elu)(x)

    ident_block_res = add([x, input_])
    ident_block_res = Activation(activation=elu)(ident_block_res)

    return ident_block_res


def convert(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def data_augment(image, target_dim):
    image = convert(image)
    if target_dim[2] == 1:
        image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(images=image, size=(target_dim[0], target_dim[1]))
    return image
