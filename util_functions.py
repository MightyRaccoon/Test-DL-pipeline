from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add


def conv_block(input_, filters, kernels, strides=(2,2)):

    x = Conv2D(filters=filters[0], kernel_size=kernels[0], strides=(1, 1), padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters[1], kernel_size=kernels[1], strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters[2], kernel_size=kernels[2], strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    shortcut = Conv2D(filters=filters[2], kernel_size=kernels[2], strides=strides, padding='same')(input_)
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation(activation='relu')(shortcut)

    conv_block_res = add([x, shortcut])
    conv_block_res_active = Activation(activation='relu')(conv_block_res)

    return conv_block_res_active


def ident_block(input_, filters, kernels, strides=(1, 1)):

    x = Conv2D(filters=filters[0], kernel_size=kernels[0], strides=strides, padding='same')(input_)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters[1], kernel_size=kernels[1], strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters[2], kernel_size=kernels[2], strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    conv_block_res = add([x, input_])
    conv_block_res_active = Activation(activation='relu')(conv_block_res)

    return conv_block_res_active