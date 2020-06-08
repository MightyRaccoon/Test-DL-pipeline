import logging
import os
import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, losses, optimizers, preprocessing, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.utils import plot_model

from util_functions import conv_block, ident_block

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level='INFO'
)

log = logging.getLogger(__name__)


def main():

    batch_size = 32
    epoch_count = 100
    learning_rate = 0.01
    fc_dropout_rate = 0.5

    log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    tf_board_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    classes_list = [
        '_v_veterinarians_office', '_m_mountain', '_c_carrousel',
        '_s_stage_outdoor', '_l_lake_natural', '_p_pond',
        '_s_swimming_pool_indoor', '_m_museum_indoor', '_e_embassy',
        '_w_windmill', '_a_assembly_line', '_b_bridge',
        '_r_recreation_room', '_f_flea_market_indoor', '_p_pharmacy',
        '_t_topiary_garden', '_p_pizzeria'
    ]

    log.info('Data generator creation')

    data_generator = ImageDataGenerator()
    train_generator = data_generator.flow_from_directory(
        directory='/home/mightyracoon/datasets/place_365/train_set',
        target_size=(256, 256),
        classes=classes_list,
        batch_size=batch_size
    )
    val_generator = data_generator.flow_from_directory(
        directory='/home/mightyracoon/datasets/place_365/val_set',
        target_size=(256, 256),
        classes=classes_list
    )

    log.info('Model Creation')

    input = Input(shape=(256, 256, 3))

    x = ZeroPadding2D(padding=(3, 3))(input)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Conv block 1
    filters_1 = [64, 64, 256]
    kernels_1 = [(1, 1), (3, 3), (1, 1)]

    x = conv_block(x, filters_1, kernels_1, (1, 1))
    x = ident_block(x, filters_1, kernels_1)
    x = ident_block(x, filters_1, kernels_1)

    # Conv block 2
    filters_2 = [126, 128, 512]
    kernels_2 = [(1, 1), (3, 3), (1, 1)]

    x = conv_block(x, filters_2, kernels_2)
    x = ident_block(x, filters_2, kernels_2)
    x = ident_block(x, filters_2, kernels_2)

    # Conv block 3
    filters_3 = [256, 256, 1024]
    kernels_3 = [(1, 1), (3, 3), (1, 1)]

    x = conv_block(x, filters_3, kernels_3)
    x = ident_block(x, filters_3, kernels_3)
    x = ident_block(x, filters_3, kernels_3)
    x = ident_block(x, filters_3, kernels_3)
    x = ident_block(x, filters_3, kernels_3)
    x = ident_block(x, filters_3, kernels_3)

    #  Conv block 4
    filters_4 = [512, 512, 2048]
    kernels_4 = [(1, 1), (3, 3), (1, 1)]

    x = conv_block(x, filters_4, kernels_4)
    x = ident_block(x, filters_4, kernels_4)
    x = ident_block(x, filters_4, kernels_4)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    x = Dense(units=1000, activation='relu')(x)
    x = Dropout(rate=fc_dropout_rate)(x)
    output = Dense(units=len(classes_list), activation='softmax')(x)

    model = models.Model(input, output)

    model.summary()
    plot_model(model, show_shapes=True)

    log.info('Model compiling')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=[metrics.Accuracy(), metrics.AUC()]
    )

    log.info('Model fitting')
    model.fit(
        x=train_generator,
        validation_data=val_generator,
        epochs=epoch_count,
        callbacks=[tf_board_callback]
    )

    log.info('Model testing')
    test_generator = data_generator.flow_from_directory(
        directory='/home/mightyracoon/datasets/place_365/test_set',
        target_size=(256, 256),
        classes=classes_list
    )

    model_test_results = model.evaluate(x=test_generator)
    log.info(f'Model loss: {model_test_results[0]}, accuracy: {model_test_results[1]}. AUC: {model_test_results[2]}')


if __name__ == '__main__':
    main()
