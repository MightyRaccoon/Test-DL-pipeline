import logging
import os
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, losses, optimizers, preprocessing, metrics, initializers, activations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.utils import plot_model
import click

from util_functions import conv_block, ident_block, data_augment



logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level='INFO'
)

log = logging.getLogger(__name__)

@click.command()
@click.option('--batch-size', type=int, help='Batch size for train set', default=16)
@click.option('--epochs-count', type=int, help='Epochs count for model fit', default=10)
@click.option('--learning-rate', type=float, help='Learning rate for optimizer', default=0.001)
@click.option('--alpha', type=float, help='Parameter for leackyReLU function', default=0.0)
@click.option('--classes-rate', type=float, help='Rate of classes from place_365', default=0.05)
@click.option('--data-dir', type=str, help='Directory with data')
def main(batch_size, epochs_count, learning_rate, alpha, classes_rate, data_dir):

    fc_dropout_rate = 0.5
    conv_dropout_rate = 0.7
    l2_reg = 0.01

    target_size = [128, 128, 1]

    elu = lambda x: activations.elu(x, alpha=alpha)
    #augment = lambda image: data_augment(image, target_size)

    log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    tf_board_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    train_dir = '/'.join([data_dir, 'train_set'])
    val_dir = '/'.join([data_dir, 'val_set'])
    test_dir = '/'.join([data_dir, 'test_set'])
    all_classes = np.array(os.listdir(train_dir))
    classes_list = all_classes[np.random.uniform(size=len(all_classes)) <= classes_rate].tolist()
    log.info(f'Model will for classes: {classes_list}')

    log.info('Data generator creation')

    data_generator = ImageDataGenerator(
        rescale=1./255
    )
    train_generator = data_generator.flow_from_directory(
        directory=train_dir,
        target_size=(256, 256),
        classes=classes_list,
        batch_size=batch_size
    )
    val_generator = data_generator.flow_from_directory(
        directory=val_dir,
        target_size=(256, 256),
        classes=classes_list
    )

    log.info('Model Creation')

    model_input = Input(shape=(256, 256, 3))

    x = ZeroPadding2D(padding=(3, 3))(model_input)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=elu)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Conv block 1
    filters_1 = [64, 64, 256]
    kernels_1 = [(1, 1), (3, 3), (1, 1)]

    x = conv_block(x, filters_1, kernels_1, (1, 1), alpha=alpha)
    x = ident_block(x, filters_1, kernels_1, (1, 1), alpha=alpha)
    x = ident_block(x, filters_1, kernels_1, (1, 1), alpha=alpha)
    x = Dropout(rate=conv_dropout_rate)(x)

    # Conv block 2
    filters_2 = [128, 128, 512]
    kernels_2 = [(1, 1), (3, 3), (1, 1)]

    x = conv_block(x, filters_2, kernels_2, (1, 1), alpha=alpha)
    x = ident_block(x, filters_2, kernels_2, (1, 1), alpha=alpha)
    x = ident_block(x, filters_2, kernels_2, (1, 1), alpha=alpha)
    x = Dropout(rate=conv_dropout_rate)(x)

    # Conv block 3
    filters_3 = [256, 256, 1024]
    kernels_3 = [(1, 1), (3, 3), (1, 1)]

    x = conv_block(x, filters_3, kernels_3, (1, 1), alpha=alpha)
    x = ident_block(x, filters_3, kernels_3, (1, 1), alpha=alpha)
    x = ident_block(x, filters_3, kernels_3, (1, 1), alpha=alpha)
    x = Dropout(rate=conv_dropout_rate)(x)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    x = Dense(units=1000, activation=elu, kernel_initializer='he_uniform')(x)
    x = Dropout(rate=fc_dropout_rate)(x)
    model_output = Dense(units=len(classes_list), activation='softmax')(x)

    model = models.Model(model_input, model_output)

    model.summary()
    plot_model(model, show_shapes=True)

    log.info('Model compiling')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=[metrics.Accuracy(), metrics.Precision(), metrics.Recall(), metrics.AUC()]
    )

    log.info('Model fitting')
    model.fit(
        x=train_generator,
        validation_data=val_generator,
        epochs=epochs_count,
        callbacks=[tf_board_callback]
    )

    log.info('Model testing')
    test_data_generator = ImageDataGenerator(
        rescale=1. / 255
    )
    test_generator = test_data_generator.flow_from_directory(
        directory=test_dir,
        target_size=(256, 256),
        classes=classes_list
    )

    model_test_results = model.evaluate(x=test_generator)
    log.info(f'Model loss: {model_test_results[0]}, accuracy: {model_test_results[1]}. AUC: {model_test_results[4]}, Presicion: {model_test_results[4]}, Recall: {model_test_results[3]}')


if __name__ == '__main__':
    main()
