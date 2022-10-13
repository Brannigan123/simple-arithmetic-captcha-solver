#!/usr/bin/env python3

import os 
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import glob
import cv2
import re
import random

from prepare_image import prepare_image
from const import LABELS, CHECKPOINT_PATH, TRAIN_DATA_PATH
from model import get_model

def load_train_data():
    """
    It loads all the images in the training data directory, and returns them
    as a numpy array
    :return: the training data and the training labels.
    """
    x_train = []
    y1_train = []
    y2_train = []
    files = glob.glob(f'{TRAIN_DATA_PATH}/*')
    random.shuffle(files)
    for path in files:
        expr = re.split('\[|\]', path)[-2].split(' ')
        label1_idx, label2_idx = LABELS.index(expr[0]), LABELS.index(expr[2])
        x_train.append(prepare_image(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))
        y1_train.append(np.array(label1_idx, dtype=np.int8))
        y2_train.append(np.array(label2_idx, dtype=np.int8))
    return np.array(x_train), [to_categorical(np.array(y1_train)), to_categorical(np.array(y2_train))]


def train(x_train, y_train,  batch_size=32, epochs=50):
    """
    We create a model, compile it, and then train it.

    :param x_train: The training data
    :param y_train: The training labels
    :param batch_size: The number of samples per gradient update, defaults to 64 (optional)
    :param epochs: The number of times the model will cycle through the data, defaults to 50 (optional)
    :return: The model.fit() function returns a History object. Its History.history attribute is a
    record of training loss values and metrics values at successive epochs, as well as validation loss
    values and validation metrics values (if applicable).
    """
    model = get_model()
    model.compile(
        optimizer='adam',
        loss={
            'digit1': 'categorical_crossentropy',
            'digit2': 'categorical_crossentropy'
        },
        metrics={
            'digit1': 'accuracy',
            'digit2': 'accuracy'
        }
    )

    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=3,
            verbose=1,
            factor=0.5,
            min_lr=0.00001
        ),
        ModelCheckpoint(
            CHECKPOINT_PATH, save_best_only=True, monitor='val_loss'
        )
    ]

    return model.fit(
        x_train, [y_train[0], y_train[1]],
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=True
    )


if __name__ == '__main__':
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    x, y = load_train_data()
    train(x, y, batch_size=256)
