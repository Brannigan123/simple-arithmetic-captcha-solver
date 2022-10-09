#!/usr/bin/env python3

from keras.utils import to_categorical
from const import LABELS, MODEL_PATH, TRAIN_DATA_PATH
from model import get_model
import numpy as np
import keras
import glob
import cv2


def load_train_data():
    """
    It loads all the images in the training data directory, converts them to grayscale, and returns them
    as a numpy array
    :return: the training data and the training labels.
    """
    x_train = []
    y_train = []
    for dir in glob.glob(f'{TRAIN_DATA_PATH}/**'):
        label = dir.split('/')[-1]
        label_idx = LABELS.index(label)
        for imgfile in glob.glob(f'{dir}/*'):
            x_train.append(
                np.array(cv2.imread(imgfile, 0), dtype=np.float32))
            y_train.append(np.array(label_idx, dtype=np.int8))
    return np.array(x_train)/255, to_categorical(np.array(y_train))


def train(x_train, y_train, x_test, y_test, batch_size=16, epochs=50):
    """
    We create a model, compile it, and then train it.

    :param x_train: The training data
    :param y_train: The training labels
    :param x_test: The test data
    :param y_test: The test labels
    :param batch_size: The number of samples per gradient update, defaults to 16 (optional)
    :param epochs: The number of times the model will cycle through the data, defaults to 30 (optional)
    :return: The model.fit() function returns a History object. Its History.history attribute is a
    record of training loss values and metrics values at successive epochs, as well as validation loss
    values and validation metrics values (if applicable).
    """
    model = get_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True)
    return model.fit(x_train, y_train, validation_data=(
        x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=1, use_multiprocessing=True)


if __name__ == '__main__':
    x, y = load_train_data()
    train(x, y, x, y)
