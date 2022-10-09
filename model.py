#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
import keras
import os
from const import INPUT_SHAPE, MODEL_PATH, OUTPUT_SHAPE


def build_model():
    """
    Builds a convolutional neural network with three convolutional layers, each followed by a
    max pooling layer, and a dropout layer. The convolutional layers have 8, 16, and 32 filters,
    respectively. The max pooling layers have a pool size of (2, 2). The dropout layers have a dropout
    rate of 0.2, 0.2, and 0.1, respectively. The convolutional layers have a kernel size of 7, 5, and 3,
    respectively. The convolutional layers have a stride of 1. The convolutional layers have a padding
    of 'same'. The convolutional layers have a ReLU activation function. The dense layer has 128
    neurons. The output layer has a softmax activation function
    :return: A model with the following layers:
        - Conv2D
        - MaxPool2D
        - Dropout
        - Conv2D
        - MaxPool2D
        - Dropout
        - Conv2D
        - MaxPool2D
        - Dropout
        - Flatten
        - Dense
        - Dense
    """
    model = Sequential()

    model.add(Conv2D(filters=8, kernel_size=7, strides=1, padding='same',
                     activation='relu', input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 1), name='conv1'))
    model.add(MaxPool2D(pool_size=(2, 2,), name='pool1'))
    model.add(Dropout(0.2, name='dropout1'))

    model.add(Conv2D(filters=16, kernel_size=5, strides=1,
                     padding='same', activation='relu', name='conv2'))
    model.add(MaxPool2D(pool_size=(2, 2), name='pool2'))
    model.add(Dropout(0.2, name='dropout2'))

    model.add(Conv2D(filters=32, kernel_size=3, strides=1,
                     padding='same', activation='relu', name='conv3'))
    model.add(MaxPool2D(pool_size=(2, 2), name='pool3'))
    model.add(Dropout(0.1, name='dropout3'))

    model.add(Flatten(name='flat'))
    model.add(Dense(128, activation='relu', name='dense'))
    model.add(Dense(OUTPUT_SHAPE, activation='softmax', name='softmax'))

    return model


def get_model():
    """
    If the model exists, load it, otherwise build it
    :return: The model is being returned.
    """
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    else:
        return build_model()


if __name__ == '__main__':
    model = get_model()
    summary = model.summary()
    print(summary)
