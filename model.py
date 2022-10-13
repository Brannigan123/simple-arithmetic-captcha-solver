#!/usr/bin/env python3

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten
import keras
import os
from const import CHECKPOINT_PATH, INPUT_SHAPE, OUTPUT_SIZE


def build_model():
    input = Input(INPUT_SHAPE)
    out = input

    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPool2D(pool_size=(2, 2))(out)

    out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPool2D(pool_size=(2, 2))(out)

    out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPool2D(pool_size=(2, 2))(out)

    out = Flatten()(out)
    out = Dense(1024,  activation='relu')(out)
    out = [
        Dense(OUTPUT_SIZE, name='digit1', activation='softmax')(out),
        Dense(OUTPUT_SIZE, name='digit2', activation='softmax')(out),
    ]
    model = Model(inputs=input, outputs=out)

    return model


def get_model():
    """
    If the model exists, load it, otherwise build it
    :return: The model is being returned.
    """
    if os.path.exists(CHECKPOINT_PATH):
        return keras.models.load_model(CHECKPOINT_PATH)
    else:
        return build_model()


if __name__ == '__main__':
    model = get_model()
    summary = model.summary()
    print(summary)
