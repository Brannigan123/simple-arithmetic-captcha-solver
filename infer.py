#!/usr/bin/env python3

from typing import List
import glob
import cv2
import sys
import numpy as np

from prepare_image import prepare_image
from const import INPUT_SHAPE, LABELS, TEST_DATA_PATH
from model import get_model


def solve_exression(tokens: List[str]):
    """
    It takes a list of strings, joins them with spaces, and evaluates the resulting expression

    :param tokens: List[str]
    :type tokens: List[str]
    :return: The result of the expression
    """
    return eval(' + '.join(tokens))


def infer(model, fname, img):
    probs = model.predict(
        img.reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), use_multiprocessing=True)
    preds = [LABELS[int(np.argmax(s_probs))] for s_probs in probs]
    result = solve_exression(preds)
    print(
        f'File: {fname}\nCharacter Predictions: {preds}\nResult: {result}\n')


if __name__ == '__main__':
    argv = sys.argv
    argn = len(argv) - 1
    paths = glob.glob(f'{TEST_DATA_PATH}/*') if argn == 0 else argv[1:]

    model = get_model()

    for path in paths:
        fname = path.split('/')[-1]
        img = prepare_image(cv2.imread(
            path, cv2.IMREAD_GRAYSCALE), preview=True)
        infer(model, fname, img)
