#!/usr/bin/env python3

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from model import get_model
from const import INPUT_SHAPE, LABELS, TEST_DATA_PATH
from prepare_image import prepare_image
from typing import List
import glob
import cv2
import sys
import numpy as np



def solve_exression(tokens: List[str]):
    """
    It takes a list of strings, joins them with spaces, and evaluates the resulting expression

    :param tokens: List[str]
    :type tokens: List[str]
    :return: The result of the expression
    """
    return eval(' + '.join(tokens))


def infer(model, fname, img, preview=False):
    if preview:
        cv2.imshow('preview', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img = prepare_image(img)

    probs = model.predict(
        img.reshape(1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), use_multiprocessing=True)
    preds = [LABELS[int(np.argmax(s_probs))] for s_probs in probs]
    result = solve_exression(preds)

    print(
        f'File: {fname}\nCharacter Predictions: {preds}\nResult: {result}\n')

    if preview:
        cv2.imshow('processed', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

   

if __name__ == '__main__':
    argv = sys.argv
    argn = len(argv) - 1
    paths = glob.glob(f'{TEST_DATA_PATH}/*') if argn == 0 else argv[1:]

    model = get_model()

    for path in paths:
        fname = path.split('/')[-1]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        infer(model, fname, img, preview=True)
