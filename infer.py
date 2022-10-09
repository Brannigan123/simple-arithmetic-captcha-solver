#!/usr/bin/env python3

from typing import List
from const import INPUT_SHAPE, LABELS, TEST_DATA_PATH
from model import get_model
from preprocess_captcha import extract_char_regions
import glob
import numpy as np


def load_test_data():
    """
    It loads the test data, extracts the characters from each image, and normalizes the pixel values to
    be between 0 and 1
    :return: A dictionary of the test data.
    """
    x_test = {}
    for captcha_path in glob.glob(f'{TEST_DATA_PATH}/**'):
        filename = captcha_path.split('/')[-1]
        x_test[filename] = np.array([
            np.array(r, dtype=np.float32) for r in extract_char_regions(captcha_path)
        ])/255
    return x_test


def solve_exression(tokens: List[str]):
    """
    It takes a list of strings, joins them with spaces, and evaluates the resulting expression

    :param tokens: List[str]
    :type tokens: List[str]
    :return: The result of the expression
    """
    return eval(' '.join(tokens))


if __name__ == '__main__':
    model = get_model()
    x = load_test_data()
    for filename, img in x.items():
        probs = model.predict(
            img.reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1), use_multiprocessing=True)
        preds = [LABELS[int(np.argmax(s_probs))] for s_probs in probs]
        result = solve_exression(preds)
        print(
            f'File: {filename}\nCharacter Predictions: {preds}\nResult: {result}\n')
