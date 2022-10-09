from typing import List
from const import INPUT_SHAPE
import numpy as np
import cv2


def pad(img: cv2.Mat):
    """
    It adds a white border to the image
    
    :param img: The image to be padded
    :type img: cv2.Mat
    :return: the image with a border around it.
    """
    top, left = int(0.05 * img.shape[0]), int(0.15 * img.shape[1])
    bottom, right = top, left
    return cv2.copyMakeBorder(img, top, bottom, left+1,
                              right, cv2.BORDER_CONSTANT, None, 255)


def thresh(img: cv2.Mat):
    """
    It takes an image and returns a thresholded version of it
    
    :param img: The image to be thresholded
    :type img: cv2.Mat
    :return: A binary image.
    """
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)


def erode(img: cv2.Mat):
    """
    Erode the image by dilating it with a 2x1 rectangle.
    
    :param img: The image to be processed
    :type img: cv2.Mat
    :return: The image is being eroded.
    """
    return cv2.dilate(img, np.ones((2, 1), np.uint8), iterations=2)


def preprocess(img: cv2.Mat):
    """
    Takes an image, threshold it, and then erode it
    
    :param img: the image to be processed
    :type img: cv2.Mat
    :return: The eroded image.
    """
    threshold = thresh(img)
    eroded = erode(threshold)
    return eroded


def split(img: cv2.Mat) -> List[cv2.Mat]:
    """
    It takes an image, splits it into three equal parts, pads each part to the desired input shape, and
    returns the three parts
    
    :param img: cv2.Mat - the image to split
    :type img: cv2.Mat
    :return: A list of 3 images, each of which is a resized version of the original image.
    """
    width = len(img[0])//3
    return [
        cv2.resize(
            pad(img[:, width*i:width*(i+1)]), INPUT_SHAPE
        )
        for i in range(3)
    ]


def extract_char_regions(path: str):
    """
    It reads the image, preprocesses it, and splits it into individual characters
    
    :param path: the path to the image
    :type path: str
    :return: A list of images of the characters
    """
    img = cv2.imread(path, 0)
    img = preprocess(img)
    return split(img)
