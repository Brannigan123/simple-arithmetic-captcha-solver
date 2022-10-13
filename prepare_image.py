
import numpy as np
import cv2

import skimage.filters as filters

from const import INPUT_SHAPE


def prepare_image(img, preview=False):
    img = 255 - cv2.fastNlMeansDenoising(img)
    smooth = cv2.medianBlur(img, 5)
    smooth = cv2.GaussianBlur(smooth, (5, 5), 0)
    division = cv2.divide(img, smooth, scale=255)
    sharp = filters.unsharp_mask(
        division, radius=1.5, amount=1.5, channel_axis=False, preserve_range=False)
    sharp = (255*sharp).clip(0, 255).astype(np.uint8)

    if preview:
        cv2.imshow('preview', sharp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.array(sharp/255.0, dtype=np.float32).reshape(INPUT_SHAPE)
