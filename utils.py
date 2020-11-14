import os

import numpy as np
import cv2
from os.path import exists

def load_image(image_path, img_size=None):
    assert exists(image_path), "image {} does not exist".format(image_path)
    img = cv2.imread(image_path)
    if (len(img.shape) != 3) or (img.shape[2] != 3):
        img = np.dstack((img, img, img))

    if (img_size is not None):
        img = cv2.resize(img, img_size)

    img = img.astype("float32")
    return img


def save_image(img, path):
    cv2.imwrite(path, np.clip(img, 0, 255).astype(np.uint8))

def get_files(img_dir):
    filenames = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
    return filenames
