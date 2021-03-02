import cv2
import os
import random
import numpy as np
import scipy.ndimage
from scipy import special
import tensorflow as tf
import rasterio

img_size = 256
batch_size = 1
path = "D:/despeckling/"


def load_image(image_path: str):
    """
    Loads an image as a numpy array, given an image path.

    :param image_path: The path to the image to be opened.
    :return: A numpy array of the image.
    """

    img = cv2.imread(image_path, 0)
    return img


def generator():
    """
    Generates a stack of images batch_size large and in the size of img_size.

    :return: Yields real_image, clean_image
    """
    for i in range(20):
        image_paths = []
        dirty_paths = []

        # Get the paths of the images.
        dirty_list = os.listdir(f"{path}noisy_1c/")
        for img_num, image in enumerate(os.listdir(f"{path}clean_1c/")):
            image_paths.append(f"{path}clean_1c/{image}")
            dirty_paths.append(f"{path}noisy_1c/{dirty_list[img_num]}")

        zipped_image_paths = list(zip(image_paths, dirty_paths))

        # Shuffle the image paths.
        random.shuffle(zipped_image_paths)

        for i, img_path in enumerate(zipped_image_paths):
            clean_path, dirty_path = img_path

            if i % batch_size == 0:
                if i != 0:
                    yield X_stack, y_stack

                X_stack = np.zeros((batch_size, img_size, img_size, 1)).astype(np.float32)
                y_stack = np.zeros((batch_size, img_size, img_size, 1)).astype(np.float32)

            # Load image
            real_image = load_image(clean_path)
            real_image = cv2.resize(real_image, (img_size, img_size))
            real_image = (real_image / 255).astype(np.float32)
            # Scale image between 0 and 1.
            dirty_image = load_image(dirty_path)
            dirty_image = cv2.resize(dirty_image, (img_size, img_size))
            dirty_image = (dirty_image / 255).astype(np.float32)
            # noisey_image, noise = simulate_sar_noise(real_image)
            real_image = real_image.reshape((1, img_size, img_size, 1))
            dirty_image = dirty_image.reshape((1, img_size, img_size, 1))
            # noise = noise.reshape((1, img_size, img_size, 1))
            X_stack[i % batch_size, :, :, :] = dirty_image
            y_stack[i % batch_size, :, :, :] = real_image
