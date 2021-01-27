import cv2
import os
import random
import numpy as np


def load_image(image_path: str):
    """
    Loads an image as a numpy array, given an image path.

    :param image_path: The path to the image to be opened.
    :return: A numpy array of the image.
    """

    img = cv2.imread(image_path, 0)
    return img

def addsalt_pepper(img, SNR):
    img_ = img.copy()
    h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    img_[mask == 1] = 0.6 # salt noise
    img_[mask == 2] = 0.2
    return img_



def generator():
    for i in range(5):
        path = "C:/Users/tim.iles/Documents/Projects/sar_speckle_filtering/data/2750/"
        image_paths = []
        # Get the paths of the images.
        for folder in os.listdir(path):
            for image in os.listdir(f"{path}{folder}"):
                image_paths.append(f"{path}{folder}/{image}")

        # Shuffle the image paths.
        random.shuffle(image_paths)

        for i, image_path in enumerate(image_paths):
            if i % 32 == 0:
                if i != 0:
                    yield X_stack, y_stack

                X_stack = np.zeros((32, 64, 64, 1))
                y_stack = np.zeros((32, 64, 64, 1))

            # Load image
            real_image = load_image(image_path)
            # Scale image between 0 and 1.
            real_image = real_image / 255

            # Create an image that's noisy.
            noise = np.random.uniform(low=-0.06, high=0.06, size=(64, 64))

            noisey_image = real_image.copy() + noise
            # Add salt + pepper noise too.
            noisey_image = addsalt_pepper(noisey_image, 0.9)

            real_image = real_image.reshape((1, 64, 64, 1))
            noisey_image = noisey_image.reshape((1, 64, 64, 1))

            X_stack[i % 32, :, :, :] = noisey_image
            y_stack[i % 32, :, :, :] = real_image
