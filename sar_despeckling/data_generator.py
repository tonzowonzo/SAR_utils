import cv2
import os
import random
import numpy as np
import scipy.ndimage
from scipy import special
import tensorflow as tf


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
    img_[mask == 1] = 180
    img_[mask == 2] = 40
    return img_


def simulate_sar_noise(img: np.array) -> np.array:
    """
    Takes in a clean image and simulates SAR speckle on it. Based on:
    https://gitlab.telecom-paris.fr/ring/SAR-CNN/-/blob/master/SAR_CNN_test.ipynb
    Turned into Numpy code for ease of use.

    :param img: A numpy array of a clean image.
    :return: A speckled image.
    """
    # Set constants for noise creation.
    M = 10.089038980848645
    m = -1.429329123112601
    L = 1
    c = (1 / 2) * (special.psi(L) - np.log(L))
    cn = c / (M - m)

    s = tf.zeros(shape=tf.shape(img))

    for k in range(0, L):
        gamma = (np.abs(tf.complex(np.random.normal(size=img.shape),
                                   np.random.normal(size=img.shape))) ** 2) / 2
        s = s + gamma
    s_amplitude = np.sqrt(s / L)
    log_speckle = np.log(s_amplitude)
    log_norm_speckle = log_speckle / (M - m)

    noisy_im = img + log_norm_speckle
    return noisy_im


def generator():
    for i in range(5):
        path = "C:/Users/tim.iles/Documents/Projects/sar_speckle_filtering/data/big_dataset/NWPU-RESISC45/"
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
            real_image = cv2.resize(real_image, (64, 64))
            # Scale image between 0 and 1.

            # # Create an image that's noisy.
            # noise = np.random.uniform(low=-18, high=18, size=(64, 64))
            #
            # noisey_image = real_image.copy() + noise
            # # Add salt + pepper noise too.
            # noisey_image = addsalt_pepper(noisey_image, 0.9)

            real_image = real_image / 255
            noisey_image = simulate_sar_noise(real_image)
            real_image = real_image.reshape((1, 64, 64, 1))
            noisey_image = noisey_image.reshape((1, 64, 64, 1))

            X_stack[i % 32, :, :, :] = noisey_image
            y_stack[i % 32, :, :, :] = real_image


