import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# the copy in the first lines of the function is so that you don't ruin
# the original image. it will create a new one. 

def add_SP_noise(im, p):
    sp_noise_im = im.copy()

    return sp_noise_im


def clean_SP_noise_single(im, radius):
    noise_im = im.copy()
    # TODO: add implementation

    return clean_im


def clean_SP_noise_multiple(images):
    # TODO: add implementation
    return clean_image


def add_Gaussian_Noise(im, s):
    gaussian_noise_im = im.copy()
    # TODO: add implementation

    return gaussian_noise_im


def clean_Gaussian_noise(im, radius, maskSTD):
    # TODO: add implementation
    return cleaned_im.astype(np.uint8)


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()
    # TODO: add implementation

    return bilateral_im.astype(np.uint8)


