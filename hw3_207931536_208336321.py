import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import convolve2d

# the copy in the first lines of the function is so that you don't ruin
# the original image. it will create a new one. 

def add_SP_noise(im, p):
    sp_noise_im = im.copy()
    # Set the range of values for the x and y coordinates
    # Create a set of all possible pixel coordinates
    width, height = np.shape(im)[1], np.shape(im)[0]

    # Create a set of all possible pixel coordinates
    all_pixels = set((x, y) for x in range(width) for y in range(height))

    k = int(p * im.size)  # number of pixels to add noise to
    random_pixels = random.sample(all_pixels, k)

    # Add noise to the selected pixels
    for x, y in random_pixels:
        # Set the value of the pixel to 0 or 255 at random
        if random.random() < 0.5:
            sp_noise_im[y, x] = 0
        else:
            sp_noise_im[y, x] = 255
    return sp_noise_im


def clean_SP_noise_single(im, radius):
    noise_im = im.copy()
    # TODO: add implementation
    clean_im=scipy.ndimage.median_filter(noise_im,(2*radius+1,2*radius+1))
    return clean_im


def clean_SP_noise_multiple(images):
    # TODO: add implementation
    clean_image = np.median(images, axis=0)
    return clean_image


def add_Gaussian_Noise(im, s):
    gaussian_noise_im = im.copy()
    # TODO: add implementation

    # Generate the noise
    noise = np.random.normal(0, s, im.shape)

    # Add the noise to the image
    gaussian_noise_im = gaussian_noise_im + noise
    gaussian_noise_im=np.clip(gaussian_noise_im,0,255)
    return gaussian_noise_im.astype(np.uint8)


def clean_Gaussian_noise(im, radius, maskSTD):
    # TODO: add implementation
    # Create a 2D Gaussian kernel with the given std and radius
    cleaned_im=im.copy()
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * (maskSTD ** 2)))
    kernel/=np.sum(kernel)
    print(kernel)
    # Convolve the image with the kernel using scipy.signal.convolve2d
    cleaned_im = convolve2d(cleaned_im.astype(np.float), kernel, mode='same')

    return cleaned_im.astype(np.uint8)




def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()
    # TODO: add implementation

    return bilateral_im.astype(np.uint8)


