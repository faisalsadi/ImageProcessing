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
    # # bilateral_im = im.copy()
    # # # TODO: add implementation
    # #
    # # return bilateral_im.astype(np.uint8)
    #
    # # Create an empty array to store the result
    # # create the coordinate grid for the neighborhood
    # X, Y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    #
    # # initialize the output image
    # cleanIm = np.zeros_like(im,dtype=float)
    # gs = np.fromfunction(lambda x, y:
    #                      np.exp(-(np.divide(pow(x - radius, 2) + pow(y - radius, 2), 2 * pow(stdSpatial, 2)))),
    #                      (2 * radius + 1, 2 * radius + 1),
    #                      dtype=np.float)
    # gs = np.divide(gs, np.sum(gs))
    # # iterate over the image pixels
    # for i in range(radius, im.shape[0] - radius):
    #     for j in range(radius, im.shape[1] - radius):
    #         # extract the neighborhood window
    #         window = im[i - radius:i + radius + 1, j - radius:j + radius + 1]
    #
    #         # # calculate the spatial weight
    #         # gs = np.exp(-((X[i - radius:i + radius + 1, j - radius:j + radius + 1] - i) ** 2 + (
    #         #             Y[i - radius:i + radius + 1, j - radius:j + radius + 1] - j) ** 2) / (2 * stdSpatial ** 2))
    #         # gs = np.divide(gs, np.sum(gs))
    #         # calculate the intensity weight
    #         gi = np.exp(-((window - im[i, j]) ** 2) / (2 * stdIntensity ** 2))
    #         gi = np.divide(gi, np.sum(gi))
    #         # combine the weights
    #
    #         mask=gs*gi
    #
    #         # apply the mask using convolution
    #         cleanIm[i, j] = np.divide(np.sum(mask * window), np.sum(mask))
    #
    # return cleanIm.astype(np.uint8)
    #


######################################################################
    # copy image
    bilateral_im = im.copy()

    # creating symmetric gaussian mask the same way as function 4
    gs = np.fromfunction(lambda x, y:
                         np.exp(-(np.divide(pow(x - radius, 2) + pow(y - radius, 2), 2 * pow(stdSpatial, 2)))),
                         (2 * radius + 1, 2 * radius + 1),
                         dtype=np.float)
    gs = np.divide(gs, np.sum(gs))

    # creating 2 more masks and re-mapping image
    for i in range(radius, im.shape[0] - radius):
        for j in range(radius, im.shape[1] - radius):
            window = im[i - radius: i + radius + 1, j - radius: j + radius + 1].astype(np.float)

            gi = np.exp(-np.divide(np.power(np.full((2 * radius + 1, 2 * radius + 1), im[i][j], dtype=float) - window, 2),
                 2 * pow(stdIntensity, 2)))
            gi = np.divide(gi, np.sum(gi))

            bilateral_im[i][j] = np.divide(np.sum(gi * gs * window), np.sum(gi * gs))

    # return value
    return bilateral_im.astype(np.uint8)



