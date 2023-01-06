import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2


def clean_baby(im):
    clean_im = im.copy()
    radius = 1
    for i in range(radius, im.shape[0] - radius):
        for j in range(radius, im.shape[1] - radius):
            clean_im[i][j] = np.median(im[i - radius: i + radius + 1, j - radius: j + radius + 1])
    src_points = np.float32([[5, 20],
                             [110, 20],

                             [5, 130],
                             [110, 130]])
    dst_points = np.float32([[0, 0],
                             [255, 0],
                             [0, 255],
                             [255, 255],
                             ]
                            )

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    clean_im = cv2.warpPerspective(clean_im, matrix, (255, 255))

    return clean_im


def clean_windmill(im):
    clean_im = im.copy()
    img_fourier = np.fft.fft2(clean_im)  # fft - remember this is a complex numbers matrix
    img_fourier = np.fft.fftshift(img_fourier)  # shift so that the DC is in the middle
    img_fourier[124][100]=0
    img_fourier[132][156] = 0
    img_inv = np.fft.ifft2(img_fourier)
    # return np.log(abs(img_fourier))
    return abs(img_inv)


def clean_watermelon(im):
    clean_im = im.copy()

    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    # Apply the high pass filter to the image
    filtered_image = convolve2d(clean_im, kernel, mode='same')

    # Adjust the intensity of the filter and add it to the original image
    enhanced_image = np.clip(clean_im + filtered_image, 0, 255).astype(np.uint8)

    return enhanced_image


def clean_umbrella(im):
    clean_im=im.copy()
    img_fourier = np.fft.fft2(clean_im)  # fft - remember this is a complex numbers matrix
    img_fourier = np.fft.fftshift(img_fourier)  # shift so that the DC is in the middle
    img_inv = np.fft.ifft2(img_fourier)
    return np.log(abs(img_fourier))
    # return abs(img_inv)


def clean_USAflag(im):
    clean_im = im.copy()
    radius = 5
    for i in range(radius, im.shape[0] - radius):
        for j in range(radius, im.shape[1] - radius):
            if j > 140 or i > 87:
                clean_im[i][j] = np.median(im[i, j - radius: j + radius + 1])

    # return value
    return clean_im


def clean_cups(im):
    clean_im = im.copy()
    img_fourier = np.fft.fft2(clean_im)  # fft - remember this is a complex numbers matrix
    img_fourier = np.fft.fftshift(img_fourier)  # shift so that the DC is in the middle
    for i in range(107,149):
        for j in range(107,149):
            img_fourier[i][j]*=2
    img_inv = np.fft.ifft2(img_fourier)
    clean_im=img_inv.copy()
    radius = 1
    for i in range(radius, img_inv.shape[0] - radius):
        for j in range(radius, img_inv.shape[1] - radius):
            clean_im[i][j] = np.median(img_inv[i - radius: i + radius + 1, j - radius: j + radius + 1])

    # return np.log(abs(img_fourier))
    return abs(clean_im)




def clean_house(im):
    clean_im = im.copy()

    img_fourier = np.fft.fft2(clean_im)  # fft - remember this is a complex numbers matrix
    img_fourier = np.fft.fftshift(img_fourier)  # shift so that the DC is in the middle
    #
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if(j%18==0):
                img_fourier[i][j]*=2

    img_inv = np.fft.ifft2(img_fourier)
    return np.log(abs(img_fourier))
    # return abs(clean_im)


def clean_bears(im):
    nim = im.copy()
    tm = histEqualization(im)
    for i in range(len(tm)):
        nim[im == i] = tm[i]
    nim = np.clip(nim, 0, 255)

    im_array = np.array(nim)

    # Set the gamma value
    gamma = 1.05

    # Correct the image using the gamma correction function
    corrected_im_array = im_array ** (1 / gamma)

    # Convert the corrected image array back to an image object
    corrected_im = np.reshape(corrected_im_array, (256, 256))
    return corrected_im

    return clean_im


def histImage(im):
    h = np.zeros(256);
    dim = np.shape(im)
    for i in range(dim[0]):
        for j in range(dim[1]):
            h[im[i, j]] += 1
    return h


def nhistImage(im):
    nh = histImage(im)
    div = np.shape(im)[0] * np.shape(im)[1]
    for i in range(256):
        nh[i] /= div
    return nh


def ahistImage(im):
    ah = histImage(im)
    for i in range(256):
        ah[i] += ah[i - 1]

    return ah


def mapImage(im, tm):
    # nim=[[tm[im[i, j]] for i in range(im.shape[0]) ] for j in range(im.shape[1]) ]
    # nim=np.clip(nim,0,255)
    nim = im.copy()
    for i in range(len(tm)):
        nim[im == i] = tm[i]
    nim = np.clip(nim, 0, 255)

    return nim


def histEqualization(im):
    accumulative = ahistImage(im)
    a = np.zeros(256)
    for i in range(256):
        a[i] = accumulative[255] / 256
    goal_hist = np.cumsum(a)
    tm = np.zeros(256)
    goal_hist_pointer = 0
    for i in range(256):
        while (accumulative[i] > goal_hist[goal_hist_pointer]):
            goal_hist_pointer += 1
        tm[i] = goal_hist_pointer
    return tm


'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images\windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_fourier = np.fft.fft2(img) # fft - remember this is a complex numbers matrix 
    img_fourier = np.fft.fftshift(img_fourier) # shift so that the DC is in the middle
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    
    plt.subplot(1,3,2)
    plt.imshow(np.log(abs(img_fourier)), cmap='gray') # need to use abs because it is complex, the log is just so that we can see the difference in values with out eyes.
    plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''
