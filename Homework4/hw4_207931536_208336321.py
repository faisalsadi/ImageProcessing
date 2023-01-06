import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2
# size of the image
m, n = 256, 256

# frame points of the blank wormhole image
src_points = np.float32([[6, 20],
                         # [int( (111-6) / 3), 20],
                         # [int(2 * (111-6) / 3), 20],
                         [111, 20],
                         [111, 130],
                         # [int(2 * (111-6) / 3), 130],
                         # [int((111-6) / 3), 130],
                         [6, 130]])

# blank wormhole frame points
dst_points = np.float32([[0, 0],
                         # [int( (255) / 3), 0],
                         # [int( 2*(255) / 3), 0],
                         [255, 0],
                         [255, 255],
                         # [int( 2*(255)), 255],
                         # [int( (255) / 3), 255],
                         [0, 255]]
                        )
def gkern(l, sig):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def cleanImageMean(im, radius, maskSTD):
    ax = np.linspace(-3 / 2., 3 / 2., 4)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(5))
    kernel = np.outer(gauss, gauss)
    filter = kernel / np.sum(kernel)
    cleaned_im = convolve2d(im,filter,mode='same')
    return cleaned_im
def histEqualization(im):
    totalN = im.shape[0] * im.shape[1]
    h_goal = totalN / 256 * np.ones((256,))
    ah, ah_goal = ahistImage(im), np.cumsum(h_goal)
    tm = np.zeros((256,))
    i, j = 0, 0
    while i < 256 and j < 256:
        ac1 = ah[i]
        ac2 = ah_goal[j]
        if ac1 <= ac2:
            tm[i] = j
            i += 1
        else:
            j += 1
    return tm

def ahistImage(im):
    h = histImage(im)
    ah = np.cumsum(h)
    return ah
def histImage(im):
    # loop over image pixels
    h = np.zeros((256,))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            g = im[i, j]
            g = int(g)
            h[g] += 1
    return h

def mapImage(im,tm):
    # code in one line
    nim = tm[im.astype(int)].astype(int)
    return nim
def clean_Gaussian_noise(im, radius, maskSTD):
    # TODO: add implementation
    # Create a 2D Gaussian kernel with the given std and radius
    cleaned_im=im.copy()
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * (maskSTD ** 2)))
    kernel/=np.sum(kernel)
    # Convolve the image with the kernel using scipy.signal.convolve2d
    cleaned_im = convolve2d(cleaned_im.astype(np.float), kernel, mode='same')

    return cleaned_im.astype(np.uint8)

def find_transform(pointset1, pointset2):
    x_tag = np.zeros((8, 1))
    A = np.zeros((8, 1))
    x = np.zeros((8, 8))

    for i in range(4):
        x_tag[2 * i, 0] = pointset2[i, 0]
        x_tag[2 * i + 1, 0] = pointset2[i, 1]
    for i in range(4):
        x[2 * i, 0] = pointset1[i, 0]
        x[2 * i, 1] = pointset1[i, 1]
        x[2 * i, 2] = 1
        x[2 * i, 6] = -(pointset1[i, 0] * pointset2[i, 0])
        x[2 * i, 7] = -(pointset1[i, 1] * pointset2[i, 0])

        x[2 * i + 1, 3] = pointset1[i, 0]
        x[2 * i + 1, 4] = pointset1[i, 1]
        x[2 * i + 1, 5] = 1
        x[2 * i + 1, 6] = -(pointset1[i, 0] * pointset2[i, 1])
        x[2 * i + 1, 7] = -(pointset1[i, 1] * pointset2[i, 1])

    A = np.matmul(np.linalg.pinv(x), x_tag)
    A = np.append(A, [1])
    A = A.reshape(3, 3)
    T = A

    return T


def trasnform_image(image, T):
    new_image = np.zeros((np.shape(image)[0], np.shape(image)[1]))
    inverse = np.linalg.inv(T)
    dim = np.shape(image)
    for y_tag in range(dim[0]):
        for x_tag in range(dim[1]):
            xyw = np.matmul(inverse, np.reshape(np.float32([x_tag, y_tag, 1]), (3, 1)))
            xyw /= xyw[2, 0]
            x = round(xyw[0, 0])
            y = round(xyw[1, 0])
            if x > (n - 1) or x < 0 or y > (m - 1) or y < 0:
                new_image[y_tag, x_tag] = 0
            else:
                new_image[y_tag, x_tag] = image[y, x]

    return new_image

def clean_SP_noise_single(im, radius):
    clean_im = im.copy()

    for i in range(radius, im.shape[0] - radius):
        for j in range(radius, im.shape[1] - radius):
            clean_im[i][j] = np.median(im[i - radius : i + radius + 1, j - radius : j + radius + 1])

    # return value
    return clean_im

def clean_baby(im):
    clean_im=clean_SP_noise_single(im,1)
    T=find_transform(src_points,dst_points)
    clean_im=trasnform_image(clean_im,T)

    return clean_im


def clean_windmill(im):
    clean_im = np.fft.fft2(im)
    clean_im[4, 28] = 0
    clean_im[252, 228] = 0
    return np.abs((np.fft.ifft2(clean_im)))


def clean_watermelon(im):
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    clean_im = cv2.filter2D(im, -1, sharpen_kernel)
    return clean_im


def clean_umbrella(im):

    msk = np.zeros(im.shape)
    msk[1, 1] = 0.5
    msk[5, 80] = 0.5
    mskFFT = np.fft.fft2(msk)
    imFFT = np.fft.fft2(im)
    ind = (np.abs(mskFFT) < 0.01)
    mskFFT[ind] = 1
    clean_im = imFFT / mskFFT
    return np.abs(np.fft.ifft2(clean_im))


def clean_USAflag(im):
    clean_im=im.copy()
    radius=5
    for i in range (radius,im.shape[0]-radius):
        for j in range (radius,im.shape[1]-radius):
            if i>87 or j>140:
                clean_im[i][j]=np.median(im[i,j-radius:j+radius+1])
    return clean_im


def clean_cups(im):
    FFT = np.fft.fftshift(np.fft.fft2(im))
    FFT[108:149, 108:149] = 1.8 * FFT[108:149, 108:149]
    clean_im = np.abs(np.fft.ifft2(np.fft.ifftshift(FFT)))
    clean_im = cleanImageMean(clean_im, 4,5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    clean_im = cv2.filter2D(clean_im, -1, sharpen_kernel)
    return clean_im


def clean_house(im):
    msk = np.zeros(im.shape)
    msk[0, 0:10] = 0.1
    mskFFT = np.fft.fft2(msk)
    ind = (np.abs(mskFFT) <= 0.01)
    mskFFT[ind] = 1
    return np.abs(np.fft.ifft2(np.fft.fft2(im) / mskFFT))


def clean_bears(im):
    im_max = im.max()
    im_min = im.min()
    a = 255/(im_max-im_min)
    cleaned_im = (a*(im-im_min)).round().astype(int)
    return cleaned_im



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