import cv2
import numpy as np
from scipy.signal import convolve2d as conv
from scipy.ndimage.filters import gaussian_filter as gaussian
import matplotlib.pyplot as plt

def Threshold(im, thres):
    thres_im = np.zeros(im.shape)
    thres_im[im[:] > thres] = 255
    return thres_im
def sobel(im):
    new_im=im.copy()
    s_x=[[-1,0,1],
        [-2,0,2]
        ,[-1,0,1]
        ]
    s_y=[[1,2,1],
        [0,0,0]
        ,[-1,-2,-1]
        ]
    s_y=np.array(s_y)
    s_x=np.array(s_x)
    s_x=s_x
    s_y=s_y
    df_dx=conv(new_im,s_x)
    df_dy=conv(new_im,s_y)
    new_im=np.sqrt(np.square(df_dx)+np.square(df_dy))
    new_im=Threshold(new_im,122)

    return np.uint8(new_im)


def canny(im):
    im=cv2.blur(im,(5,5))
    return cv2.Canny(im,50,200,10,L2gradient = True)


def hough_circles(im):
    im_c = im.copy()
    circles = cv2.HoughCircles(cv2.medianBlur(im,7), cv2.HOUGH_GRADIENT, 1, 70,
                              param1=180, param2=35, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    # Draw the detected circles on the original image
    # Draw the circles on the original image
    for i in circles[0, :]:
        cv2.circle(im_c, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(im_c, (i[0], i[1]), 2, (0, 0, 255), 3)

    return im_c


def hough_lines(im):
    im_l = im.copy()
    
    return im_l
