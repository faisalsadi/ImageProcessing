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
    # Apply edge detection method on the image
    #edges = canny(cv2.blur(im,(5,5)))
    edges = cv2.Canny(cv2.blur(im,(5,5)),120,300,5,L2gradient = True)
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 140)


    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(im_l, (x1, y1), (x2, y2), (0, 0, 255), 4)
    ##
    # for line in lines:
    #     rho, theta = line[0]
    #
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #
    #     x1 = int(a * rho + 1000 * (-b))
    #     y1 = int(b * rho + 1000 * (a))
    #     x2 = int(a * rho - 1000 * (-b))
    #     y2 = int(b * rho - 1000 * (a))
    #
    #     cv2.line(im_l, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return im_l
