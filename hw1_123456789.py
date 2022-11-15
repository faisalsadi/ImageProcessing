import numpy as np
import matplotlib.pyplot as plt
import cv2


def histImage(im):
    h = [0] * 256
    for i in range(im.shape[0]):  # traverses through height of the image
        for j in range(im.shape[1]):  # traverses through width of the image
            # print (im[i][j])
            h[im[i][j]] += 1
    return h


def nhistImage(im):
    nh = histImage(im)
    pixnum = im.shape[0] * im.shape[1]
    for i in range(len(nh)):
        nh[i] /= pixnum
    return nh


def ahistImage(im):
    ah = histImage(im)
    sum = 0
    for i in range(len(ah)):
        sum += ah[i]
        ah[i] = sum
    return ah


def calcHistStat(h):
    m = np.matmul(h, np.reshape(np.arange(0, 256), (256, 1))) / sum(h)
    e = np.matmul(h, np.reshape(pow(np.arange(0, 256) - m, 2), (256, 1))) / sum(h)
    return m, e


def mapImage(im, tm):
    nim = [[tm[im[i][j]] for i in range(im.shape[0])] for j in range(im.shape[1])]
    nim=np.clip(nim,0,255)
    return nim

# def histEqualization(im):
#
#     return tm
#
