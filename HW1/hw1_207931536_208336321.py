import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2

def histImage(im):
    h=np.zeros(256);
    dim=np.shape(im)
    for i in range(dim[0]):
        for j in range(dim[1]):
            h[im[i,j]]+=1
    return h


def nhistImage(im):
    nh=histImage(im)
    div=np.shape(im)[0]*np.shape(im)[1]
    for i in range(256):
        nh[i]/=div
    return nh


def ahistImage(im):
    ah=histImage(im)
    for i in range(256):
        ah[i]+=ah[i-1]

    return ah


def calcHistStat(h):
    m = np.matmul(h, np.reshape(np.arange(0, 256), (256, 1)))/sum(h)
    e = np.matmul(h, np.reshape(pow(np.arange(0, 256)-m, 2), (256 , 1)))/sum(h)
    return m, e


def mapImage(im,tm):
    # nim=[[tm[im[i, j]] for i in range(im.shape[0]) ] for j in range(im.shape[1]) ]
    # nim=np.clip(nim,0,255)
    nim=im.copy()
    for i in range (len(tm)):
        nim[im==i]=tm[i]
    nim = np.clip(nim, 0, 255)

    return nim


def histEqualization(im):
    accumulative=ahistImage(im)
    a=np.zeros(256)
    for i in range (256):
        a[i]= accumulative[255]/256
    goal_hist=np.cumsum(a)
    tm=np.zeros(256)
    goal_hist_pointer=0
    for i  in range (256):
        while(accumulative[i]>goal_hist[goal_hist_pointer]):
            goal_hist_pointer+=1
        tm[i]=goal_hist_pointer
    return tm
