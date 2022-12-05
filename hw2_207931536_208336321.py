import cv2
import matplotlib.pyplot as plt
import numpy as np
# size of the image
m,n = 921, 750

# frame points of the blank wormhole image
src_points = np.float32([[0, 0],
                            [int(n / 3), 0],
                            [int(2 * n /3), 0],
                            [n, 0],
                            [n, m],
                            [int(2 * n / 3), m],
                            [int(n / 3), m],
                            [0, m]])

# blank wormhole frame points
dst_points = np.float32([[96, 282],
                       [220, 276],
                       [344, 276],
                       [468, 282],
                       [474, 710],
                       [350, 744],
                       [227, 742],
                       [103, 714]]
                      )


def find_transform(pointset1, pointset2):
    x_tag=np.zeros((16, 1))
    A=np.zeros((8, 1))
    x=np.zeros((16, 8))

    x_tag[0, 0]=pointset2[0,0]
    x_tag[1, 0] = pointset2[0, 1]
    x_tag[2, 0] = pointset2[3, 0]
    x_tag[3, 0] = pointset2[3, 1]
    x_tag[4, 0] = pointset2[4, 0]
    x_tag[5, 0] = pointset2[4, 1]
    x_tag[6, 0] = pointset2[7, 0]
    x_tag[7, 0] = pointset2[7, 1]

    x_tag[8, 0]=pointset2[1,0]
    x_tag[9, 0] = pointset2[1, 1]
    x_tag[10, 0] = pointset2[2, 0]
    x_tag[11, 0] = pointset2[2, 1]
    x_tag[12, 0] = pointset2[5, 0]
    x_tag[13, 0] = pointset2[5, 1]
    x_tag[14, 0] = pointset2[6, 0]
    x_tag[15, 0] = pointset2[6, 1]


    # x_tag[0, 0]=pointset2[2,0]
    # x_tag[1, 0] = pointset2[2, 1]
    # x_tag[2, 0] = pointset2[3, 0]
    # x_tag[3, 0] = pointset2[3, 1]
    # x_tag[4, 0] = pointset2[4, 0]
    # x_tag[5, 0] = pointset2[4, 1]
    # x_tag[6, 0] = pointset2[5, 0]
    # x_tag[7, 0] = pointset2[5, 1]
    #########################################
    x[0,0]=pointset1[0,0]
    x[0,1]=pointset1[0,1]
    x[0,2]=1
    x[0,6]=-(pointset1[0,0]*pointset2[0,0])
    x[0,7]=-(pointset1[0,1]*pointset2[0,0])

    x[1,3]=pointset1[0,0]
    x[1,4]=pointset1[0,1]
    x[1,5]=1
    x[1,6]=-(pointset1[0,0]*pointset2[0,1])
    x[1,7]=-(pointset1[0,1]*pointset2[0,1])
    #########################################
    x[2, 0] = pointset1[3, 0]
    x[2, 1] = pointset1[3, 1]
    x[2, 2] = 1
    x[2, 6] = -(pointset1[3, 0] * pointset2[3, 0])
    x[2, 7] = -(pointset1[3, 1] * pointset2[3, 0])

    x[3, 3] = pointset1[3, 0]
    x[3, 4] = pointset1[3, 1]
    x[3, 5] = 1
    x[3, 6] = -(pointset1[3, 0] * pointset2[3, 1])
    x[3, 7] = -(pointset1[3, 1] * pointset2[3, 1])

    #########################################
    x[4, 0] = pointset1[4, 0]
    x[4, 1] = pointset1[4, 1]
    x[4, 2] = 1
    x[4, 6] = -(pointset1[4, 0] * pointset2[4, 0])
    x[4, 7] = -(pointset1[4, 1] * pointset2[4, 0])

    x[5, 3] = pointset1[4, 0]
    x[5, 4] = pointset1[4, 1]
    x[5, 5] = 1
    x[5, 6] = -(pointset1[4, 0] * pointset2[4, 1])
    x[5, 7] = -(pointset1[4, 1] * pointset2[4, 1])

    #########################################
    x[6, 0] = pointset1[7, 0]
    x[6, 1] = pointset1[7, 1]
    x[6, 2] = 1
    x[6, 6] = -(pointset1[7, 0] * pointset2[7, 0])
    x[6, 7] = -(pointset1[7, 1] * pointset2[7, 0])

    x[7, 3] = pointset1[7, 0]
    x[7, 4] = pointset1[7, 1]
    x[7, 5] = 1
    x[7, 6] = -(pointset1[7, 0] * pointset2[7, 1])
    x[7, 7] = -(pointset1[7, 1] * pointset2[7, 1])



    # ## ## ############################
    ########################################
    x[8,0]=pointset1[1,0]
    x[8,1]=pointset1[1,1]
    x[8,2]=1
    x[8,6]=-(pointset1[1,0]*pointset2[1,0])
    x[8,7]=-(pointset1[1,1]*pointset2[1,0])

    x[9,3]=pointset1[1,0]
    x[9,4]=pointset1[1,1]
    x[9,5]=1
    x[9,6]=-(pointset1[1,0]*pointset2[1,1])
    x[9,7]=-(pointset1[1,1]*pointset2[1,1])
    #########################################
    x[10, 0] = pointset1[2, 0]
    x[10, 1] = pointset1[2, 1]
    x[10, 2] = 1
    x[10, 6] = -(pointset1[2, 0] * pointset2[2, 0])
    x[10, 7] = -(pointset1[2, 1] * pointset2[2, 0])

    x[11, 3] = pointset1[2, 0]
    x[11, 4] = pointset1[2, 1]
    x[11, 5] = 1
    x[11, 6] = -(pointset1[2, 0] * pointset2[2, 1])
    x[11, 7] = -(pointset1[2, 1] * pointset2[2, 1])


    #########################################
    x[12, 0] = pointset1[5, 0]
    x[12, 1] = pointset1[5, 1]
    x[12, 2] = 1
    x[12, 6] = -(pointset1[5, 0] * pointset2[5, 0])
    x[12, 7] = -(pointset1[5, 1] * pointset2[5, 0])

    x[13, 3] = pointset1[5, 0]
    x[13, 4] = pointset1[5, 1]
    x[13, 5] = 1
    x[13, 6] = -(pointset1[5, 0] * pointset2[5, 1])
    x[13, 7] = -(pointset1[5, 1] * pointset2[5, 1])

    #########################################
    x[14, 0] = pointset1[6, 0]
    x[14, 1] = pointset1[6, 1]
    x[14, 2] = 1
    x[14, 6] = (pointset1[6, 0] * pointset2[6, 0])
    x[14, 7] = (pointset1[6, 1] * pointset2[6, 0])

    x[15, 3] = pointset1[6, 0]
    x[15, 4] = pointset1[6, 1]
    x[15, 5] = 1
    x[15, 6] = (pointset1[6, 0] * pointset2[6, 1])
    x[15, 7] = (pointset1[6, 1] * pointset2[6, 1])


    ## ## ## ############################

    A= np.matmul( np.linalg.pinv(x),x_tag)
    A=np.append(A,[1])
    A=A.reshape(3,3)
    T=A


    return T


def trasnform_image(image, T):
    new_image=np.zeros( (np.shape(image)[0],np.shape(image)[1]) )
    inverse=np.linalg.inv(T)
    dim = np.shape(image)
    for y_tag in range ( dim[0] ):
        for x_tag in range (dim[1] ):
            xyw=np.matmul(inverse,np.reshape(np.float32([x_tag,y_tag,1]),(3,1) ) )
            xyw/=xyw[2,0]
            x = round(xyw[0, 0])
            y = round(xyw[1, 0])
            if x>n-1 or x<0 or y>m-1 or y<0:
                new_image[y_tag,x_tag]=0
            else:
                #new_image[y_tag,x_tag]=image[ np.clip(round(xyw[1,0]),0,m-1), np.clip(round(xyw[0,0]),0,n-1)]
                new_image[y_tag, x_tag] = image[y,x]




    return new_image


def create_wormhole(im, T, iter=5):
    new_image = im
    im_t=im
    for i  in range (iter):
        im_t =trasnform_image(im_t,T)
        new_image=np.add(new_image,im_t)

    new_image = np.clip(new_image, 0, 255)
    return new_image