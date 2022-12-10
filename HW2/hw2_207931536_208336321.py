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

    for i in range(8):
        x_tag[2*i, 0]=pointset2[i,0]
        x_tag[2*i+1, 0] = pointset2[i, 1]
    for i in range (8):
        x[2*i, 0] = pointset1[i, 0]
        x[2*i, 1] = pointset1[i, 1]
        x[2*i, 2] = 1
        x[2*i, 6] = -(pointset1[i, 0] * pointset2[i, 0])
        x[2*i, 7] = -(pointset1[i, 1] * pointset2[i, 0])

        x[2*i+1, 3] = pointset1[i, 0]
        x[2*i+1, 4] = pointset1[i, 1]
        x[2*i+1, 5] = 1
        x[2*i+1, 6] = -(pointset1[i, 0] * pointset2[i, 1])
        x[2*i+1, 7] = -(pointset1[i, 1] * pointset2[i, 1])

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