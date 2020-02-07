import numpy as np
import cv2
from math import ceil


# a very basic implementation of contrasting
# takes every pixel and multiplies it with a fixed number
def app_cont(image, contrast):
    h, w = image.shape
    new_image = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            b = ceil(image[i][j] * contrast)
            if b > 255:
                b = 255
            new_image[i][j] = b

    return new_image





if __name__ == "__main__":
    # change the image location to your own image location
    image   = cv2.imread('/home/biscuit/Python-files/images/bookpage.jpg')
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = 1.6
    cont_img = app_cont(gray, contrast)


    cv2.imshow('original', gray)
    cv2.imshow('contrasted', cont_img)
    cv2.imwrite('simple_contrast.jpg', cont_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# END
