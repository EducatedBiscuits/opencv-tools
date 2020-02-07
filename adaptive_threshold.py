import numpy as np
import matplotlib.pyplot as plt
import cv2

# my adaptive threshold implementation based on an scholarly paper
# similar to wellner's method


# The lower the t the more stricter the rule to become 0 gets.
t = 10 # percentage
# cv2.namedWindow('threshold')
# cv2.createTrackbar('t', 'threshold', t, 100, update)

# form the integral image
def get_integral_image(image):
    h, w = image.shape
    print(f'image shape : {image.shape}')
    integral_image = np.zeros((h, w))
    print(f'integral image shape : {integral_image.shape}')

    for i in range(h):
        sum = 0
        for j in range(w):
            sum += image[i, j]
            if i == 0:
                integral_image[i, j] = sum
            else:
                integral_image[i, j] = sum + integral_image[i-1, j]

    return integral_image

# pads a grayscale image with wellner's method (s)
def zero_pad(image):
    h, w = image.shape
    s = np.uint32((1/8)*w)
    print(f"Wellner's : {s}")

    # accounts for the even kernel sizes
    if s%2 == 0:
        s = s - 1
    pad = np.uint8(((s-1)/2))
    print(f"Padding : {pad}")

    # form the newly padded image
    padded_image = np.zeros((h + 2*pad, w + 2*pad))
    padded_image[pad : pad + h, pad : pad + w] = image[:]

    return padded_image

# takes a grayscale image an outputs a threshold image
def threshold(image, t = 38):
    h, w    = image.shape
    s       = np.uint32((1/8)*w)
    pad     = np.uint32((s-1)/2)
    intimg  = get_integral_image(image)
    intimg  = zero_pad(intimg)
    output  = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            x1 = np.int32(i - s/2) + pad - 1
            x2 = np.int32(i + s/2) + pad - 1
            y1 = np.int32(j - s/2) + pad - 1
            y2 = np.int32(j + s/2) + pad - 1

            count = s*s
            sum   = intimg[x2, y2] - intimg[x2, y1 - 1] - intimg[x1 - 1, y2] + intimg[x1 - 1, y1 - 1]
            if(image[i, j] <= (sum/count)*(100 - t)/100):
                output[i, j] = 0
            else:
                output[i, j] = 255

    return output




if __name__ == "__main__":
    image = cv2.imread('/home/biscuit/Python-files/images/bookpage2.png')
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray  = np.uint8(gray)
    integral_image  = get_integral_image(gray)
    padded_image    = zero_pad(gray)

    thresh          = threshold(padded_image, t)
    # t = cv2.getTrackbarPos('t', 'threshold')
    # converting the maps to uint8 in order to display them
    thresh          = np.uint8(thresh)
    padded_image    = np.uint8(padded_image)
    integral_image  = np.uint8(integral_image)


    cv2.imshow('threshold', thresh)
    cv2.imshow('integral', integral_image)
    cv2.imshow('padded', padded_image)
    cv2.imshow('original', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    # END
