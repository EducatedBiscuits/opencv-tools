import numpy as np

# autocontrast algorithm
def autocontrast(image):
    min = 255
    max = 0
    h, w = image.shape
    new_image = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if image[i, j] > max:
                max = image[i, j]
            if image[i, j] < min:
                min = image[i, j]
    for i in range(h):
        for j in range(w):
            new_image[i, j] = (float(image[i, j] - min)/(max - min))*255
    return new_image




if __name__ == "__main__":
    from adaptive_threshold import threshold
    import cv2
    image = cv2.imread('/home/biscuit/Python-files/images/bookpage.jpg')
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    new_image = autocontrast(gray)
    thresh  = threshold(new_image, t=30)

    # display the image
    new_image = np.uint8(new_image)
    cv2.imshow('contrasted image', new_image)
    cv2.imshow('thresholded image', thresh)
    cv2.imwrite('autocontrast.jpg', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# END
