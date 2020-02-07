import numpy as np
from histogram import cum_hist, get_hist
import matplotlib.pyplot as plt


# issue : For very low contrast images it acts like a threshold algorithm :/ (fixed convert to uint8)
# takes a grayscale image and performs linear contrast mapping
def mautocontrast(image, p):
    h, w = image.shape
    pixels = h*w
    histogram = get_hist(image, 0)
    # plt.plot(histogram, color = 'gray')
    # plt.show()
    c_hist = cum_hist(histogram)

    print("forming histogram...")
    new_image = np.zeros((h, w))
    print("creating new image...")

    for i in np.arange(256):
        if c_hist[i] >= pixels*p:
            a_low = i
            break
    for i in np.arange(255, -1, -1):
        if c_hist[i] <= pixels*(1-p):
            a_high = i
            break

    print(f"Lower threshold for cumulative hist : {a_low}")
    print(f"higher threshold for cumulative hist : {a_high}")

    for i in range(h):
        for j in range(w):
            if image[i, j] <= a_low:
                new_image[i, j] = 0
            elif image[i, j] >= a_high:
                new_image[i, j] = 255
            else:
                new_image[i, j] = (float(image[i, j] - a_low)/(a_high - a_low))*255

    return new_image



if __name__ == "__main__":

    import cv2
    image = cv2.imread('/home/biscuit/Python-files/images/bookpage.jpg')
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    new_image = mautocontrast(gray, p = 0.005)

    # display image
    new_image = np.uint8(new_image)

    cv2.imshow('original', gray)
    cv2.imshow('modified_autocontrast', new_image)
    cv2.imwrite('modified_autocontrast_bookpage.jpg', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# END
