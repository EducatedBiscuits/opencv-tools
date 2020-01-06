import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor

# plots the images
def plot(images, rows, cols):

    length = len(images)
    fig = plt.figure(figsize = (10, 6))

    for i in range(1 ,length + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(images[i-1])
    plt.show()

# calculates the ratio of the rows versus columns
def divide_and_conquer(length):
    num_sqrt    = sqrt(length)
    round_sqrt  = floor(num_sqrt)
    if round_sqrt < num_sqrt:
        rows    = round_sqrt
        cols    = round_sqrt + 1
    else:
        rows    = round_sqrt
        cols     = round_sqrt
    return rows, cols



if __name__ == "__main__":

    import glob
    # import cv2
    image_links = glob.glob('/home/biscuit/Python-files/images/*.jpg')
    length = len(image_links)
    # images = [cv2.imread(i) for i in image_links]
    dummy_images = [np.random.randint(0, 255, (10, 10, 3)) for i in range(length)]

    print(len(dummy_images))

    rows, cols = divide_and_conquer(length)
    print(f"rows : {rows}")
    print(f"cols : {cols}")
    # plot(images, rows, cols)
    plot(dummy_images, rows, cols)



# END
