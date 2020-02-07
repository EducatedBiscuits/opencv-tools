import numpy as np

# simple histogram
# bins = 256
def get_hist(image, channel):
    hist = np.zeros((256))
    try:
        h, w, c = image.shape
        print("colored image")
    except:
        h, w    = image.shape
        print("grayscale image")

    if len(image.shape) == 3:
        for i in range(h):
            for j in range(w):
                hist[image[i, j, channel]] += 1
    else:
        for i in range(h):
            for j in range(w):
                hist[image[i, j]] += 1
    return hist

def cum_hist(hist):
    cumulative_hist = hist
    for i in np.arange(1, 256):
        cumulative_hist[i] = cumulative_hist[i - 1] + cumulative_hist[i]
    return cumulative_hist


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    image = cv2.imread('/home/biscuit/Python-files/images/bookpage.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    color = ('blue', 'green', 'red')

    plt.figure(1)
    for i, n in enumerate(color):
        h = get_hist(image, i)
        plt.plot(h, color = n)
        plt.title('colored image histogram')
    plt.savefig('color_hist.png')

    plt.figure(2)
    h1 = get_hist(gray, 0)
    plt.plot(h1, color = 'gray')
    plt.title('grayscale image histogram')

    plt.savefig('gray_hist.png')
    plt.figure(3)
    for i, n in enumerate(color):
        h2 = cum_hist(get_hist(image, i))
        plt.plot(h2, color = n)
        plt.title('cumulative histogram')
    plt.savefig('cum_hist.png')
    plt.show()





# END
