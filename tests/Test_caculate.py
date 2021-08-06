import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from core import FixationPoint_Standardization

def histequ(gray, nlevels=256):
    # Compute histogram
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    print ("histogram: ", histogram)

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    print ("uniform hist: ", uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity
    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            uniform_gray[i,j] = uniform_hist[gray[i,j]]

    return uniform_gray

if __name__ == '__main__':
    img = cv.imread('../image/0.jpg')
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    h, w = img.shape[:2]
    print(img[h//2][w//4*3])
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    # 画出直方图
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("number of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

    # dst = FixationPoint_Standardization.adaptive_histogram_equalization(img)
    dst = cv.equalizeHist(img)
    hist1 = cv.calcHist([dst], [0], None, [256], [0, 256])
    # 画出直方图
    plt.figure()
    plt.title("adaptive_histogram_equalization Histogram")
    plt.xlabel("Bins")
    plt.ylabel("number of Pixels")
    plt.plot(hist1)
    plt.xlim([0, 256])
    plt.show()

    ret, img1 = cv.threshold(img, 65, 255, cv.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    img1 = cv.erode(img1, kernel, iterations=2)

    # imgs = np.hstack([img, dst])
    cv.imshow('img', img)
    cv.imshow('dst', dst)
    cv.imshow('img1', img1)

    cv.waitKey(1)
