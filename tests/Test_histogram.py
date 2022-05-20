from PIL import Image
from pylab import *
from numpy import *

import matplotlib.pyplot as plt

def histeq(im,nbr_bins = 256):
    """对一幅灰度图像进行直方图均衡化"""
    #计算图像的直方图
    #在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
    imhist,bins = histogram(im.flatten(), nbr_bins, density=True)
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    #使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


pil_im = Image.open(r'D:\python_Projects\EyeTracking_NN\image\calibration\0epoch_0point_0(-2.0, 0.5).bmp')   #打开原图
pil_im_gray = pil_im.convert('L')     #转化为灰度图像
# pil_im_gray.show()         #显示灰度图像

im = array(Image.open(r'D:\python_Projects\EyeTracking_NN\image\calibration\0epoch_0point_0(-2.0, 0.5).bmp').convert('L'))
plt.hist(im.ravel(), 256)
plt.figure()
# figure()
# hist(im.flatten(),256)

im2, cdf = histeq(im)
plt.hist(im2.ravel(), 256)
plt.figure()
plt.show()
# figure()
# hist(im2.flatten(),256)
# show()

im2 = Image.fromarray(uint8(im2))
# im2.show()
# print(cdf)
# plot(cdf)
im2.save(r"D:\python_Projects\EyeTracking_NN\image\junheng.jpg")