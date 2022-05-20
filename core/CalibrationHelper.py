from numpy import *

class CalibrationHelper(object):
    """该类是标定帮助类,保存的是注视点在屏幕中心时的值

    """
    # 定义类变量
    ec_x = None  # (float):中心点离眼部区域左侧的距离
    ec_y = None  # (float):中心点离眼部区域下侧的距离
    top2bottomDist = None  # (float):上眼皮和下眼皮之间的距离

    @staticmethod
    def histeq(im, nbr_bins=256):
        """对一幅灰度图像进行直方图均衡化"""
        # 计算图像的直方图
        # 在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
        imhist, bins = histogram(im.flatten(), nbr_bins, density=True)
        cdf = imhist.cumsum()
        cdf = 255.0 * cdf / cdf[-1]
        # 使用累积分布函数的线性插值，计算新的像素值
        im2 = interp(im.flatten(), bins[:-1], cdf)
        return im2.reshape(im.shape), cdf
