import cv2
import math
import numpy as np


def detect_circle(eye_roi):
    h, w = eye_roi.shape
    eye_roi = cv2.medianBlur(eye_roi, 3)
    eye_roi = cv2.GaussianBlur(eye_roi, (17, 19), 0)
    canny_img = cv2.Canny(eye_roi, 30, 40)
    cv2.imwrite('../image/canny11.bmp', canny_img)
    circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT, 1, h / 8, param1=40, param2=6,
                               minRadius=int(h / 16), maxRadius=int(h))
    # circles = np.uint16(np.around(circles))
    critical = h / 2
    trials = {}
    print('critical', critical)
    for circle in circles[0]:
        trials[circle[1]] = circle

    if circles is not None and len(circles) > 0:
        print('circleNum= ', len(circles[0]))
        # print(circles[0])
        # 多个圆中选取合适的拟合圆
        _, target_circle = min(trials.items(), key=(lambda p: abs(p[0] - critical)))

        x, y, radius = target_circle
        # cv2.circle(eye_roi, (math.ceil(x), math.ceil(y)), math.ceil(radius), (132, 135, 239), 2)
        # cv2.imwrite('../image/' + "hough.jpg", eye_roi)
        return x, y, radius

def test_detect_circle(eye_roi):
    eye_roi = cv2.medianBlur(eye_roi, 3)
    eye_roi = cv2.GaussianBlur(eye_roi, (17, 19), 0)
    canny_img = cv2.Canny(eye_roi, 30, 40)
    cv2.imwrite('../image/' + "canny.jpg", canny_img)
    circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT, 1, eye_roi.shape[0]/8,
                               param1=40, param2=6, minRadius=25, maxRadius=30)
    # circles = np.uint16(np.around(circles))
    print('circles=', circles)
    print(type(circles))
    for circle in circles[0]:
        cv2.circle(eye_roi, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (132, 135, 239), 2)
    cv2.imwrite('../image/' + "hough.jpg", eye_roi)
    if circles is not None and len(circles) > 0:
        print('circleNum= ', len(circles[0]))
        # print(circles[0])
        x, y, radius = circles[0][0]
        return x, y, radius
    return
# img_gray = cv2.imread('../image/eye.jpg', cv2.IMREAD_GRAYSCALE)
# x, y, radius = detect_circle(img_gray)
# print(x, y, radius)

# import time
# import cv2
# import math
# from core import Hough_transform
# from core import Canny
#
# Path = '../image/eye.jpg'  # 图片路径
# Save_Path = '../image/'  # 结果保存路径
# Reduced_ratio = 2  # 为了提高计算效率，将图片进行比例缩放所使用的比例值
# Guassian_kernal_size = 3
# HT_high_threshold = 80
# HT_low_threshold = 50
# Hough_transform_step = 6
# Hough_transform_threshold = 110
#
# if __name__ == '__main__':
#     start_time = time.time()
#
#     img_gray = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
#     y, x = img_gray.shape[0:2]
#     img_gray = cv2.resize(img_gray, (int(x / Reduced_ratio), int(y / Reduced_ratio)))  # 图片缩放
#
#     # canny
#     print('Canny ...')
#     canny = Canny.Canny(Guassian_kernal_size, img_gray, HT_high_threshold, HT_low_threshold)
#     canny.canny_algorithm()
#     # cv2.imshow('canny', canny.img)
#     cv2.imwrite(Save_Path + "canny_result.jpg", canny.img)
#     canny_img = cv2.Canny(img_gray, 50, 80)
#     cv2.imwrite(Save_Path + "canny.jpg", canny_img)
#
#     # hough
#     print('Hough ...')
#     Hough = Hough_transform.Hough_transform(canny_img, canny.angle, Hough_transform_step, Hough_transform_threshold)
#     circles = Hough.Calculate()
#     for circle in circles:
#         cv2.circle(img_gray, (math.ceil(circle[0]), math.ceil(circle[1])), math.ceil(circle[2]), (132, 135, 239), 2)
#     cv2.imwrite(Save_Path + "hough_result.jpg", img_gray)
#     # cv2.imshow('hough', img_gray)
#     print('Finished!')
#
#     end_time = time.time()
#     print("running time" + str(end_time - start_time))
