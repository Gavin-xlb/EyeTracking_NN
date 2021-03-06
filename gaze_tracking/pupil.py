import numpy as np
import cv2

from core import FixationPoint_Standardization, Hough
from core.CalibrationHelper import CalibrationHelper
from core.Config import Config

from PIL import Image
import matplotlib.pyplot as plt

from gaze_tracking import calibration


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None  # 虹膜二值化图像
        self.threshold = threshold  # 最佳二值化阈值
        self.x = None  # CG以眼睛区域左上角为坐标原点的横坐标
        self.y = None  # CG以眼睛区域左上角为坐标原点的纵坐标
        self.cg_x = None  # CG以眼睛区域左下角为坐标原点的横坐标
        self.cg_y = None  # CG以眼睛区域左下角为坐标原点的纵坐标
        self.radius = None  # 虹膜半径

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        # eye_frame = FixationPoint_Standardization.adaptive_histogram_equalization(eye_frame)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        # erode_pre = new_frame
        # new_frame = cv2.resize(new_frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA)
        # new_frame = cv2.erode(new_frame, kernel, iterations=2)
        # new_frame = cv2.resize(new_frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        erode_pre = new_frame
        hist = cv2.calcHist([new_frame], [0], None, [256], [0, 256])

        # imgs = np.hstack([erode_pre, new_frame])
        # cv2.imshow('show_image', imgs)
        ret, new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)
        # print('OTSU threshold = ', ret)
        return new_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        average_iris_size = Config.AVERAGE_IRIS_SIZE
        trials = {}
        histogram_eye, cdf = CalibrationHelper.histeq(np.array(eye_frame))
        histogram_eye= np.uint8(histogram_eye)

        # Algorithm1:对人眼区域二值化后，最小外接圆拟合
        for thres in range(self.threshold - 10, self.threshold + 11, 1):
            temp_bin_iris = self.image_processing(histogram_eye, thres)
            trials[thres] = calibration.Calibration.iris_size(temp_bin_iris)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        print('adapterBestThres=', best_threshold)

        self.iris_frame = self.image_processing(histogram_eye, best_threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contours[-2])

        # Algorithm2:利用算法1获得大致的瞳孔中心位置，从而获得ROI
        # 对瞳孔区域(ROI区域)进行边缘检测后，霍夫变换圆形检测
        # height, width = eye_frame.shape
        # estimated_radius = height / 4
        # eye_roi = histogram_eye[y-estimated_radius:y+estimated_radius, x-estimated_radius:x+estimated_radius]
        # Hough.detect_circle(eye_roi)

        self.x = round(x, 2)
        self.y = round(y, 2)
        self.cg_x = round(x, 2)
        self.cg_y = self.iris_frame.shape[0] - round(y, 2) - 1

        # radius = Config.AVERAGE_PUPIL_RADIUS
        self.radius = radius
        print('radius = ', radius)

        height, width = eye_frame.shape[:2]
        print('eyeH:%d;eyeW:%d' % (height, width))
        pupil_outline = np.full((height, width), 255, np.uint8)
        cv2.circle(pupil_outline, (int(x), int(y)), int(radius), (0, 0, 0), 1)
        for point in contours[-2]:
            cv2.circle(pupil_outline, (int(point[0][0]), int(point[0][1])), 0, (0, 0, 255), 1)
            cv2.circle(eye_frame, (int(point[0][0]), int(point[0][1])), 0, (0, 0, 255), 1)

        cv2.circle(eye_frame, (int(x), int(y)), int(radius), (0, 0, 0), 1)
        imgs = np.hstack([self.iris_frame, eye_frame, pupil_outline])
        cv2.imshow('show_image', imgs)

