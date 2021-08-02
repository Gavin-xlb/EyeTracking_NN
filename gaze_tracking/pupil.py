import numpy as np
import cv2

from core import FixationPoint_Standardization
from core.Config import Config

from PIL import Image




class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None
        self.radius = None

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
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        ret, new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)
        # print('OTSU threshold = ', ret)
        return new_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """

        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contours[-2])
        self.x = x
        self.y = y

        radius = Config.AVERAGE_PUPIL_RADIUS
        self.radius = radius
        print('radius = ', radius)

        height, width = eye_frame.shape[:2]
        pupil_outline = np.full((height, width), 255, np.uint8)
        cv2.circle(pupil_outline, (int(x), int(y)), int(radius), (0, 0, 0), 1)

        mask = np.full((height, width), 255, np.uint8)

        cv2.fillPoly(mask, [contours[-2]], (0, 0, 0))
        cv2.circle(eye_frame, (int(x), int(y)), int(radius), (0, 255, 0), 1)
        imgs = np.hstack([self.iris_frame, eye_frame, pupil_outline])
        cv2.imshow('show_image', imgs)
        # try:
        #     moments = cv2.moments(contours[-2])
        #     self.x = moments['m10'] / moments['m00']
        #     self.y = moments['m01'] / moments['m00']
        # except (IndexError, ZeroDivisionError):
        #     pass
