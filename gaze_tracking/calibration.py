from __future__ import division
import cv2

import numpy as np
from core import FixationPoint_Standardization
from PIL import Image

from core.Config import Config
from gaze_tracking.pupil import Pupil


class Calibration(object):

    best_thres = 60  # 最佳阈值

    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames or len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """Returns the threshold value for the given eye.

        Argument:
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if side == 0:
            best_thres = int(sum(self.thresholds_left) / len(self.thresholds_left))
            return best_thres
        elif side == 1:
            best_thres = int(sum(self.thresholds_right) / len(self.thresholds_right))
            return best_thres

    @staticmethod
    def iris_size(frame):
        """Returns the percentage of space that the iris takes up on
        the surface of the eye.

        Argument:
            frame (numpy.ndarray): Binarized iris frame
        """
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        if contours is not None and len(contours) > 1:
            img = frame[5:-5, :]
            height, width = img.shape[:2]
            nb_pixels = height * width
            contours = sorted(contours, key=cv2.contourArea)
            nb_blacks = cv2.contourArea(contours[-2])
            if nb_pixels == 0:
                return 0
            return nb_blacks / nb_pixels
        else:
            return 0

    @staticmethod
    def find_best_threshold(eye_frame):
        """Calculates the optimal threshold to binarize the
        frame for the given eye.

        Argument:
            eye_frame (numpy.ndarray): Frame of the eye to be analyzed
        """

        average_iris_size = Config.AVERAGE_IRIS_SIZE
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)

            trials[threshold] = Calibration.iris_size(iris_frame)
            # cv2.imwrite(
            #     '../image/calibration/' + str(threshold) + '_' + str(trials[threshold]) + '.png', iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        print('iris_size=', iris_size)
        return best_threshold

    def evaluate(self, eye_frame, side):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)
