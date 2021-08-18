import math
import numpy as np
import cv2

from gaze_tracking.calibration import Calibration
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
    cnt = 0
    def __init__(self, original_frame, landmarks, side, calibration, option):
        self.frame = None
        self.origin = None
        self.center = None
        self.top2bottom = None
        self.pupil = None

        # self._analyze(original_frame, landmarks, side, calibration)
        # option : 0:预先的瞳孔阈值调整 1:正式开始注视点标定
        if option == 0:
            self.adjust_threshold(original_frame, landmarks, side, calibration)
        elif option == 1:
            self.find_pupil(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        print('region', region)

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def isolate_eye(self, frame, landmarks):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array(landmarks)
        region = region.astype(np.int32)

        dst1 = region[5][1] - region[1][1]
        dst2 = region[4][1] - region[2][1]
        self.top2bottom = (dst1 + dst2) / 2

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0])
        max_x = np.max(region[:, 0])
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        cv2.imwrite('../image/gray_eye/' + str(self.cnt) + '.jpg', self.frame)
        Eye.cnt += 1
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2 - 0.5, height / 2 - 0.5)
        print('center:', self.center)

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.

        Arguments:
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)

        Returns:
            The computed ratio
        """
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)

    def find_pupil(self, original_frame, landmarks):
        """寻找虹膜/瞳孔

        :param original_frame: 捕获原图像
        :param landmarks: 眼睛区域特征点
        :return:
        """
        #  original_frame是眼睛的灰度图像（矩形）
        self.isolate_eye(original_frame, landmarks)
        #  内眼角点
        # x, y = landmarks[0]
        # inner_eye = original_frame[y - 1:y + 1, x:x + 3]
        # inner_eye_gray = np.mean(np.array(inner_eye))
        # threshold = inner_eye_gray
        # print('innerEyeGray= ', threshold)
        threshold = Calibration.best_thres
        self.pupil = Pupil(self.frame, threshold)

    def adjust_threshold(self, original_frame, landmarks, side, calibration):
        """自适应阈值计算

        :param original_frame: 捕获原图像
        :param landmarks: 眼睛区域特征点
        :param side: 左/右眼
        :param calibration: 标定对象
        :return:
        """
        self.isolate_eye(original_frame, landmarks)
        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)
        if calibration.is_complete():
            Calibration.best_thres = calibration.threshold(side)
