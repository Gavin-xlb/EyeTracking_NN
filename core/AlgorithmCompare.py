import cv2
from core.CalibrationHelper import CalibrationHelper
from numpy import *
import face_recognition
import os
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from core import ScreenHelper
from core.Config import Config

from core.Draw3D import Draw3D
from core import Surface_fitting
from core.Evaluate import Evaluate
from core.get_parabola import solution
from core.get_point import get_args, get_result, cal_quartic_ik
from core.Hough import detect_circle

screenhelper = ScreenHelper.ScreenHelper()
screen_H = screenhelper.getHResolution() / 1.25
screen_W = screenhelper.getWResolution() / 1.25
judge_critical = screen_H / 6
calibration_path = '../image/raw_calibration'
prediction_base_path = '../image/raw_prediction'
point_list = []  # 9 points' coordinates
ECCG_list = []
ROW_POINT = Config.CALIBRATION_POINTS_ROW
COL_POINT = Config.CALIBRATION_POINTS_COL
average_accuracy = 0
average_accuracy_X = 0
average_accuracy_Y = 0
delta_distance_list = [] # 存放距离差


class Eye(object):
    def __init__(self, original_frame, landmarks, method = 'minEnclosingCircle'):
        self.center = None
        self.top2bottom = None
        self.frame = None
        self.origin = None
        self.pupil = None

        self.find_pupil(original_frame, landmarks, method)

    def find_pupil(self, original_frame, landmarks, method):
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
        self.pupil = Pupil(self.frame, method)

    def isolate_eye(self, frame, landmarks):
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (list): Points of an eye (from the 68 Multi-PIE landmarks)
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
        region[1][1] -= 5
        region[2][1] -= 5
        region[4][1] += 5
        region[5][1] += 5
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0])
        max_x = np.max(region[:, 0])
        min_y = np.min(region[:, 1])
        max_y = np.max(region[:, 1])

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2 - 0.5, height / 2 - 0.5)
        print('center:', self.center)

class Pupil(object):
    def __init__(self, eye_frame, method):
        self.iris_frame = None  # 虹膜二值化图像
        self.threshold = int(get_bestThreshold())  # 最佳二值化阈值
        self.x = None  # CG以眼睛区域左上角为坐标原点的横坐标
        self.y = None  # CG以眼睛区域左上角为坐标原点的纵坐标
        self.cg_x = None  # CG以眼睛区域左下角为坐标原点的横坐标
        self.cg_y = None  # CG以眼睛区域左下角为坐标原点的纵坐标
        self.radius = None  # 虹膜半径
        self.iris_eye_ratio = None  # iris_frame（瞳孔图像）,eye_frame(眼眶图像)中非白的像素点个数之比

        self.detect_iris(eye_frame, method)
        self.cal_ratio(eye_frame)

    def image_processing(self, eye_frame, threshold):
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

    def cal_ratio(self, eye_frame):
        if self.iris_frame is not None:
            iris_count = 0
            eye_count = 0
            for line in self.iris_frame:
                for pixel in line:
                    if pixel != 255:
                        iris_count += 1
            for line in eye_frame:
                for pixel in line:
                    if pixel != 255:
                        eye_count += 1
            self.iris_eye_ratio = iris_count/ eye_count


    def iris_size(self, frame):
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

    def detect_iris(self, eye_frame, method):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        average_iris_size = Config.AVERAGE_IRIS_SIZE
        trials = {}

        cv2.imwrite('../image/eye_frame.bmp', eye_frame)
        histogram_eye, cdf = CalibrationHelper.histeq(np.array(eye_frame))
        histogram_eye = np.uint8(histogram_eye)
        # histogram_eye = eye_frame

        # Algorithm1:对人眼区域二值化后，最小外接圆拟合
        for thres in range(self.threshold - 10, self.threshold + 11, 1):
            temp_bin_iris = self.image_processing(histogram_eye, thres)
            trials[thres] = self.iris_size(temp_bin_iris)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        print('adapterBestThres=', best_threshold)

        self.iris_frame = self.image_processing(histogram_eye, best_threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contours[-2])

        # Algorithm2:利用算法1获得大致的瞳孔中心位置，从而获得ROI
        # 对瞳孔区域(ROI区域)进行边缘检测后，霍夫变换圆形检测
        if method == 'Hough':
            height, width = eye_frame.shape
            estimated_radius = height / 2
            min_y = int(y - estimated_radius)
            max_y = int(y + estimated_radius)
            min_x = int(x -estimated_radius)
            max_x = int(x + estimated_radius)
            eye_roi = histogram_eye[5:height-5, min_x:max_x]
            cv2.imwrite('../image/test.bmp', histogram_eye)
            cv2.imwrite('../image/eye_roi.bmp', eye_roi)
            x, y, radius = detect_circle(eye_roi)
            # x, y, radius = detect_circle(self.iris_frame)

        self.x = round(x, 2)
        self.y = round(y, 2)
        self.cg_x = round(x, 2)
        self.cg_y = self.iris_frame.shape[0] - round(y, 2) - 1

        # radius = Config.AVERAGE_PUPIL_RADIUS
        self.radius = radius
        print('radius = ', radius)

        height, width = eye_frame.shape[:2]
        print('eyeH:%d;eyeW:%d' % (height, width))
        # pupil_outline = np.full((height, width), 255, np.uint8)
        # cv2.circle(pupil_outline, (int(x), int(y)), int(radius), (0, 0, 0), 1)
        # for point in contours[-2]:
        #     cv2.circle(pupil_outline, (int(point[0][0]), int(point[0][1])), 0, (0, 0, 255), 1)
        #     cv2.circle(eye_frame, (int(point[0][0]), int(point[0][1])), 0, (0, 0, 255), 1)
        #
        # cv2.circle(eye_frame, (int(x), int(y)), int(radius), (0, 0, 0), 1)
        # imgs = np.hstack([self.iris_frame, eye_frame, pupil_outline])
        # cv2.imshow('show_image', imgs)

def get_bestThreshold():
    with open('../res/processing_data.txt', 'r') as fo:
        s = fo.readline().rstrip('\n')
        l = s.split('=')
        if l[0] == 'best_threshold':
            return float(l[1])
        return 31

def getPointPos():
    def f(i):
        return i % COL_POINT

    def g(i):
        return i // ROW_POINT

    global point_list
    d = Config.CALIBRATION_POINTS_INTERVAL_EDGE
    w = Config.CALIBRATION_POINTS_WIDTH
    h = Config.CALIBRATION_POINTS_HEIGHT
    center_index = Config.CALIBRATION_POINTS_NUM // 2
    for i in range(9):
        if i == 0:
            x = f(center_index) * (screen_W - 2 * d) / (COL_POINT - 1) + d
            y = g(center_index) * (screen_H - 2 * d) / (ROW_POINT - 1) + d
        elif i <= center_index:
            x = f(i - 1) * (screen_W - 2 * d) / (COL_POINT - 1) + d
            y = g(i - 1) * (screen_H - 2 * d) / (ROW_POINT - 1) + d
        else:
            x = f(i) * (screen_W - 2 * d) / (COL_POINT - 1) + d
            y = g(i) * (screen_H - 2 * d) / (ROW_POINT - 1) + d
        point_list.append((x, y))

def image_preprocessing(image):
    '''

    :param image: 原始灰度图像
    :return: 直方图均衡化之后的图像
    '''
    # f = cv2.cvtColor(array(image), cv2.COLOR_RGB2GRAY)
    histogram_f, cdf = CalibrationHelper.histeq(array(image))
    frame = uint8(histogram_f)
    return frame

def find_faceimage(image):
    '''

    :param image: 直方图均衡化之后的图像
    :return:
    '''
    s = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    face_locations = face_recognition.face_locations(s, model='cnn')
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5
        face_image = image[top:bottom, left:right]
        return face_image


def read_calibration_images():
    d_0 = {}
    d_1 = {}
    d_2 = {}
    for filename in os.listdir(calibration_path):
        # print(filename) #just for test
        img = cv2.imread(calibration_path + "/" + filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = image_preprocessing(img)
        name_list = filename.split('_')
        epoch = name_list[0].strip()[0]
        point = name_list[1].strip()[0]
        if epoch == '0':
            if point in d_0.keys():
                d_0[point].append(img)
                # d_0[point].append(filename)
            else:
                d_0[point] = [img]
                # d_0[point] = [filename]
        elif epoch == '1':
            if point in d_1.keys():
                d_1[point].append(img)
                # d_1[point].append(filename)
            else:
                d_1[point] = [img]
                # d_1[point] = [filename]
        elif epoch == '2':
            if point in d_2.keys():
                d_2[point].append(img)
                # d_2[point].append(filename)
            else:
                d_2[point] = [img]
                # d_2[point] = [filename]
    return d_0, d_1, d_2


def read_prediction_images():
    prediction_path = prediction_base_path + '/2022-05-18 15h46m46s'
    d = {}
    for filename in os.listdir(prediction_path):
        real_pos = filename[11:]
        # print(real_pos)
        point_imgs_path = prediction_path + '/' + filename
        imgs = []
        for imgname in os.listdir(point_imgs_path):
            # print(imgname) #just for test
            img = cv2.imread(point_imgs_path + "/" + imgname)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = image_preprocessing(img)
            imgs.append(img)
        d[real_pos] = imgs
    return d


def get_relationship_eye_screenpoint():
    """将EC-CG向量和屏幕坐标的关系写入文件

    :return: None
    """
    # input dict into txt
    fo = open("../res/compare_ECCG_screenPoint.txt", "a")
    print('len(ECCG_list)', len(ECCG_list))
    for i in range(len(ECCG_list)):
        if i % 9 == 0:
            fo.write('\n')
        fo.write('%s:%s\n' % (ECCG_list[i], point_list[i % 9]))
    fo.write('\n')
    # 关闭打开的文件
    fo.close()

# 最小二乘
def caculateCoeficiente():
    """最小二乘法拟合函数

    :return: A,B
    """
    '''
        Z_screenX = a0  * x ^ 2 + a1 * x * y + a2 * y ^ 2 + a3 * x + a4 * y + a5
        Z_screenY = b0  * x ^ 2 + b1 * x * y + b2 * y ^ 2 + b3 * x + b4 * y + b5
    '''
    global ECCG_list

    X = [x[0] for x in ECCG_list]
    Y = [x[1] for x in ECCG_list]
    Z_screenX = [x[0] for x in point_list] * 3
    Z_screenY = [x[1] for x in point_list] * 3
    A = Surface_fitting.matching_3D(X, Y, Z_screenX)
    B = Surface_fitting.matching_3D(X, Y, Z_screenY)

    # 离散点和拟合函数可视化
    # 拟合注视点横坐标
    Draw3D.drawMap(X, Y, Z_screenX, 'x')
    # 拟合注视点纵坐标
    Draw3D.drawMap(X, Y, Z_screenY, 'y')

    return A, B

# SVR
def caculateCoeficiente_SVR():
    """SVR算法拟合函数

    :return: clf_x, clf_y
    """
    print("strat fitting.....")
    eccgpoint = np.array([x for x in ECCG_list])
    Z_screenX = np.array([x[0] for x in point_list]*3)
    Z_screenY = np.array([x[1] for x in point_list]*3)
    print("X: ", eccgpoint)
    print("Y: ", Z_screenX)
    clf_x = SVR(kernel='poly', degree=4, gamma="auto", coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                cache_size=200, verbose=False, max_iter=- 1)
    clf_y = SVR(kernel='poly', degree=4, gamma="auto", coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                cache_size=200, verbose=False, max_iter=- 1)
    clf_x.fit(eccgpoint, Z_screenX)
    # print(clf_x.predict(eccgpoint))
    clf_y.fit(eccgpoint, Z_screenY)
    # print(clf_y.predict(eccgpoint))
    return clf_x, clf_y


# RandomForestClassifier
def caculateCoeficiente_RF():
    """随机森林算法拟合函数

    :return: rfc_x, rfc_y
    """
    eccgpoint = np.array([x for x in ECCG_list])
    Z_screenX = np.array([x[0] for x in point_list]*3)
    Z_screenY = np.array([x[1] for x in point_list]*3)
    print("X: ", eccgpoint)
    print("Y: ", Z_screenX)
    rfc_x = RandomForestRegressor(n_estimators=50, random_state=1)
    rfc_x.fit(eccgpoint, Z_screenX)
    rfc_y = RandomForestRegressor(n_estimators=50, random_state=1)
    rfc_y.fit(eccgpoint, Z_screenY)

    # y_predict = rfc_x.predict(eccgpoint)
    # print('随机森林准确率', rfc_x.score(eccgpoint, Z_screenX))
    # print('随机森林精确率', precision_score(Z_screenX, y_predict, average='macro'))
    # print('随机森林召回率', recall_score(Y_test, y_predict, average='macro'))
    # print('F1', f1_score(Y_test, y_predict, average='macro'))
    return rfc_x, rfc_y


def caculateCoeficiente_distance():
    """最小二乘法拟合函数

    :return: D
    """
    '''
        Z_screenX = a0  * x ^ 2 + a1 * x * y + a2 * y ^ 2 + a3 * x + a4 * y + a5
        Z_screenY = b0  * x ^ 2 + b1 * x * y + b2 * y ^ 2 + b3 * x + b4 * y + b5
    '''

    global ECCG_list
    X = [x[0] for x in ECCG_list]
    Z_screenY = [x[1] for x in point_list] * 3
    print('##################################################')
    print(len(X))
    print('##################################################')
    print(len(delta_distance_list))
    D = Surface_fitting.matching_3D(X, delta_distance_list, Z_screenY)
    Draw3D.drawMap(X, delta_distance_list, Z_screenY, 'y')
    return D


def predict_1(A, B):
    '''
    Z_screenX = a0  * x ^ 2 + a1 * x * y + a2 * y ^ 2 + a3 * x + a4 * y + a5
    Z_screenY = b0  * x ^ 2 + b1 * x * y + b2 * y ^ 2 + b3 * x + b4 * y + b5
    '''
    global average_accuracy
    global average_accuracy_X
    global average_accuracy_Y
    a0 = A[0]
    a1 = A[1]
    a2 = A[2]
    a3 = A[3]
    a4 = A[4]
    a5 = A[5]
    b0 = B[0]
    b1 = B[1]
    b2 = B[2]
    b3 = B[3]
    b4 = B[4]
    b5 = B[5]
    prediction_imgs_dict = read_prediction_images()
    for real_pos, imgs_list in prediction_imgs_dict.items():
        print('len(imgs_list)=', len(imgs_list))
        points_list = []  # 所有预测成功的点
        dist_list = []  # 预测点到真实点的距离
        points_list_X = []  # 所有横坐标预测成功的点
        dist_list_X = []  # 预测点横坐标到真实点的距离
        points_list_Y = []  # 所有纵坐标预测成功的点
        dist_list_Y = []  # 预测点纵坐标到真实点的距离
        real_x, real_y = real_pos.split('_')
        real_x = int(real_x)
        real_y = int(real_y)
        evaluate = Evaluate(real_x, real_y)
        for img in imgs_list:
            # Extract the region of the image that contains the face
            face_image = find_faceimage(img)
            face_landmarks_list = face_recognition.face_landmarks(face_image)
            for face_landmarks in face_landmarks_list:
                right_eye_point = face_landmarks['right_eye']
                right_eye = Eye(face_image, right_eye_point)
                if right_eye is not None:
                    delta_dst = right_eye.top2bottom - CalibrationHelper.top2bottomDist
                    cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y + delta_dst)
                    EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round((cg[1] - CalibrationHelper.ec_y), 2))
                    if EC_CG:
                        print('real_pos=', real_pos)
                        print('predict_eccg:', EC_CG)
                        x = EC_CG[0]
                        y = EC_CG[1]
                        # x_predict = A.predict(np.array([EC_CG]))
                        # y_predict = B.predict(np.array([EC_CG]))
                        x_predict = a0 * x * x + a1 * x * y + a2 * y * y + a3 * x + a4 * y + a5
                        y_predict = b0 * x * x + b1 * x * y + b2 * y * y + b3 * x + b4 * y + b5
                        print('Z_screenX=%.2f,Z_screenY=%.2f' % (x_predict, y_predict))

                        points_list.append((x_predict, y_predict))
                        points_list_X.append((x_predict, y_predict))
                        points_list_Y.append((x_predict, y_predict))
                        dist = math.sqrt(
                            math.pow((x_predict - evaluate.screenX), 2) + math.pow((y_predict - evaluate.screenY),
                                                                                   2))
                        dist_list.append(dist)
                        dist_X = math.fabs(x_predict - evaluate.screenX)
                        dist_list_X.append(dist_X)
                        dist_Y = math.fabs(y_predict - evaluate.screenY)
                        dist_list_Y.append(dist_Y)
        evaluate.missed_num = 0
        evaluate.num = 120

        if len(dist_list) > 2:
            # (X,Y)同时考虑
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('XY\n')
            ma = max(dist_list)
            del points_list[dist_list.index(ma)]
            dist_list.remove(ma)

            mi = min(dist_list)
            del points_list[dist_list.index(mi)]
            dist_list.remove(mi)

            print('critical=', judge_critical)
            valid_points = []
            valid_dists = []
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('realPoint:%s' % '(' + str(evaluate.screenX) + ',' + str(
                    evaluate.screenY) + ')\n')
                fo.write('critical distance is %.2f\n' % judge_critical)
            for i in range(len(dist_list)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list[i]))
                if dist_list[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists.append(dist_list[i])
                    valid_points.append(points_list[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num = len(valid_dists)

            # 只考虑X
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('X\n')
            ma = max(dist_list_X)
            del points_list_X[dist_list_X.index(ma)]
            dist_list_X.remove(ma)

            mi = min(dist_list_X)
            del points_list_X[dist_list_X.index(mi)]
            dist_list_X.remove(mi)

            valid_points_X = []
            valid_dists_X = []
            for i in range(len(dist_list_X)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list_X[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list_X[i]))
                if dist_list_X[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists_X.append(dist_list_X[i])
                    valid_points_X.append(points_list_X[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num_X = len(valid_dists_X)

            # 只考虑Y
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('Y\n')
            ma = max(dist_list_Y)
            del points_list_Y[dist_list_Y.index(ma)]
            dist_list_Y.remove(ma)

            mi = min(dist_list_Y)
            del points_list_Y[dist_list_Y.index(mi)]
            dist_list_Y.remove(mi)

            valid_points_Y = []
            valid_dists_Y = []
            for i in range(len(dist_list_Y)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list_Y[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list_Y[i]))
                if dist_list_Y[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists_Y.append(dist_list_Y[i])
                    valid_points_Y.append(points_list_Y[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num_Y = len(valid_dists_Y)

            evaluate.caculate()
            average_accuracy += evaluate.acceptRatio
            average_accuracy_X += evaluate.acceptRatio_X
            average_accuracy_Y += evaluate.acceptRatio_Y

            output_predictInfo(evaluate, valid_points, valid_points_X, valid_points_Y)

            print('missRatio=', evaluate.missRatio)
            print('acceptRatio=', evaluate.acceptRatio)
            print('acceptRatio_X=', evaluate.acceptRatio_X)
            print('acceptRatio_Y=', evaluate.acceptRatio_Y)
    average_accuracy /= 9
    average_accuracy_X /= 9
    average_accuracy_Y /= 9
    fo = open("../res/compare_predictInfo.txt", "a+")
    msg = '平均准确率=' + str(round(average_accuracy, 2)) + '\nX平均准确率=' + str(
        round(average_accuracy_X, 2)) + '\nY平均准确率=' + str(round(average_accuracy_Y, 2))
    fo.write('%s\n\n' % msg)
    # 关闭打开的文件
    fo.close()


def predict_2(A, B):
    '''
    Z_screenX = a0  * x ^ 2 + a1 * x * y + a2 * y ^ 2 + a3 * x + a4 * y + a5
    Z_screenY = b0  * x ^ 2 + b1 * x * y + b2 * y ^ 2 + b3 * x + b4 * y + b5
    '''
    global average_accuracy
    global average_accuracy_X
    global average_accuracy_Y
    a0 = A[0]
    a1 = A[1]
    a2 = A[2]
    a3 = A[3]
    a4 = A[4]
    a5 = A[5]
    b0 = B[0]
    b1 = B[1]
    b2 = B[2]
    b3 = B[3]
    b4 = B[4]
    b5 = B[5]
    prediction_imgs_dict = read_prediction_images()
    for real_pos, imgs_list in prediction_imgs_dict.items():
        print('len(imgs_list)=', len(imgs_list))
        points_list = []  # 所有预测成功的点
        dist_list = []  # 预测点到真实点的距离
        points_list_X = []  # 所有横坐标预测成功的点
        dist_list_X = []  # 预测点横坐标到真实点的距离
        points_list_Y = []  # 所有纵坐标预测成功的点
        dist_list_Y = []  # 预测点纵坐标到真实点的距离
        real_x, real_y = real_pos.split('_')
        real_x = int(real_x)
        real_y = int(real_y)
        evaluate = Evaluate(real_x, real_y)
        for img in imgs_list:
            # Extract the region of the image that contains the face
            face_image = find_faceimage(img)
            face_landmarks_list = face_recognition.face_landmarks(face_image)
            for face_landmarks in face_landmarks_list:
                right_eye_point = face_landmarks['right_eye']
                right_eye = Eye(face_image, right_eye_point)
                if right_eye is not None:
                    delta_dst = right_eye.top2bottom - CalibrationHelper.top2bottomDist
                    cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y + delta_dst)
                    EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round(right_eye.pupil.iris_eye_ratio, 2))
                    if EC_CG:
                        print('real_pos=', real_pos)
                        print('predict_eccg:', EC_CG)
                        x = EC_CG[0]
                        y = EC_CG[1]
                        x_predict = a0 * x * x + a1 * x * y + a2 * y * y + a3 * x + a4 * y + a5
                        y_predict = b0 * x * x + b1 * x * y + b2 * y * y + b3 * x + b4 * y + b5
                        print('Z_screenX=%.2f,Z_screenY=%.2f' % (x_predict, y_predict))

                        points_list.append((x_predict, y_predict))
                        points_list_X.append((x_predict, y_predict))
                        points_list_Y.append((x_predict, y_predict))
                        dist = math.sqrt(
                            math.pow((x_predict - evaluate.screenX), 2) + math.pow((y_predict - evaluate.screenY),
                                                                                   2))
                        dist_list.append(dist)
                        dist_X = math.fabs(x_predict - evaluate.screenX)
                        dist_list_X.append(dist_X)
                        dist_Y = math.fabs(y_predict - evaluate.screenY)
                        dist_list_Y.append(dist_Y)
        evaluate.missed_num = 0
        evaluate.num = 120

        if len(dist_list) > 2:
            # (X,Y)同时考虑
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('XY\n')
            ma = max(dist_list)
            del points_list[dist_list.index(ma)]
            dist_list.remove(ma)

            mi = min(dist_list)
            del points_list[dist_list.index(mi)]
            dist_list.remove(mi)

            print('critical=', judge_critical)
            valid_points = []
            valid_dists = []
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('realPoint:%s' % '(' + str(evaluate.screenX) + ',' + str(
                    evaluate.screenY) + ')\n')
                fo.write('critical distance is %.2f\n' % judge_critical)
            for i in range(len(dist_list)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list[i]))
                if dist_list[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists.append(dist_list[i])
                    valid_points.append(points_list[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num = len(valid_dists)

            # 只考虑X
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('X\n')
            ma = max(dist_list_X)
            del points_list_X[dist_list_X.index(ma)]
            dist_list_X.remove(ma)

            mi = min(dist_list_X)
            del points_list_X[dist_list_X.index(mi)]
            dist_list_X.remove(mi)

            valid_points_X = []
            valid_dists_X = []
            for i in range(len(dist_list_X)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list_X[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list_X[i]))
                if dist_list_X[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists_X.append(dist_list_X[i])
                    valid_points_X.append(points_list_X[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num_X = len(valid_dists_X)

            # 只考虑Y
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('Y\n')
            ma = max(dist_list_Y)
            del points_list_Y[dist_list_Y.index(ma)]
            dist_list_Y.remove(ma)

            mi = min(dist_list_Y)
            del points_list_Y[dist_list_Y.index(mi)]
            dist_list_Y.remove(mi)

            valid_points_Y = []
            valid_dists_Y = []
            for i in range(len(dist_list_Y)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list_Y[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list_Y[i]))
                if dist_list_Y[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists_Y.append(dist_list_Y[i])
                    valid_points_Y.append(points_list_Y[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num_Y = len(valid_dists_Y)

            evaluate.caculate()
            average_accuracy += evaluate.acceptRatio
            average_accuracy_X += evaluate.acceptRatio_X
            average_accuracy_Y += evaluate.acceptRatio_Y

            output_predictInfo(evaluate, valid_points, valid_points_X, valid_points_Y)

            print('missRatio=', evaluate.missRatio)
            print('acceptRatio=', evaluate.acceptRatio)
            print('acceptRatio_X=', evaluate.acceptRatio_X)
            print('acceptRatio_Y=', evaluate.acceptRatio_Y)
    average_accuracy /= 9
    average_accuracy_X /= 9
    average_accuracy_Y /= 9
    fo = open("../res/compare_predictInfo.txt", "a+")
    msg = '平均准确率=' + str(round(average_accuracy, 2)) + '\nX平均准确率=' + str(
        round(average_accuracy_X, 2)) + '\nY平均准确率=' + str(round(average_accuracy_Y, 2))
    fo.write('%s\n\n' % msg)
    # 关闭打开的文件
    fo.close()


def predict_3(A, B, C):
    '''
        Z_screenX = a0  * x ^ 2 + a1 * x * y + a2 * y ^ 2 + a3 * x + a4 * y + a5
        Z_screenY = b0  * x ^ 2 + b1 * x * y + b2 * y ^ 2 + b3 * x + b4 * y + b5
        '''
    global average_accuracy
    global average_accuracy_X
    global average_accuracy_Y
    a0 = A[0]
    a1 = A[1]
    a2 = A[2]
    a3 = A[3]
    a4 = A[4]
    a5 = A[5]
    b0 = B[0]
    b1 = B[1]
    b2 = B[2]
    b3 = B[3]
    b4 = B[4]
    b5 = B[5]
    c0 = C[0]
    c1 = C[1]
    c2 = C[2]
    c3 = C[3]
    c4 = C[4]
    c5 = C[5]
    prediction_imgs_dict = read_prediction_images()
    for real_pos, imgs_list in prediction_imgs_dict.items():
        print('len(imgs_list)=', len(imgs_list))
        points_list = []  # 所有预测成功的点
        dist_list = []  # 预测点到真实点的距离
        points_list_X = []  # 所有横坐标预测成功的点
        dist_list_X = []  # 预测点横坐标到真实点的距离
        points_list_Y = []  # 所有纵坐标预测成功的点
        dist_list_Y = []  # 预测点纵坐标到真实点的距离
        real_x, real_y = real_pos.split('_')
        real_x = int(real_x)
        real_y = int(real_y)
        evaluate = Evaluate(real_x, real_y)
        for img in imgs_list:
            distance1 = 0
            distance2 = 0
            # Extract the region of the image that contains the face
            face_image = find_faceimage(img)
            face_landmarks_list = face_recognition.face_landmarks(face_image)
            for face_landmarks in face_landmarks_list:
                right_eye_point = face_landmarks['right_eye']
                right_eye = Eye(face_image, right_eye_point)
                if right_eye is not None:
                    # 拟合抛物线
                    print("Get prabola:")
                    X1 = np.array(
                        [right_eye_point[0][0], right_eye_point[1][0], right_eye_point[2][0],
                         right_eye_point[3][0]])
                    Y1 = np.array(
                        [right_eye_point[0][1], right_eye_point[1][1], right_eye_point[2][1],
                         right_eye_point[3][1]])
                    X2 = np.array(
                        [right_eye_point[0][0], right_eye_point[4][0], right_eye_point[5][0],
                         right_eye_point[3][0]])
                    Y2 = np.array(
                        [right_eye_point[0][1], right_eye_point[4][1], right_eye_point[5][1],
                         right_eye_point[3][1]])
                    para_args1, para_args2 = solution(X1, Y1, X2, Y2)

                    c_x = right_eye.pupil.x + right_eye.origin[0]
                    c_y = right_eye.pupil.y + right_eye.origin[1]
                    radius = right_eye.pupil.radius
                    print("c_x= ", c_x)
                    print("c_y= ", c_y)
                    print("radius= ", radius)
                    circle_args_list = [1, 1, -(2 * c_x), -(2 * c_y), c_x ** 2 + c_y ** 2 - radius ** 2]  # 圆方程一般式系数
                    print("circle equation: x^2+y^2+", circle_args_list[2], "x+", circle_args_list[3], "y+",
                          circle_args_list[4])

                    # 求瞳孔圆方程与上下眼眶抛物线方程的交点
                    num1 = 0  # 瞳孔圆方程与下眼眶抛物线交点数量
                    num2 = 0  # 瞳孔圆方程与上眼眶抛物线交点数量
                    for i in range(2):
                        if i == 0:
                            args1 = get_args(circle_args_list, para_args1)
                            # print(args1)
                            num1, x1 = cal_quartic_ik(args1)
                            print("num1: ", num1)
                            print("x1: ", x1)
                            '''
                            for point1 in x1:
                                x1_tmp.append(point1)'''
                            if x1 != 0:
                                result1 = get_result(args1, x1)
                                print("result1: ", result1)
                        elif i == 1:
                            args2 = get_args(circle_args_list, para_args2)
                            # print(args2)
                            num2, x2 = cal_quartic_ik(args2)
                            print("num2: ", num2)
                            print("x2: ", x2)
                            '''
                            for point2 in x2:
                                x2_tmp.append(point2)'''
                            if x2 != 0:
                                result2 = get_result(args2, x2)
                                print("result2: ", result2)
                    if num1 == 2:
                        distance1 = math.sqrt(
                            (result1[0][0] - result1[1][0]) ** 2 + (result1[0][1] - result1[1][1]) ** 2)
                    elif num1 == 1 | num1 == 0:
                        distance1 = 0
                    if num2 == 2:
                        distance2 = math.sqrt(
                            (result2[0][0] - result2[1][0]) ** 2 + (result2[0][1] - result2[1][1]) ** 2)
                    elif num2 == 1 | num2 == 0:
                        distance2 = 0
                    dist = distance1 - distance2

                    delta_dst = right_eye.top2bottom - CalibrationHelper.top2bottomDist
                    cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y + delta_dst)
                    EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round((cg[1] - CalibrationHelper.ec_y), 2))

                    if EC_CG:
                        print('real_pos=', real_pos)
                        print('predict_eccg:', EC_CG)
                        x = EC_CG[0]
                        y = EC_CG[1]
                        x_predict = a0 * x * x + a1 * x * y + a2 * y * y + a3 * x + a4 * y + a5
                        y_predict = c0 * x * x + c1 * x * dist + c2 * dist * dist + c3 * x + c4 * dist + c5
                        print('Z_screenX=%.2f,Z_screenY=%.2f' % (x_predict, y_predict))

                        points_list.append((x_predict, y_predict))
                        points_list_X.append((x_predict, y_predict))
                        points_list_Y.append((x_predict, y_predict))
                        dist = math.sqrt(
                            math.pow((x_predict - evaluate.screenX), 2) + math.pow((y_predict - evaluate.screenY),
                                                                                   2))
                        dist_list.append(dist)
                        dist_X = math.fabs(x_predict - evaluate.screenX)
                        dist_list_X.append(dist_X)
                        dist_Y = math.fabs(y_predict - evaluate.screenY)
                        dist_list_Y.append(dist_Y)
        evaluate.missed_num = 0
        evaluate.num = 120

        if len(dist_list) > 2:
            # (X,Y)同时考虑
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('XY\n')
            ma = max(dist_list)
            del points_list[dist_list.index(ma)]
            dist_list.remove(ma)

            mi = min(dist_list)
            del points_list[dist_list.index(mi)]
            dist_list.remove(mi)

            print('critical=', judge_critical)
            valid_points = []
            valid_dists = []
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('realPoint:%s' % '(' + str(evaluate.screenX) + ',' + str(
                    evaluate.screenY) + ')\n')
                fo.write('critical distance is %.2f\n' % judge_critical)
            for i in range(len(dist_list)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list[i]))
                if dist_list[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists.append(dist_list[i])
                    valid_points.append(points_list[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num = len(valid_dists)

            # 只考虑X
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('X\n')
            ma = max(dist_list_X)
            del points_list_X[dist_list_X.index(ma)]
            dist_list_X.remove(ma)

            mi = min(dist_list_X)
            del points_list_X[dist_list_X.index(mi)]
            dist_list_X.remove(mi)

            valid_points_X = []
            valid_dists_X = []
            for i in range(len(dist_list_X)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list_X[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list_X[i]))
                if dist_list_X[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists_X.append(dist_list_X[i])
                    valid_points_X.append(points_list_X[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num_X = len(valid_dists_X)

            # 只考虑Y
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('Y\n')
            ma = max(dist_list_Y)
            del points_list_Y[dist_list_Y.index(ma)]
            dist_list_Y.remove(ma)

            mi = min(dist_list_Y)
            del points_list_Y[dist_list_Y.index(mi)]
            dist_list_Y.remove(mi)

            valid_points_Y = []
            valid_dists_Y = []
            for i in range(len(dist_list_Y)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list_Y[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list_Y[i]))
                if dist_list_Y[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists_Y.append(dist_list_Y[i])
                    valid_points_Y.append(points_list_Y[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num_Y = len(valid_dists_Y)

            evaluate.caculate()
            average_accuracy += evaluate.acceptRatio
            average_accuracy_X += evaluate.acceptRatio_X
            average_accuracy_Y += evaluate.acceptRatio_Y

            output_predictInfo(evaluate, valid_points, valid_points_X, valid_points_Y)

            print('missRatio=', evaluate.missRatio)
            print('acceptRatio=', evaluate.acceptRatio)
            print('acceptRatio_X=', evaluate.acceptRatio_X)
            print('acceptRatio_Y=', evaluate.acceptRatio_Y)
    average_accuracy /= 9
    average_accuracy_X /= 9
    average_accuracy_Y /= 9
    fo = open("../res/compare_predictInfo.txt", "a+")
    msg = '平均准确率=' + str(round(average_accuracy, 2)) + '\nX平均准确率=' + str(
        round(average_accuracy_X, 2)) + '\nY平均准确率=' + str(round(average_accuracy_Y, 2))
    fo.write('%s\n\n' % msg)
    # 关闭打开的文件
    fo.close()


def predict_4(A, B):
    '''
    Z_screenX = a0  * x ^ 2 + a1 * x * y + a2 * y ^ 2 + a3 * x + a4 * y + a5
    Z_screenY = b0  * x ^ 2 + b1 * x * y + b2 * y ^ 2 + b3 * x + b4 * y + b5
    '''
    global average_accuracy
    global average_accuracy_X
    global average_accuracy_Y
    a0 = A[0]
    a1 = A[1]
    a2 = A[2]
    a3 = A[3]
    a4 = A[4]
    a5 = A[5]
    b0 = B[0]
    b1 = B[1]
    b2 = B[2]
    b3 = B[3]
    b4 = B[4]
    b5 = B[5]
    prediction_imgs_dict = read_prediction_images()
    for real_pos, imgs_list in prediction_imgs_dict.items():
        print('len(imgs_list)=', len(imgs_list))
        points_list = []  # 所有预测成功的点
        dist_list = []  # 预测点到真实点的距离
        points_list_X = []  # 所有横坐标预测成功的点
        dist_list_X = []  # 预测点横坐标到真实点的距离
        points_list_Y = []  # 所有纵坐标预测成功的点
        dist_list_Y = []  # 预测点纵坐标到真实点的距离
        real_x, real_y = real_pos.split('_')
        real_x = int(real_x)
        real_y = int(real_y)
        evaluate = Evaluate(real_x, real_y)
        for img in imgs_list:
            # Extract the region of the image that contains the face
            face_image = find_faceimage(img)
            face_landmarks_list = face_recognition.face_landmarks(face_image)
            for face_landmarks in face_landmarks_list:
                right_eye_point = face_landmarks['right_eye']
                right_eye = Eye(face_image, right_eye_point, method='Hough')
                if right_eye is not None:
                    delta_dst = right_eye.top2bottom - CalibrationHelper.top2bottomDist
                    cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y + delta_dst)
                    EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round((cg[1] - CalibrationHelper.ec_y), 2))
                    if EC_CG:
                        print('real_pos=', real_pos)
                        print('predict_eccg:', EC_CG)
                        x = EC_CG[0]
                        y = EC_CG[1]
                        x_predict = a0 * x * x + a1 * x * y + a2 * y * y + a3 * x + a4 * y + a5
                        y_predict = b0 * x * x + b1 * x * y + b2 * y * y + b3 * x + b4 * y + b5
                        print('Z_screenX=%.2f,Z_screenY=%.2f' % (x_predict, y_predict))

                        points_list.append((x_predict, y_predict))
                        points_list_X.append((x_predict, y_predict))
                        points_list_Y.append((x_predict, y_predict))
                        dist = math.sqrt(
                            math.pow((x_predict - evaluate.screenX), 2) + math.pow((y_predict - evaluate.screenY),
                                                                                   2))
                        dist_list.append(dist)
                        dist_X = math.fabs(x_predict - evaluate.screenX)
                        dist_list_X.append(dist_X)
                        dist_Y = math.fabs(y_predict - evaluate.screenY)
                        dist_list_Y.append(dist_Y)
        evaluate.missed_num = 0
        evaluate.num = 120

        if len(dist_list) > 2:
            # (X,Y)同时考虑
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('XY\n')
            ma = max(dist_list)
            del points_list[dist_list.index(ma)]
            dist_list.remove(ma)

            mi = min(dist_list)
            del points_list[dist_list.index(mi)]
            dist_list.remove(mi)

            print('critical=', judge_critical)
            valid_points = []
            valid_dists = []
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('realPoint:%s' % '(' + str(evaluate.screenX) + ',' + str(
                    evaluate.screenY) + ')\n')
                fo.write('critical distance is %.2f\n' % judge_critical)
            for i in range(len(dist_list)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list[i]))
                if dist_list[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists.append(dist_list[i])
                    valid_points.append(points_list[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num = len(valid_dists)

            # 只考虑X
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('X\n')
            ma = max(dist_list_X)
            del points_list_X[dist_list_X.index(ma)]
            dist_list_X.remove(ma)

            mi = min(dist_list_X)
            del points_list_X[dist_list_X.index(mi)]
            dist_list_X.remove(mi)

            valid_points_X = []
            valid_dists_X = []
            for i in range(len(dist_list_X)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list_X[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list_X[i]))
                if dist_list_X[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists_X.append(dist_list_X[i])
                    valid_points_X.append(points_list_X[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num_X = len(valid_dists_X)

            # 只考虑Y
            with open('../res/compare_pointsList.txt', 'a+') as fo:
                fo.write('Y\n')
            ma = max(dist_list_Y)
            del points_list_Y[dist_list_Y.index(ma)]
            dist_list_Y.remove(ma)

            mi = min(dist_list_Y)
            del points_list_Y[dist_list_Y.index(mi)]
            dist_list_Y.remove(mi)

            valid_points_Y = []
            valid_dists_Y = []
            for i in range(len(dist_list_Y)):
                with open('../res/compare_pointsList.txt', 'a+') as fo:
                    fo.write('point:')
                    fo.write(str(points_list_Y[i]))
                    fo.write('   distance=')
                    fo.write(str(dist_list_Y[i]))
                if dist_list_Y[i] <= judge_critical:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('     √\n')
                    valid_dists_Y.append(dist_list_Y[i])
                    valid_points_Y.append(points_list_Y[i])
                else:
                    with open('../res/compare_pointsList.txt', 'a+') as fo:
                        fo.write('\n')
            evaluate.accept_num_Y = len(valid_dists_Y)

            evaluate.caculate()
            average_accuracy += evaluate.acceptRatio
            average_accuracy_X += evaluate.acceptRatio_X
            average_accuracy_Y += evaluate.acceptRatio_Y

            output_predictInfo(evaluate, valid_points, valid_points_X, valid_points_Y)

            print('missRatio=', evaluate.missRatio)
            print('acceptRatio=', evaluate.acceptRatio)
            print('acceptRatio_X=', evaluate.acceptRatio_X)
            print('acceptRatio_Y=', evaluate.acceptRatio_Y)
    average_accuracy /= 9
    average_accuracy_X /= 9
    average_accuracy_Y /= 9
    fo = open("../res/compare_predictInfo.txt", "a+")
    msg = '平均准确率=' + str(round(average_accuracy, 2)) + '\nX平均准确率=' + str(
        round(average_accuracy_X, 2)) + '\nY平均准确率=' + str(round(average_accuracy_Y, 2))
    fo.write('%s\n\n' % msg)
    # 关闭打开的文件
    fo.close()


def output_predictInfo(evaluate, points_list, points_list_X, points_list_Y):
    """将预测结果写入文件

    :param evaluate: 评估结果
    :param points_list: 注视点坐标列表
    :return: None
    """
    points = ''
    for row in range(len(points_list)):
        points += ',('+str(points_list[row][0])+','+str(points_list[row][1])+')'
    points_X = ''
    for row in range(len(points_list_X)):
        points_X += ',('+str(points_list_X[row][0])+','+str(points_list_X[row][1])+')'
    points_Y = ''
    for row in range(len(points_list_Y)):
        points_Y += ',(' + str(points_list_Y[row][0]) + ',' + str(points_list_Y[row][1]) + ')'
    # input dict into txt
    fo = open("../res/compare_predictInfo.txt", "a+")

    fo.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % (('('+str(evaluate.screenX)+','+str(evaluate.screenY)+')').ljust(15), str(evaluate.missRatio).ljust(8), str(evaluate.acceptRatio).ljust(8), str(evaluate.acceptRatio_X).ljust(8), str(evaluate.acceptRatio_Y).ljust(8), str(evaluate.missed_num).ljust(8), str(evaluate.accept_num).ljust(8), str(evaluate.accept_num_X).ljust(8), str(evaluate.accept_num_Y).ljust(8), str(evaluate.num).ljust(8), points, points_X, points_Y))
    # 关闭打开的文件
    fo.close()

def displayTitle():
    fo = open("../res/compare_predictInfo.txt", "a+")

    fo.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % ('屏幕坐标'.ljust(12), '错失率'.ljust(8), '接受率'.ljust(8), 'X接受率'.ljust(8), 'Y接受率'.ljust(8), '错失数'.ljust(8), '接受数'.ljust(8), 'X接受数'.ljust(8), 'Y接受数'.ljust(8), '总数'.ljust(8), '接受预测点坐标', 'X接受预测点坐标', 'Y接受预测点坐标'))
    # 关闭打开的文件
    fo.close()


def Algorithm_1():
    with open("../res/compare_predictInfo.txt", "a+") as fo:
        fo.write('Algorithm_1\n')
    displayTitle()
    getPointPos()
    for epoch_dict in read_calibration_images():
        eachEpochlist = [epoch_dict['0'], epoch_dict['1'], epoch_dict['2'], epoch_dict['3'], epoch_dict['4'], epoch_dict['6'], epoch_dict['7'], epoch_dict['8'], epoch_dict['9']]
        # print(len(eachEpochlist))
        for i in range(len(eachEpochlist)):
            EC = []
            CG = []
            top2bottom_list = []
            temp_ec = None
            for img in eachEpochlist[i]:
                face_image = find_faceimage(img)
                face_landmarks_list = face_recognition.face_landmarks(face_image)
                for face_landmarks in face_landmarks_list:
                    # right_eye_point has 6 points
                    right_eye_point = face_landmarks['right_eye']
                    print('right_eye', face_landmarks['right_eye'])
                    right_eye = Eye(face_image, right_eye_point)
                    if right_eye is not None:
                        if i == 0:
                            # 如果标定点是屏幕中点，则需要计算EC
                            temp_ec = right_eye.center

                        cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y)
                        print('each_cg=', cg)
                        if cg is None or cg[0] is None or cg[1] is None:
                            break
                        if i == 0:
                            # temp_ec = cg
                            EC.append(temp_ec)
                        CG.append(cg)
                        temp_dst = right_eye.top2bottom
                        top2bottom_list.append(temp_dst)

                        # delta = 0 if i == 0 else temp_dst - CalibrationHelper.top2bottomDist
                        # each_cg = (cg[0], cg[1] + delta)
                        # each_ec = temp_ec if i == 0 else (CalibrationHelper.ec_x, CalibrationHelper.ec_y)
                        # ec_cg = (round((each_cg[0] - each_ec[0]), 2), round((each_cg[1] - each_ec[1]), 2))
                        # print('-------------------testECCG----------------:', str(ec_cg))
            p = 0
            for d in top2bottom_list:
                p += d
            avg_dst = p / len(top2bottom_list)
            print('avg_dst=', avg_dst)
            if i == 0:
                x = 0
                y = 0
                for t in EC:
                    x += t[0]
                    y += t[1]
                ec = (x / len(EC), y / len(EC))
                print('avg_ec=', ec)
                CalibrationHelper.ec_x = ec[0]
                CalibrationHelper.ec_y = ec[1]
                CalibrationHelper.top2bottomDist = avg_dst
            x = 0
            y = 0
            for t in CG:
                x += t[0]
                y += t[1]
            delta_dst = avg_dst - CalibrationHelper.top2bottomDist
            cg = (x / len(CG), y / len(CG) + delta_dst)
            print('avg_cg=', cg)

            EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round((cg[1] - CalibrationHelper.ec_y), 2))
            print('EC_CG:', EC_CG)

            ECCG_list.append(EC_CG)
    get_relationship_eye_screenpoint()
    A, B = caculateCoeficiente()
    predict_1(A, B)
    # rfc_x, rfc_y = caculateCoeficiente_SVR()
    # predict_1(rfc_x, rfc_y)

def Algorithm_2():
    with open("../res/compare_predictInfo.txt", "a+") as fo:
        fo.write('Algorithm_2\n')
    displayTitle()
    getPointPos()
    for epoch_dict in read_calibration_images():
        eachEpochlist = [epoch_dict['0'], epoch_dict['1'], epoch_dict['2'], epoch_dict['3'], epoch_dict['4'], epoch_dict['6'], epoch_dict['7'], epoch_dict['8'], epoch_dict['9']]
        # print(len(eachEpochlist))
        for i in range(len(eachEpochlist)):
            EC = []
            CG = []
            ratio = []
            top2bottom_list = []
            temp_ec = None
            for img in eachEpochlist[i]:
                face_image = find_faceimage(img)
                face_landmarks_list = face_recognition.face_landmarks(face_image)
                for face_landmarks in face_landmarks_list:
                    # right_eye_point has 6 points
                    right_eye_point = face_landmarks['right_eye']
                    print('right_eye', face_landmarks['right_eye'])
                    right_eye = Eye(face_image, right_eye_point)
                    if right_eye is not None:
                        if i == 0:
                            # 如果标定点是屏幕中点，则需要计算EC
                            temp_ec = right_eye.center

                        cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y)
                        r = right_eye.pupil.iris_eye_ratio
                        print('each_cg=', cg)
                        if cg is None or cg[0] is None or cg[1] is None:
                            break
                        if i == 0:
                            # temp_ec = cg
                            EC.append(temp_ec)
                        CG.append(cg)
                        ratio.append(r)
                        temp_dst = right_eye.top2bottom
                        top2bottom_list.append(temp_dst)

                        # delta = 0 if i == 0 else temp_dst - CalibrationHelper.top2bottomDist
                        # each_cg = (cg[0], cg[1] + delta)
                        # each_ec = temp_ec if i == 0 else (CalibrationHelper.ec_x, CalibrationHelper.ec_y)
                        # ec_cg = (round((each_cg[0] - each_ec[0]), 2), round((each_cg[1] - each_ec[1]), 2))
                        # print('-------------------testECCG----------------:', str(ec_cg))
            p = 0
            for d in top2bottom_list:
                p += d
            avg_dst = p / len(top2bottom_list)
            print('avg_dst=', avg_dst)
            avg_ratio = 0
            for r in ratio:
                avg_ratio += r
            avg_ratio /= len(ratio)
            print('avg_ratio=', avg_ratio)
            if i == 0:
                x = 0
                y = 0
                for t in EC:
                    x += t[0]
                    y += t[1]
                ec = (x / len(EC), y / len(EC))
                print('avg_ec=', ec)
                CalibrationHelper.ec_x = ec[0]
                CalibrationHelper.ec_y = ec[1]
                CalibrationHelper.top2bottomDist = avg_dst
            x = 0
            y = 0
            for t in CG:
                x += t[0]
                y += t[1]
            delta_dst = avg_dst - CalibrationHelper.top2bottomDist
            cg = (x / len(CG), y / len(CG) + delta_dst)
            print('avg_cg=', cg)

            EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round(avg_ratio, 2))
            print('EC_CG:', EC_CG)

            ECCG_list.append(EC_CG)
    get_relationship_eye_screenpoint()
    A, B = caculateCoeficiente()
    predict_2(A, B)

def Algorithm_3():
    with open("../res/compare_predictInfo.txt", "a+") as fo:
        fo.write('Algorithm_3\n')
    displayTitle()
    getPointPos()
    for epoch_dict in read_calibration_images():
        eachEpochlist = [epoch_dict['0'], epoch_dict['1'], epoch_dict['2'], epoch_dict['3'], epoch_dict['4'], epoch_dict['6'], epoch_dict['7'], epoch_dict['8'], epoch_dict['9']]
        # print(len(eachEpochlist))
        for i in range(len(eachEpochlist)):
            EC = []
            CG = []
            top2bottom_list = []
            temp_ec = None
            distance_list = []
            for img in eachEpochlist[i]:
                distance1 = 0  # 瞳孔圆方程与下眼眶抛物线交点间的距离
                distance2 = 0  # 瞳孔圆方程与上眼眶抛物线交点间的距离
                face_image = find_faceimage(img)
                face_landmarks_list = face_recognition.face_landmarks(face_image)
                for face_landmarks in face_landmarks_list:
                    # right_eye_point has 6 points
                    right_eye_point = face_landmarks['right_eye']
                    print('right_eye', face_landmarks['right_eye'])
                    right_eye = Eye(face_image, right_eye_point)

                    if right_eye is not None:

                        # 拟合抛物线
                        print("Get prabola:")
                        X1 = np.array(
                            [right_eye_point[0][0], right_eye_point[1][0], right_eye_point[2][0],
                             right_eye_point[3][0]])
                        Y1 = np.array(
                            [right_eye_point[0][1], right_eye_point[1][1], right_eye_point[2][1],
                             right_eye_point[3][1]])
                        X2 = np.array(
                            [right_eye_point[0][0], right_eye_point[4][0], right_eye_point[5][0],
                             right_eye_point[3][0]])
                        Y2 = np.array(
                            [right_eye_point[0][1], right_eye_point[4][1], right_eye_point[5][1],
                             right_eye_point[3][1]])
                        para_args1, para_args2 = solution(X1, Y1, X2, Y2)

                        c_x = right_eye.pupil.x + right_eye.origin[0]
                        c_y = right_eye.pupil.y + right_eye.origin[1]
                        radius = right_eye.pupil.radius
                        print("c_x= ", c_x)
                        print("c_y= ", c_y)
                        print("redius= ", radius)
                        circle_args_list = [1, 1, -(2 * c_x), -(2 * c_y), c_x ** 2 + c_y ** 2 - radius ** 2]  # 圆方程一般式系数
                        print("circle equation: x^2+y^2+", circle_args_list[2], "x+", circle_args_list[3], "y+",
                              circle_args_list[4])

                        # 求瞳孔圆方程与上下眼眶抛物线方程的交点
                        num1 = 0  # 瞳孔圆方程与下眼眶抛物线交点数量
                        num2 = 0  # 瞳孔圆方程与上眼眶抛物线交点数量
                        for j in range(2):
                            if j == 0:
                                args1 = get_args(circle_args_list, para_args1)
                                # print(args1)
                                num1, x1 = cal_quartic_ik(args1)
                                print("num1: ", num1)
                                print("x1: ", x1)
                                '''
                                for point1 in x1:
                                    x1_tmp.append(point1)'''
                                if x1 != 0:
                                    result1 = get_result(args1, x1)
                                    print("result1: ", result1)
                            elif j == 1:
                                args2 = get_args(circle_args_list, para_args2)
                                # print(args2)
                                num2, x2 = cal_quartic_ik(args2)
                                print("num2: ", num2)
                                print("x2: ", x2)
                                '''
                                for point2 in x2:
                                    x2_tmp.append(point2)'''
                                if x2 != 0:
                                    result2 = get_result(args2, x2)
                                    print("result2: ", result2)
                        if num1 == 2:
                            distance1 = math.sqrt(
                                (result1[0][0] - result1[1][0]) ** 2 + (result1[0][1] - result1[1][1]) ** 2)
                        elif num1 == 1 | num1 == 0:
                            distance1 = 0
                        if num2 == 2:
                            distance2 = math.sqrt(
                                (result2[0][0] - result2[1][0]) ** 2 + (result2[0][1] - result2[1][1]) ** 2)
                        elif num2 == 1 | num2 == 0:
                            distance2 = 0

                        if i == 0:
                            # 如果标定点是屏幕中点，则需要计算EC
                            temp_ec = right_eye.center

                        cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y)
                        print('each_cg=', cg)
                        if cg is None or cg[0] is None or cg[1] is None:
                            break
                        if i == 0:
                            # temp_ec = cg
                            EC.append(temp_ec)
                        CG.append(cg)
                        temp_dst = right_eye.top2bottom
                        top2bottom_list.append(temp_dst)
                        distance_list.append(distance1 - distance2)

                        # delta = 0 if i == 0 else temp_dst - CalibrationHelper.top2bottomDist
                        # each_cg = (cg[0], cg[1] + delta)
                        # each_ec = temp_ec if i == 0 else (CalibrationHelper.ec_x, CalibrationHelper.ec_y)
                        # ec_cg = (round((each_cg[0] - each_ec[0]), 2), round((each_cg[1] - each_ec[1]), 2))
                        # print('-------------------testECCG----------------:', str(ec_cg))
            p = 0
            for d in top2bottom_list:
                p += d
            avg_dst = p / len(top2bottom_list)
            print('avg_dst=', avg_dst)
            q = 0
            for j in distance_list:
                q += j
            avg_distance1_2 = q / len(distance_list)
            delta_distance_list.append(avg_distance1_2)
            if i == 0:
                x = 0
                y = 0
                for t in EC:
                    x += t[0]
                    y += t[1]
                ec = (x / len(EC), y / len(EC))
                print('avg_ec=', ec)
                CalibrationHelper.ec_x = ec[0]
                CalibrationHelper.ec_y = ec[1]
                CalibrationHelper.top2bottomDist = avg_dst
            x = 0
            y = 0
            for t in CG:
                x += t[0]
                y += t[1]
            delta_dst = avg_dst - CalibrationHelper.top2bottomDist
            cg = (x / len(CG), y / len(CG) + delta_dst)
            print('avg_cg=', cg)

            EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round((cg[1] - CalibrationHelper.ec_y), 2))
            print('EC_CG:', EC_CG)

            ECCG_list.append(EC_CG)
    get_relationship_eye_screenpoint()
    A, B = caculateCoeficiente()
    C = caculateCoeficiente_distance()
    predict_3(A, B, C)

def Algorithm_4():
    with open("../res/compare_predictInfo.txt", "a+") as fo:
        fo.write('Algorithm_4\n')
    displayTitle()
    getPointPos()
    for epoch_dict in read_calibration_images():
        eachEpochlist = [epoch_dict['0'], epoch_dict['1'], epoch_dict['2'], epoch_dict['3'], epoch_dict['4'], epoch_dict['6'], epoch_dict['7'], epoch_dict['8'], epoch_dict['9']]
        # print(len(eachEpochlist))
        for i in range(len(eachEpochlist)):
            EC = []
            CG = []
            top2bottom_list = []
            temp_ec = None
            num = 0
            for img in eachEpochlist[i]:
                face_image = find_faceimage(img)
                face_landmarks_list = face_recognition.face_landmarks(face_image)
                for face_landmarks in face_landmarks_list:
                    # right_eye_point has 6 points
                    right_eye_point = face_landmarks['right_eye']
                    print('right_eye', face_landmarks['right_eye'])
                    right_eye = Eye(face_image, right_eye_point, method='Hough')
                    if right_eye is not None:
                        num += 1
                        if i == 0:
                            # 如果标定点是屏幕中点，则需要计算EC
                            temp_ec = right_eye.center

                        cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y)
                        print('each_cg=', cg)
                        if cg is None or cg[0] is None or cg[1] is None:
                            break
                        if i == 0:
                            # temp_ec = cg
                            EC.append(temp_ec)
                        CG.append(cg)
                        temp_dst = right_eye.top2bottom
                        top2bottom_list.append(temp_dst)
            print('num= ', num)
            p = 0
            for d in top2bottom_list:
                p += d
            avg_dst = p / len(top2bottom_list)
            print('avg_dst=', avg_dst)
            if i == 0:
                x = 0
                y = 0
                for t in EC:
                    x += t[0]
                    y += t[1]
                ec = (x / len(EC), y / len(EC))
                print('avg_ec=', ec)
                CalibrationHelper.ec_x = ec[0]
                CalibrationHelper.ec_y = ec[1]
                CalibrationHelper.top2bottomDist = avg_dst
            x = 0
            y = 0
            for t in CG:
                x += t[0]
                y += t[1]
            delta_dst = avg_dst - CalibrationHelper.top2bottomDist
            cg = (x / len(CG), y / len(CG) + delta_dst)
            print('avg_cg=', cg)

            EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round((cg[1] - CalibrationHelper.ec_y), 2))
            print('EC_CG:', EC_CG)

            ECCG_list.append(EC_CG)
    get_relationship_eye_screenpoint()
    A, B = caculateCoeficiente()
    predict_4(A, B)


Algorithm_1()
# Algorithm_2()
# Algorithm_3()
# Algorithm_4()