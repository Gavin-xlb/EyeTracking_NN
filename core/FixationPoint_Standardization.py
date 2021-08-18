from tkinter import *
import cv2
import face_recognition
import os
import tkinter.messagebox
import numpy as np
import _thread
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from PIL import ImageTk, Image

from core.CalibrationHelper import CalibrationHelper
from core.Distortion import Distortion
from core.Draw3D import Draw3D

from sklearn.metrics import precision_score
from core import Surface_fitting
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from core.ScreenHelper import ScreenHelper
from core.Config import Config
from gaze_tracking.gaze_tracking import GazeTracking

outdir = r'D:\EyeTracking_NN-master\image'
screenhelper = ScreenHelper()

btn_index = 0
BTN_ALL = Config.CALIBRATION_POINTS_NUM  # the number of points
ROW_POINT = Config.CALIBRATION_POINTS_ROW
COL_POINT = Config.CALIBRATION_POINTS_COL
relationship_eye_screenpoint = {}  # EC-CG --> (screen_x, screen_y)
point_list = []  # 9 points' coordinates
btn_list = []  # 9 buttons
ECCG_list = []  # 9 points' EC-CG
CG_list = []
gaze = GazeTracking()
img_open = None
img = None
ec = None


def f(i):
    return i % COL_POINT


def g(i):
    return i // ROW_POINT


def create_btn(cap, frame, screen_width, screen_height):
    """产生按钮

    :param cap: 相机对象
    :param frame: 画布对象
    :param screen_width: 屏幕宽度
    :param screen_height: 屏幕高度
    :return:
    """
    global img_open
    global img
    global screenhelper
    global btn_index
    PPI = screenhelper.getPPI()
    # 创建一个Canvas，设置其背景色为白色
    # cv = Canvas(root, bg='white', height=screen_height, width=screen_width)
    d = Config.CALIBRATION_POINTS_INTERVAL_EDGE
    w = Config.CALIBRATION_POINTS_WIDTH
    h = Config.CALIBRATION_POINTS_HEIGHT
    center_index = Config.CALIBRATION_POINTS_NUM // 2
    iscenter = False
    print('createBtn...')
    if btn_index == 0:
        iscenter = True
        x = f(center_index) * (screen_width - 2 * d) / (COL_POINT - 1) + d
        y = g(center_index) * (screen_height - 2 * d) / (ROW_POINT - 1) + d
    else:
        iscenter = False
        x = f(btn_index-1) * (screen_width - 2 * d) / (COL_POINT - 1) + d
        y = g(btn_index-1) * (screen_height - 2 * d) / (ROW_POINT - 1) + d
    point_list.append((x, y))
    img_open = Image.open('../res/button_img.jpg')
    img = ImageTk.PhotoImage(img_open)
    btn = Button(frame, width=w, height=h, image=img)
    btn['command'] = lambda: shot(cap, frame, btn_list, screen_width, screen_height, iscenter)
    btn.place(x=(x - w / 2), y=(y - h / 2))
    frame.pack()
    btn_list.append(btn)
    return point_list, btn_list


def adaptive_histogram_equalization(gray_image):
    """对一幅灰度图像进行直方图均衡化

    :param gray_image: 灰度图像
    :return: 限制对比度的自适应阈值均衡化后的图像
    """
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(2.0, (8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(gray_image)
    return dst


def get_relationship_eye_screenpoint():
    """将EC-CG向量和屏幕坐标的关系写入文件

    :return: None
    """
    # input dict into txt
    fo = open("../res/ECCG_screenPoint.txt", "w")
    for i in range(len(ECCG_list)):
        fo.write('%s:%s\n' % (ECCG_list[i], point_list[i]))
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
    if len(ECCG_list) < BTN_ALL:
        return [], []

    X = [x[0] for x in ECCG_list]
    Y = [x[1] for x in ECCG_list]
    Z_screenX = [x[0] for x in point_list]
    Z_screenY = [x[1] for x in point_list]
    A = Surface_fitting.matching_3D(X, Y, Z_screenX)
    B = Surface_fitting.matching_3D(X, Y, Z_screenY)

    # 离散点和拟合函数可视化
    # 拟合注视点横坐标
    Draw3D.drawMap(X, Y, Z_screenX, 'x')
    # Draw3D.drawSurfaceMap(A, 'x')
    # Draw3D.drawWireFrameMap(A, 'x')
    # 拟合注视点纵坐标
    Draw3D.drawMap(X, Y, Z_screenY, 'y')
    # Draw3D.drawSurfaceMap(B, 'y')
    # Draw3D.drawWireFrameMap(B, 'y')

    return A, B


# SVR
def caculateCoeficiente_SVR():
    """SVR算法拟合函数

    :return: clf_x, clf_y
    """
    print("strat fitting.....")
    if len(ECCG_list) < BTN_ALL:
        return None, None
    eccgpoint = np.array([x for x in ECCG_list])
    Z_screenX = np.array([x[0] for x in point_list])
    Z_screenY = np.array([x[1] for x in point_list])
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
    if len(ECCG_list) < BTN_ALL:
        return None, None
    eccgpoint = np.array([x for x in ECCG_list])
    Z_screenX = np.array([x[0] for x in point_list])
    Z_screenY = np.array([x[1] for x in point_list])
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


def find_max_region(mask_sel):
    """二值化图像找最大连通域

    :param mask_sel: 二值化图像
    :return: 最大连通域
    """
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel


def shot(video_capture, frame_WIN, btn_list, screen_width, screen_height, iscenter):
    """视线标定

    :param video_capture: 相机对象
    :param frame_WIN: 画布对象
    :param btn_list: 按钮列表
    :param screen_width: 屏幕宽度
    :param screen_height: 屏幕高度
    :param iscenter: 是否是屏幕中心点
    :return:
    """
    # _thread.start_new_thread(interrupt_mainThread, ())
    # global index
    global btn_index, ec

    print('photo ' + str(btn_index) + ' is proccessing...')
    # the number of frames per eye_point
    frame_num = 5
    frame_interval = 6
    i = 0
    frame = []
    small_frame = []
    pre_frame = []

    while i < (frame_num - 1) * frame_interval + 1:
        if i % frame_interval == 0:
            # Grab a single frame of video
            ret, f = video_capture.read()
            # f = Distortion.dedistortion(f)
            # pre_frame.append(f)
            # f = adaptive_histogram_equalization(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            frame.append(f)

            # Resize frame of video to 1/5 size for faster face detection processing
            s = cv2.resize(f, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

            small_frame.append(s)
        i = i + 1
    j = 0
    # the number of frame which is successfully detected
    num = 0
    EC = []
    CG = []
    top2bottom_list = []
    temp_ec = None
    face_image = None
    # frame2 = None
    while j < frame_num:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame[j], model='cnn')
        # Display the results
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5

            # Extract the region of the image that contains the face
            face_image = frame[j][top:bottom, left:right]

            # cv2.imwrite(os.path.join(outdir, 'face_image' + str(btn_index) + '_' + str(i) + '.jpg'),
            #             face_image)
            face_landmarks_list = face_recognition.face_landmarks(face_image)
            for face_landmarks in face_landmarks_list:
                for facial_feature in face_landmarks.keys():
                    # get right_eye's point
                    if facial_feature == 'right_eye':
                        # right_eye_point has 6 points
                        right_eye_point = face_landmarks[facial_feature]
                        print(facial_feature, face_landmarks[facial_feature])

                        gaze.find_iris(face_image, right_eye_point, 1, 1)
                        # 画出眼睛的6个点
                        for point in right_eye_point:
                            cv2.circle(face_image, point, 0, (0, 0, 255), 3)

                        right_eye = gaze.eye_right
                        if right_eye is not None:
                            if iscenter:
                                # 如果标定点是屏幕中点，则需要计算EC
                                temp_ec = right_eye.center

                            cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y)
                            if cg is None or cg[0] is None or cg[1] is None:
                                break
                            if iscenter:
                                # temp_ec = cg
                                EC.append(temp_ec)
                            CG.append(cg)
                            temp_dst = right_eye.top2bottom
                            top2bottom_list.append(temp_dst)

                            num += 1

                            break
        j += 1
    # num为虹膜二值化成功的次数，并非人眼检测成功的次数
    print('num=', num)
    # shot successfully
    if (j == frame_num) and (num > 0):
        p = 0
        for d in top2bottom_list:
            p += d
        avg_dst = p / num
        print('avg_dst=', avg_dst)
        if iscenter:
            x = 0
            y = 0
            for t in EC:
                x += t[0]
                y += t[1]
            ec = (x / num, y / num)
            print('ec=', ec)
            CalibrationHelper.ec_x = ec[0]
            CalibrationHelper.ec_y = ec[1]
            CalibrationHelper.top2bottomDist = avg_dst
        x = 0
        y = 0
        for t in CG:
            x += t[0]
            y += t[1]
        delta_dst = avg_dst - CalibrationHelper.top2bottomDist
        cg = (x / num, y / num + delta_dst)
        print('cg=', cg)

        EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round((cg[1] - CalibrationHelper.ec_y), 2))
        print('EC_CG:', EC_CG)

        ECCG_list.append(EC_CG)

        frame1 = gaze.annotated_frame(face_image, delta_dst)
        cv2.imshow('frame_calibration', frame1)
        cv2.imwrite('../image/calibration/' + str(btn_index) + str(EC_CG) + '.jpg', frame1)

        # 成功录入btn_index才会+1
        if btn_index > Config.CALIBRATION_POINTS_NUM // 2 + 1:
            btn_list[btn_index - 1].destroy()
        else:
            btn_list[btn_index].destroy()  # delete button which has been clicked just now
        btn_index += 1
        if btn_index == Config.CALIBRATION_POINTS_NUM // 2 + 1:
            btn_index += 1
        if btn_index <= BTN_ALL:
            create_btn(video_capture, frame_WIN, screen_width, screen_height)
        if btn_index > BTN_ALL:
            result = tkinter.messagebox.showinfo('提示', '注定点标定结束!')
            # get the relationship between eye and screenpoint
            get_relationship_eye_screenpoint()
            if result == tkinter.messagebox.OK:
                tkinter.messagebox.showinfo('提示', '视线追踪开始!')
    else:
        tkinter.messagebox.showinfo('提示', '未成功录入此注视点，请重新点击!')
