from tkinter import *
import cv2
import face_recognition
import os
import tkinter.messagebox
import numpy as np
import _thread

from PIL import ImageTk, Image

from core import Surface_fitting
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from core.ScreenHelper import ScreenHelper

outdir = r'D:\python_Projects\eyeTracking_NN\image'
screenhelper = ScreenHelper()

btn_index = 0
BTN_ALL = 25 #the number of points
ROW_POINT = 5
COL_POINT = 5
relationship_eye_screenpoint = {} #EC-CG --> (screen_x, screen_y)
point_list = []  # 9 points' coordinates
btn_list = []  # 9 buttons
ECCG_list = []  # 9 points' EC-CG

img_open = None
img = None

def f(i):
    return i % COL_POINT


def g(i):
    return i // ROW_POINT


def create_btn(cap, frame, screen_width, screen_height):
    global img_open
    global img
    global screenhelper
    PPI = screenhelper.getPPI()
    # 创建一个Canvas，设置其背景色为白色
    # cv = Canvas(root, bg='white', height=screen_height, width=screen_width)
    d = 50
    w = 20
    h = 20

    x = f(btn_index) * (screen_width - 2 * d) / (COL_POINT-1) + d
    y = g(btn_index) * (screen_height - 2 * d) / (ROW_POINT-1) + d
    point_list.append((x, y))
    img_open = Image.open('../res/button_img.jpg')
    img = ImageTk.PhotoImage(img_open)
    btn = Button(frame, width=w, height=h, command=lambda: shot(cap, frame, btn_list, screen_width, screen_height), image=img)
    btn.pack()
    btn.place(x=(x - w / 2), y=(y - h / 2) - 20)
    frame.pack()
    btn_list.append(btn)
    return point_list, btn_list

    # for i in range(9):
    #     x = f(i) * (screen_width - 2 * d) / 2 + d
    #     y = g(i) * (screen_height - 2 * d) / 2 + d
    #     point_list.append((x, y))
    #     if i == 0:
    #         btn = Button(frame, width=w, height=h, text=i, bg='yellow', command=lambda: shot(cap, frame, btn_list))
    #     else:
    #         btn = Button(frame, width=w, height=h, text=i, bg='white', command=lambda: shot(cap, frame, btn_list))
    #     btn.pack()
    #     btn.place(x=(x - w / 2), y=(y - h / 2) - 20)
    #     frame.pack()
    #     btn_list.append(btn)
    # return point_list, btn_list


def rectangle_eye(eye_point_list):
    x_min = eye_point_list[0][0]
    x_max = eye_point_list[0][0]
    y_min = eye_point_list[0][1]
    y_max = eye_point_list[0][1]
    for point in eye_point_list:
        if point[0] < x_min:
            x_min = point[0]
        if point[0] > x_max:
            x_max = point[0]
        if point[1] < y_min:
            y_min = point[1]
        if point[1] > y_max:
            y_max = point[1]
    return x_min, x_max, y_min, y_max


def get_relationship_eye_screenpoint():
    # input dict into txt
    fo = open("../res/ECCG_screenPoint.txt", "w")
    for i in range(len(ECCG_list)):
        fo.write('%s:%s\n' % (ECCG_list[i], point_list[i]))
    # 关闭打开的文件
    fo.close()

#最小二乘
def caculateCoeficiente():
    '''
        Z_screenX = a0  * x ^ 2 + a1 * x * y + a2 * y ^ 2 + a3 * x + a4 * y + a5
        Z_screenY = b0  * x ^ 2 + b1 * x * y + b2 * y ^ 2 + b3 * x + b4 * y + b5
    '''
    if len(ECCG_list) < 9:
        return [], []

    X = [x[0] for x in ECCG_list]
    Y = [x[1] for x in ECCG_list]
    Z_screenX = [x[0] for x in point_list]
    Z_screenY = [x[1] for x in point_list]
    A = Surface_fitting.matching_3D(X, Y, Z_screenX)
    B = Surface_fitting.matching_3D(X, Y, Z_screenY)

    return A, B


def caculateCoeficiente_SVR():
    if len(ECCG_list) < 9:
        return None, None
    eccgpoint = np.array([x for x in ECCG_list])
    Z_screenX = np.array([x[0] for x in point_list])
    Z_screenY = np.array([x[1] for x in point_list])

    clf_x = SVR(kernel='poly', degree=4, gamma="auto", coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
              cache_size=200, verbose=False, max_iter=- 1)
    clf_y = SVR(kernel='poly', degree=4, gamma="auto", coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
              cache_size=200, verbose=False, max_iter=- 1)
    clf_x.fit(eccgpoint, Z_screenX)
    # print(clf_x.predict(eccgpoint))
    clf_y.fit(eccgpoint, Z_screenY)
    # print(clf_y.predict(eccgpoint))
    return clf_x, clf_y


def find_max_region(mask_sel):
    __, contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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


def interrupt_mainThread():
    try:
        _thread.interrupt_main()
    except KeyboardInterrupt:
        print('Exception:KeyboardInterrupt')


def shot(video_capture, frame_WIN, btn_list, screen_width, screen_height):
    # _thread.start_new_thread(interrupt_mainThread, ())
    # global index
    global btn_index

    print('photo ' + str(btn_index) + ' is proccessing...')
    # the number of frames per eye_point
    frame_num = 5
    i = 0
    frame = []
    small_frame = []

    while i < frame_num :
        # Grab a single frame of video
        ret, f = video_capture.read()
        frame.append(f)
        # Resize frame of video to 1/5 size for faster face detection processing
        s = cv2.resize(frame[i], (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        small_frame.append(s)
        i = i + 1

    i = 0
    # the number of frame which is successfully detected
    num = 0
    EC = []
    CG = []
    while i < frame_num:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame[i])

        # Display the results
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5

            # Extract the region of the image that contains the face
            face_image = frame[i][top:bottom, left:right]
            # cv2.imwrite(os.path.join(outdir, 'face_image' + str(btn_index) + '_' + str(i) + '.jpg'),
            #             face_image)
            face_landmarks_list = face_recognition.face_landmarks(face_image)
            for face_landmarks in face_landmarks_list:
                for facial_feature in face_landmarks.keys():
                    # get right_eye's point
                    if facial_feature == 'right_eye':

                        right_eye_point = face_landmarks[facial_feature]
                        print(facial_feature, face_landmarks[facial_feature])
                        # rectangle the location of right_eye
                        right_eye_location = rectangle_eye(right_eye_point)
                        # minor changes according to experience
                        right_eye_location_change = (
                            right_eye_location[0] + 2, right_eye_location[1] + 4, right_eye_location[2] - 2,
                            right_eye_location[3] + 1)
                        right_eye_image = face_image[right_eye_location_change[2]:right_eye_location_change[3],
                                          right_eye_location_change[0]:right_eye_location_change[1]]
                        right_eye_height = right_eye_image.shape[0]
                        right_eye_width = right_eye_image.shape[1]
                        magnify_times = 5
                        magnify_right_eye_img = cv2.resize(right_eye_image, (0, 0), fx=magnify_times, fy=magnify_times, interpolation=cv2.INTER_LINEAR)
                        magnify_right_eye_img_height = magnify_right_eye_img.shape[0]
                        magnify_right_eye_img_width = magnify_right_eye_img.shape[1]

                        ec_xInpixel = magnify_right_eye_img_width / 2
                        ec_yInpixel = magnify_right_eye_img_height / 2
                        PPI = screenhelper.getPPI()
                        EC.append((ec_xInpixel / magnify_times, ec_yInpixel / magnify_times))

                        # EC.append((round(right_eye_width / 2), round(right_eye_height / 2)))
                        # print('right_eye_location_change:', right_eye_location_change)
                        print('EC:', EC)

                        cv2.imwrite(os.path.join(outdir, 'right_eye_image' + str(btn_index) + '_' + str(i) + '.jpg'),
                                    magnify_right_eye_img)
                        # Binaryzation processing to right eye
                        gray_right_eye_image = cv2.cvtColor(magnify_right_eye_img, cv2.COLOR_BGR2GRAY)
                        shape = gray_right_eye_image.shape
                        median_x = round(shape[0] / 2)
                        median_y = round(shape[1] / 2)
                        print('shape', shape)
                        print(median_x, median_y)
                        # get the mean to the 9 central pixels of right_eye
                        mean_eye = np.mean(gray_right_eye_image[(median_x-1):(median_x+1), (median_y-1):(median_y+1)])
                        print('mean:', mean_eye)
                        # mean_eye = gray_right_eye_image[median_x][median_y]  # get the mean to the central pixel of right_eye
                        # print('mean_eye: ', mean_eye)
                        ret, right_eye_binaryzation = cv2.threshold(gray_right_eye_image, mean_eye,
                                                                    255,
                                                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                        mask_sel = find_max_region(right_eye_binaryzation)
                        mu = cv2.moments(mask_sel, False)
                        mc_x = mu['m10'] / mu['m00']
                        mc_y = mu['m01'] / mu['m00']

                        mc = (mc_x / magnify_times, mc_y / magnify_times)
                        # print('mc:', mc)
                        cv2.imwrite(os.path.join(outdir, 'right_eye_binaryzation' + str(btn_index) + '_' + str(i) + '.jpg'),
                                    right_eye_binaryzation)
                        # cv2.circle(mask_sel, mc, 0, (0, 0, 255), 5)
                        cv2.imwrite(os.path.join(outdir, 'right_eye_binaryzation_mc' + str(btn_index) + '_' + str(i) + '.jpg'),
                                    mask_sel)

                        CG.append(mc)
                        print('CG:', CG)

                        num += 1
        i += 1
    # shot successfully
    if (i == frame_num) and (num != 0):
        x = 0
        y = 0
        for t in EC:
            x += t[0]
            y += t[1]
        ec = (x / num, y / num)
        x = 0
        y = 0
        for t in CG:
            x += t[0]
            y += t[1]
        cg = (x / num, y / num)

        EC_CG = (cg[0] - ec[0], cg[1] - ec[1])
        print('EC_CG:', EC_CG)

        ECCG_list.append(EC_CG)

        # 成功录入btn_index才会+1
        btn_list[btn_index].destroy()  #delete button which has been clicked just now
        btn_index += 1
        if btn_index < BTN_ALL:
            create_btn(video_capture, frame_WIN, screen_width, screen_height)
        if btn_index == BTN_ALL:
            result = tkinter.messagebox.showinfo('提示', '注定点标定结束!')
            # get the relationship between eye and screenpoint
            get_relationship_eye_screenpoint()
            if result == tkinter.messagebox.OK:
                tkinter.messagebox.showinfo('提示', '视线追踪开始!')
    else:
        tkinter.messagebox.showinfo('提示', '未成功录入此注视点，请重新点击!')