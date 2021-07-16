import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import face_recognition
import numpy as np
from gaze_tracking import GazeTracking
from core import FixationPoint_Standardization

# Get a reference to webcam #0 (the default one)
from core.FixationPoint_Standardization import screenhelper

video_capture = cv2.VideoCapture(0)
gaze = GazeTracking()


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


def caculate_eccg():
    # the number of frames per eye_point
    frame_num = 1
    i = 0
    frame = []
    small_frame = []

    while i < frame_num:
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
    pupil_list = []
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

            face_landmarks_list = face_recognition.face_landmarks(face_image)
            for face_landmarks in face_landmarks_list:
                for facial_feature in face_landmarks.keys():
                    # get right_eye's point
                    if facial_feature == 'right_eye':
                        right_eye_point = face_landmarks[facial_feature]

                        gaze = GazeTracking()
                        gaze.find_iris(face_image, right_eye_point, 1, 1)
                        gaze.annotated_frame()
                        right_eye = gaze.eye_right
                        if right_eye is not None:
                            ec = right_eye.center
                            cg = (right_eye.pupil.x, right_eye.pupil.y)
                            if ec is None or ec[0] is None or ec[1] is None or cg is None or cg[0] is None or cg[1] is None:
                                return ()
                            EC.append(ec)
                            CG.append(cg)
                            print('ec=', ec)
                            print('cg=', cg)

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
        return EC_CG
    else:
        return ()


def caculatePointAndDisplay(A, B):
    '''
    Z_screenX = a0  * x ^ 2 + a1 * x * y + a2 * y ^ 2 + a3 * x + a4 * y + a5
    Z_screenY = b0  * x ^ 2 + b1 * x * y + b2 * y ^ 2 + b3 * x + b4 * y + b5
    '''
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
    eccg = caculate_eccg()
    if eccg:
        print(eccg)
        x = eccg[0]
        y = eccg[1]
        Z_screenX = a0 * x * x + a1 * x * y + a2 * y * y + a3 * x + a4 * y + a5
        Z_screenY = b0 * x * x + b1 * x * y + b2 * y * y + b3 * x + b4 * y + b5
        print(Z_screenX, Z_screenY)
        return Z_screenX, Z_screenY
    else:
        return ()


def geteccg():
    return caculate_eccg()
