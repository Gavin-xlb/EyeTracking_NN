import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import face_recognition
import numpy as np
from core import FixationPoint_Standardization

# Get a reference to webcam #0 (the default one)
from core.FixationPoint_Standardization import screenhelper

video_capture = cv2.VideoCapture(0)


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
                        # rectangle the location of right_eye
                        right_eye_location = rectangle_eye(right_eye_point)
                        # minor changes according to experience
                        right_eye_location_change = (
                            right_eye_location[0] + 2, right_eye_location[1] + 4, right_eye_location[2] - 2,
                            right_eye_location[3] + 1)
                        right_eye_image = face_image[right_eye_location_change[2]:right_eye_location_change[3],
                                          right_eye_location_change[0]:right_eye_location_change[1]]
                        # EC is relative to the right_eye_image
                        right_eye_height = right_eye_image.shape[0]
                        right_eye_width = right_eye_image.shape[1]
                        magnify_times = 5
                        magnify_right_eye_img = cv2.resize(right_eye_image, (0, 0), fx=magnify_times, fy=magnify_times,
                                                           interpolation=cv2.INTER_LINEAR)
                        magnify_right_eye_img_height = magnify_right_eye_img.shape[0]
                        magnify_right_eye_img_width = magnify_right_eye_img.shape[1]

                        ec_xInpixel = magnify_right_eye_img_width / 2
                        ec_yInpixel = magnify_right_eye_img_height / 2
                        PPI = screenhelper.getPPI()
                        EC.append((ec_xInpixel / magnify_times, ec_yInpixel / magnify_times))
                        # Binaryzation processing to right eye
                        gray_right_eye_image = cv2.cvtColor(magnify_right_eye_img, cv2.COLOR_BGR2GRAY)
                        shape = gray_right_eye_image.shape
                        median_x = round(shape[0] / 2)
                        median_y = round(shape[1] / 2)

                        # get the mean to the 9 central pixels of right_eye
                        mean_eye = np.mean(
                            gray_right_eye_image[(median_x - 1):(median_x + 1), (median_y - 1):(median_y + 1)])

                        ret, right_eye_binaryzation = cv2.threshold(gray_right_eye_image, mean_eye,
                                                                    255,
                                                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                        mask_sel = FixationPoint_Standardization.find_max_region(right_eye_binaryzation)
                        mu = cv2.moments(mask_sel, False)
                        mc_x = mu['m10'] / mu['m00']
                        mc_y = mu['m01'] / mu['m00']
                        mc = (mc_x / magnify_times, mc_y / magnify_times)
                        # cv2.circle(right_eye_image, mc, 0, (0, 0, 255), 5)
                        cv2.imshow("right_eye_img",right_eye_image)
                        CG.append(mc)
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
