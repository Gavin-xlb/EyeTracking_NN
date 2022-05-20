import cv2
import face_recognition
from numpy import *
import numpy as np

from core.CalibrationHelper import CalibrationHelper
from core.Distortion import Distortion
from gaze_tracking import GazeTracking
from core.Config import Config
from core import FixationPoint_Standardization


# Get a reference to webcam #0 (the default one)
from core.FixationPoint_Standardization import screenhelper


gaze = GazeTracking()


class Video(object):
    """预测时计算EC-CG、计算预测落点

    """
    video_capture = cv2.VideoCapture(Config.TYPE_CAMERA)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    predict_num = 0

    def caculate_eccg(self):
        """计算EC-CG

        :return: (tuple)EC-CG
        """
        # the number of frames per eye_point
        frame_num = 1
        i = 0
        frame = []
        small_frame = []
        pre_frame = None  # 整张原图像

        while i < frame_num:
            # Grab a single frame of video
            ret, f = self.video_capture.read()
            pre_frame = np.copy(f)
            f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            histogram_f, cdf = CalibrationHelper.histeq(array(f))
            # f = Distortion.dedistortion(f)

            # f = FixationPoint_Standardization.adaptive_histogram_equalization(
            #     cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            frame.append(uint8(histogram_f))
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
        top2bottom_list = []
        face_image = None
        while i < frame_num:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame[i], model='cnn')

            # Display the results
            for top, right, bottom, left in face_locations:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 5
                right *= 5
                bottom *= 5
                left *= 5

                # Extract the region of the image that contains the face
                face_image = frame[i][top:bottom, left:right]
                pre_face_image = np.copy(pre_frame[top:bottom, left:right])

                face_landmarks_list = face_recognition.face_landmarks(face_image)
                for face_landmarks in face_landmarks_list:
                    right_eye_point = face_landmarks['right_eye']

                    gaze.find_iris(face_image, right_eye_point, 1, 1)
                    for point in right_eye_point:
                        cv2.circle(pre_face_image, point, 0, (0, 0, 255), 3)
                    right_eye = gaze.eye_right
                    if right_eye is not None:
                        # ec = right_eye.center
                        cg = (right_eye.pupil.cg_x, right_eye.pupil.cg_y)
                        if cg is None or cg[0] is None or cg[1] is None:
                            break
                        # EC.append(ec)
                        CG.append(cg)
                        temp_dst = right_eye.top2bottom
                        top2bottom_list.append(temp_dst)
                        # print('ec=', ec)
                        print('cg=', cg)

                        num += 1
            i += 1
        # shot successfully
        if (i == frame_num) and (num != 0):
            p = 0
            for d in top2bottom_list:
                p += d
            avg_dst = p / num
            x = 0
            y = 0
            for t in CG:
                x += t[0]
                y += t[1]
            delta_dst = avg_dst - CalibrationHelper.top2bottomDist
            cg = (x / num, y / num + delta_dst)
            EC_CG = (round((cg[0] - CalibrationHelper.ec_x), 2), round((cg[1] - CalibrationHelper.ec_y), 2))
            frame1 = gaze.annotated_frame(pre_face_image, delta_dst)
            return EC_CG, frame1, pre_frame
        else:
            return

    def caculatePointAndDisplay(self, A, B):
        """计算预测落点坐标

        :param A: (a0, a1, a2, a3, a4, a5)
        :param B: (b0, b1, b2, b3, b4, b5)
        :return: (tuple)预测落点坐标
        """
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

        result = self.caculate_eccg()
        if result:
            eccg, frame, pre_frame = result
            if eccg:
                print('predict_eccg:', eccg)
                x = eccg[0]
                y = eccg[1]
                Z_screenX = a0 * x * x + a1 * x * y + a2 * y * y + a3 * x + a4 * y + a5
                Z_screenY = b0 * x * x + b1 * x * y + b2 * y * y + b3 * x + b4 * y + b5
                print('Z_screenX=%.2f,Z_screenY=%.2f' % (Z_screenX, Z_screenY))
                return (Z_screenX, Z_screenY), result
            else:
                return
        else:
            return

    def geteccg(self):
        return self.caculate_eccg()
