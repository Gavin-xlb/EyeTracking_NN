import cv2
import face_recognition
from core.Draw3D import Draw3D
import math
import numpy as np

# list = [(-0.14, 1.23),
# (4.09, 3.28),
# (1.97, 4.47),
# (-2.6, 2.8),
# (4.13, 0.0),
# (-1.49, 1.92),
# (3.57, -0.22),
# (0.48, -0.08),
# (-2.85, -0.4),
# (0.67, 1.6),
# (4.76, 2.03),
# (1.92, 3.04),
# (-2.5, 2.2),
# (4.46, 1.41),
# (-1.38, 1.88),
# (3.78, -0.51),
# (0.81, 0.26),
# (-2.08, 0.43),
# (1.31, 1.48),
# (4.24, 2.0),
# (1.83, 3.17),
# (-2.2, 2.3),
# (3.77, 0.3),
# (-2.07, 0.91),
# (3.58, -0.1),
# (0.17, -0.8),
# (-2.57, -0.41)]
# Draw3D.drawScatterMap(list)

# z1 = 5.303818486733606e-08
# z2 = 5.345221409615416e-10
# D = -0.00456875000000001
# z = 5.347285342777392e-05
# a = 0.0006250000000000141
# b = -0.3200000000000072
# E = 3.469446951953614e-18
# print((D + np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) / 3.0)
# print(math.sqrt(
#             (D + np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) / 3.0))
# x1 = (math.sqrt(
#             (D + np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) / 3.0)
#             + math.sqrt((2 * D - np.sign(z1) * pow(abs(z1), 1.0 / 3.0) - np.sign(z2) * pow(abs(z2), 1.0 / 3.0)
#             + 2 * math.sqrt(z)) / 3.0)) / (4 * a)
# print(x1)
# eye_frame = cv2.imread('../image/123.png')
# new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
# cv2.imwrite('../image/456.png', new_frame)
# f = cv2.imread('../image/raw_calibration/0epoch_0point_0(-2.0, 0.5).bmp')
# f1 = cv2.GaussianBlur(f, (5,5),0)
# cv2.imwrite('../image/789.bmp', f1)

# f = cv2.imread('../image/789.bmp')
# small_frame = cv2.resize(f, (0, 0), fx=0.2, fy=0.2)
#
# # Find all the faces and face encodings in the current frame of video
# face_locations = face_recognition.face_locations(small_frame, model="hog")
# a = 1
# point_list = []
# if len(face_locations) != 0:
#     # 找到最大的人脸作为检测人脸
#     max_area = 0
#     max_index = 0
#     for i in range(len(face_locations)):
#         top, right, bottom, left = face_locations[i]
#         if math.fabs((top - bottom) * (right - left)) > max_area:
#             max_area = math.fabs((top - bottom) * (right - left))
#             max_index = i
#     top, right, bottom, left = face_locations[max_index]
#     top *= 5
#     right *= 5
#     bottom *= 5
#     left *= 5
#
#     # Extract the region of the image that contains the face
#     face_image = f[top:bottom, left:right]
#     face_landmarks_list = face_recognition.face_landmarks(face_image)
#
#     for face_landmarks in face_landmarks_list:
#         # Let's trace out each facial feature in the image with a line!
#         for facial_feature in face_landmarks.keys():
#             if facial_feature == 'right_eye':
#                 cv2.
#                 # print(facial_feature, face_landmarks[facial_feature])
#                 for point in face_landmarks[facial_feature]:
#                     point_list.append(point)
#                     cv2.circle(f, (point[0]+left,point[1]+top), 0, (0, 0, 255), 3)
#                     # cv2.putText(f,str(a),(point[0]+left,point[1]+top),color=(0,255,0),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3)
#                     a += 1
#     for i in range(len(point_list)):
#         if i == 1 or i == 2:
#             cv2.circle(f, (point_list[i][0] + left, point_list[i][1] - 5 + top), 0, (255, 0, 0), 3)
#         if i == 4 or i == 5:
#             cv2.circle(f, (point_list[i][0] + left, point_list[i][1] + 5 + top), 0, (255, 0, 0), 3)
#
#
#
#     # f[top:bottom, left:right] = face_image
# cv2.imwrite('../image/333.bmp', f)

f = cv2.imread('../image/eye_frame.bmp', cv2.IMREAD_GRAYSCALE)
ret, new_frame = cv2.threshold(f, 30, 255, cv2.THRESH_BINARY)
cv2.imwrite('../image/333.bmp', new_frame)