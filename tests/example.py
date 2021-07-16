"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2

from core.ScreenHelper import ScreenHelper
from gaze_tracking import GazeTracking
import imageio
import numpy as np
import skimage
from PIL import Image

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# imgreader = imageio.get_reader('../res/demo.gif')
# while True:
#     for frame in imgreader:
#         img = Image.fromarray(frame, 'RGBA')
#         r,g,b,a = img.split()
#         f = Image.merge('RGB',(r,g,b))
#         f=cv2.cvtColor(np.array(f),cv2.COLOR_RGB2BGR)
#         # We send this frame to GazeTracking to analyze it
#         gaze.refresh(np.array(f))
#
#         frame = gaze.annotated_frame()
#         text = ""
#
#         if gaze.is_blinking():
#             text = "Blinking"
#         elif gaze.is_right():
#             text = "Looking right"
#         elif gaze.is_left():
#             text = "Looking left"
#         elif gaze.is_center():
#             text = "Looking center"
#
#         cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
#
#         left_pupil = gaze.pupil_left_coords()
#         right_pupil = gaze.pupil_right_coords()
#         cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#         cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#         print(frame.shape)
#         cv2.imshow("Demo", frame)
#         if cv2.waitKey(1) == 27:
#              break



while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) == 27:
        break


