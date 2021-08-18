import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2

from core.Config import Config
from core.Distortion import Distortion

Distortion.inter_corner_shape = (11, 8)
Distortion.size_per_grid = 0.02
Distortion.calib('../image/distortion_img', 'png')
Distortion.dedistortion_validate('../image/distortion_img', 'png', '../image/dedistortion_img')
video = cv2.VideoCapture(Config.TYPE_CAMERA)
while True:
    _, img = video.read()
    deimg = Distortion.dedistortion(img)
    cv2.imshow('img', img)
    cv2.imshow('deimg', deimg)
    cv2.waitKey(1)