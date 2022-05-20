import cv2
import numpy as np


def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo', dst)


def Threshold(threshold):
    frame = cv2.bilateralFilter(gray, 10, 15, 15)
    ret, frame1 = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(frame1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    if contours is not None and len(contours) > 1:
        img1 = frame1[25:-25, :]
        height, width = img1.shape[:2]
        print(str(height) + ' ' + str(width))
        nb_pixels = height * width
        nb_blacks1 = nb_pixels - cv2.countNonZero(img1)


        contours = sorted(contours, key=cv2.contourArea)

        mask = np.ones((frame.shape[0], frame.shape[1]))
        cv2.fillPoly(mask, [contours[-2]], 0)

        nb_blacks3 = frame.shape[0] * frame.shape[1] - cv2.countNonZero(mask)
        (x, y), r = cv2.minEnclosingCircle(contours[-2])
        nb_blacks2 = cv2.contourArea(contours[-2])
        print(str(nb_pixels) + ' ' + str(nb_blacks1) + ' ' + str(nb_blacks2) + ' ' + str(nb_blacks3))
        if nb_pixels != 0:
            print('ratio1=', nb_blacks1 / nb_pixels)
            print('ratio2=', nb_blacks2 / nb_pixels)
            print('ratio3=', nb_blacks3 / nb_pixels)
        cv2.circle(frame, (int(x), int(y)), int(r), (0, 0, 255), 1)
        imgs = np.hstack([frame1, mask])
        cv2.imshow('frame', frame)
        cv2.imshow('canny demo', imgs)

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread('../image/gray_eye/0.png')
img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')

# cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
cv2.createTrackbar('threshold', 'canny demo', lowThreshold, max_lowThreshold, Threshold)

# CannyThreshold(0)  # initialization
Threshold(0)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()