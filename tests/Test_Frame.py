import cv2
import numpy as np

video_capture = cv2.VideoCapture(1)

def adaptive_histogram_equalization(gray_image):

    # 创建CLAHE对象
    clahe = cv2.createCLAHE(2.0, (8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(gray_image)
    return dst


while True:

    ret, img = video_capture.read()
    # histogram_dis = adaptive_histogram_equalization(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    cv2.imshow('', img)
    print(np.mean(y))
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        video_capture.release()



