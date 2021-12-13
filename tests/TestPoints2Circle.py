import datetime
import os
import cv2

def Threepoints2Circle(p1, p2, p3):
    import math
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x1x1 = x1 * x1
    y1y1 = y1 * y1
    x2x2 = x2 * x2
    y2y2 = y2 * y2
    x3x3 = x3 * x3
    y3y3 = y3 * y3
    x2y3 = x2 * y3
    x3y2 = x3 * y2
    x2_x3 = x2 - x3
    y2_y3 = y2 - y3
    x1x1py1y1 = x1x1 + y1y1
    x2x2py2y2 = x2x2 + y2y2
    x3x3py3y3 = x3x3 + y3y3

    A = x1 * y2_y3 - y1 * x2_x3 + x2y3 - x3y2
    B = x1x1py1y1 * (-y2_y3) + x2x2py2y2 * (y1 - y3) + x3x3py3y3 * (y2 - y1)
    C = x1x1py1y1 * x2_x3 + x2x2py2y2 * (x3 - x1) + x3x3py3y3 * (x1 - x2)
    D = x1x1py1y1 * (x3y2 - x2y3) + x2x2py2y2 * (x1 * y3 - x3 * y1) + x3x3py3y3 * (x2 * y1 - x1 * y2)

    if A == 0:
        return (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3
    x = -B / (2 * A)
    y = -C / (2 * A)
    r = math.sqrt((B * B + C * C - 4 * A * D) / (4 * A * A))
    return x, y


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    else:
        print(path + 'already exists!')


# ECCG_list = [(0,0),(0,4),(0,2),(0,0),(0,4),(0,2),(0,0),(0,4),(0,2),(0,0),(0,4),(0,2),(0,0),(0,4),(0,2),(0,0),(0,4),(0,2),(0,0),(0,4),(0,2),(0,0),(0,4),(0,2),(0,0),(0,4),(0,2),]
# temp_list = [Threepoints2Circle(ECCG_list[i], ECCG_list[i + 9], ECCG_list[i + 18]) for i in range(9)]
# ECCG_list = temp_list
# print('a=', ECCG_list)

# path = '../image/prediction/' + 'title'
# mkdir(path)
# img = cv2.imread(r'D:\python_Projects\EyeTracking_NN\image\calibration\0epoch_0point_0(-0.5, 2.0).jpg')
# cv2.imwrite(path + '/' + '1.jpg', img)

str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(str, type(str))