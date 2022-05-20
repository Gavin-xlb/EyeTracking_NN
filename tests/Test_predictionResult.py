import math

from core import Surface_fitting
import os

from core import ScreenHelper
screenhelper = ScreenHelper.ScreenHelper()
screen_H = screenhelper.getHResolution() / 1.25
screen_W = screenhelper.getWResolution() / 1.25
critical = screen_H / 4

def splitItem(list):
    ec = []
    cg = []
    x = []
    y = []
    for item in list:
        eccg, xy = item.split(':')
        eccg = eccg[1:-1]
        xy = xy[1:-1]
        ec.append(float(eccg.split(',')[0]))
        cg.append(float(eccg.split(',')[1]))
        x.append(float(xy.split(',')[0]))
        y.append(float(xy.split(',')[1]))
    return ec, cg, x, y

def evaluate(A, B):
    base_path = '../image/raw_prediction/2022-05-18 15h46m46s'
    img_dirs = os.listdir(base_path)
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

    avg_xaccept = 0
    avg_yaccept = 0
    avg_xyaccept = 0
    for file in img_dirs:
        real_xy = file[11:].strip().split('_')
        real_x, real_y = float(real_xy[0]), float(real_xy[1])
        img_files = os.listdir(os.path.join(base_path, file))
        img_num = len(img_files)
        valid_xynum = 0
        valid_xnum = 0
        valid_ynum = 0
        for img in img_files:
            if img.endswith('.bmp'):
                temp = img.split('_')[0][:-1]
                eccg = temp[temp.index('(') + 1:]
                x = float(eccg.split(',')[0])
                y = float(eccg.split(',')[1])
                predicted_x = a0 * x * x + a1 * x * y + a2 * y * y + a3 * x + a4 * y + a5
                predicted_y = b0 * x * x + b1 * x * y + b2 * y * y + b3 * x + b4 * y + b5
                if math.fabs(predicted_x - real_x) < critical:
                    valid_xnum += 1
                if math.fabs(predicted_y - real_y) < critical:
                    valid_ynum += 1
                if math.sqrt(math.pow((predicted_x - real_x), 2) + math.pow((predicted_y - real_y), 2)) < critical:
                    valid_xynum += 1
        with open('../res/evaluate_data.txt', 'a+') as fo:
            fo.write(str(file) + '\n')
            fo.write('x_acceptRatio={}\n'.format(valid_xnum / img_num))
            fo.write('y_acceptRatio={}\n'.format(valid_ynum / img_num))
            fo.write('xy_acceptRatio={}\n\n'.format(valid_xynum / img_num))
            avg_xaccept += valid_xnum / img_num
            avg_yaccept += valid_ynum / img_num
            avg_xyaccept += valid_xynum / img_num
    with open('../res/evaluate_data.txt', 'a+') as fo:
        fo.write('avg_xaccept={}\n'.format(avg_xaccept / 9))
        fo.write('avg_yaccept={}\n'.format(avg_yaccept / 9))
        fo.write('avg_xyaccept={}\n\n'.format(avg_xyaccept / 9))

if __name__ == '__main__':
    with open('../res/data.txt', 'r') as fo:
        data = [item.replace('\n', '') for item in fo.readlines() if item != '\n']
    round_1 = data[:9]
    round_2 = data[9:18]
    round_3 = data[18:27]
    round_12 = round_1 + round_2
    round_13 = round_1 + round_3
    round_23 = round_2 + round_3
    round_123 = data

    # 利用第一轮数据
    with open('../res/evaluate_data.txt', 'a+') as fo:
        fo.write('利用第一轮的数据\n')
    ec1, cg1, x1, y1 = splitItem(round_1)
    A = Surface_fitting.matching_3D(ec1, cg1, x1)
    B = Surface_fitting.matching_3D(ec1, cg1, y1)
    evaluate(A, B)

    # 利用第二轮数据
    with open('../res/evaluate_data.txt', 'a+') as fo:
        fo.write('利用第二轮的数据\n')
    ec2, cg2, x2, y2 = splitItem(round_2)
    A = Surface_fitting.matching_3D(ec2, cg2, x2)
    B = Surface_fitting.matching_3D(ec2, cg2, y2)
    evaluate(A, B)

    # 利用第三轮数据
    with open('../res/evaluate_data.txt', 'a+') as fo:
        fo.write('利用第三轮的数据\n')
    ec3, cg3, x3, y3 = splitItem(round_3)
    A = Surface_fitting.matching_3D(ec3, cg3, x3)
    B = Surface_fitting.matching_3D(ec3, cg3, y3)
    evaluate(A, B)

    # 利用第一、二轮数据
    with open('../res/evaluate_data.txt', 'a+') as fo:
        fo.write('利用第一、二轮的数据\n')
    ec12, cg12, x12, y12 = splitItem(round_12)
    A = Surface_fitting.matching_3D(ec12, cg12, x12)
    B = Surface_fitting.matching_3D(ec12, cg12, y12)
    evaluate(A, B)

    # 利用第一、三轮数据
    with open('../res/evaluate_data.txt', 'a+') as fo:
        fo.write('利用第一、三轮的数据\n')
    ec13, cg13, x13, y13 = splitItem(round_13)
    A = Surface_fitting.matching_3D(ec13, cg13, x13)
    B = Surface_fitting.matching_3D(ec13, cg13, y13)
    evaluate(A, B)

    # 利用第二、三轮数据
    with open('../res/evaluate_data.txt', 'a+') as fo:
        fo.write('利用第二、三轮的数据\n')
    ec23, cg23, x23, y23 = splitItem(round_23)
    A = Surface_fitting.matching_3D(ec23, cg23, x23)
    B = Surface_fitting.matching_3D(ec23, cg23, y23)
    evaluate(A, B)

    # 利用第一、二、三轮数据
    with open('../res/evaluate_data.txt', 'a+') as fo:
        fo.write('利用第一、二、三轮的数据\n')
    ec123, cg123, x123, y123 = splitItem(round_123)
    A = Surface_fitting.matching_3D(ec123, cg123, x123)
    B = Surface_fitting.matching_3D(ec123, cg123, y123)
    evaluate(A, B)