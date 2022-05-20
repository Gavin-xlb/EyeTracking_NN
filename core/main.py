import datetime
import math
import random
import time

from core.Evaluate import Evaluate
from core.FixationPoint_Standardization import *
from tkinter import *
import cv2
from PIL import ImageTk, Image
from core import FixationPoint_Standardization, ScreenHelper
from core.video import Video
from numpy import *


validate_interval = Config.VALIDATE_INTERVAL  # the time your eyes can be tolerated to leave
time_start = 0
time_end = 0
time_interval = 0
validate = True
img_open_main = None
img_main = None
btn_index_main = 0
BTN_ALL = Config.PREDICTION_POINTS_NUM
btn_list = []
label_list = []
gaze = GazeTracking()
video = Video()
average_accuracy = 0
average_accuracy_X = 0
average_accuracy_Y = 0
label = None
demoflag = False
Distortion.inter_corner_shape = Config.Distortion_inter_corner_shape
Distortion.size_per_grid = Config.Distortion_size_per_grid
present_time = datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss')

def isValidate(x, y):
    """判断视线落点是否在合法区域内

    :param x: 预测点横坐标
    :param y: 预测点纵坐标
    :return: 是否合法
    """
    validate_x1 = screen_width / 6
    validate_y1 = screen_height / 6
    validate_x2 = screen_width / 6 * 5
    validate_y2 = screen_height / 6 * 5
    if x >= validate_x1 and x <= validate_x2 and y >= validate_y1 and y <= validate_y2:
        return True
    return False


def adjust_threshold(frame):
    """自适应调整虹膜二值化阈值

    :param frame: 原图像
    :return: None
    """
    # histogram_dis = adaptive_histogram_equalization(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    # cv2.imshow('hist', small_frame)
    # cv2.waitKey(1)
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame, model='cnn')

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        # Extract the region of the image that contains the face
        face_image = frame[top:bottom, left:right]

        # cv2.imwrite(os.path.join(outdir, 'face_image' + str(btn_index) + '_' + str(i) + '.jpg'),
        #             face_image)
        face_landmarks_list = face_recognition.face_landmarks(face_image)
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                # get right_eye's point
                if facial_feature == 'right_eye':
                    # right_eye_point has 6 points
                    right_eye_point = face_landmarks[facial_feature]
                    print(facial_feature, face_landmarks[facial_feature])

                    gaze.find_iris(face_image, right_eye_point, 1, 0)
    cv2.waitKey(1)


def destroy():
    video.video_capture.release()
    cv2.destroyAllWindows()


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    else:
        print(path + ' already exists!')


def create():
    """随机生成注视点

    :return: None
    """
    global img_open_main
    global img_main
    global canvas
    frame = canvas

    d = Config.CALIBRATION_POINTS_INTERVAL_EDGE
    w = Config.CALIBRATION_POINTS_WIDTH
    h = Config.CALIBRATION_POINTS_HEIGHT
    print('createBtn...')
    x = random.randint(0, screen_width)
    y = random.randint(0, screen_height)
    point_list.append((x, y))
    evaluate = Evaluate(x, y)
    img_open_main = Image.open('../res/button_img.jpg')
    img_main = ImageTk.PhotoImage(img_open_main)
    btn = Button(frame, width=w, height=h, image=img_main)
    btn['command'] = lambda: predict(evaluate)
    btn.pack()
    btn.place(x=(x - w / 2), y=(y - h / 2))
    frame.pack()
    btn_list.append(btn)


def predict(evaluate):
    """针对每一个注视点进行预测

    :param evaluate: 对该注视点的预测点进行的评估对象
    :return: None
    """
    global canvas
    global btn_index_main
    global average_accuracy
    global average_accuracy_X
    global average_accuracy_Y
    global label_list
    points_list = []  # 所有预测成功的点
    dist_list = []  # 预测点到真实点的距离
    points_list_X = []  # 所有横坐标预测成功的点
    dist_list_X = []  # 预测点横坐标到真实点的距离
    points_list_Y = []  # 所有纵坐标预测成功的点
    dist_list_Y = []  # 预测点纵坐标到真实点的距离

    critical = screen_height / 6

    num = Video.fps * 4  # 每一个真实注视点捕获帧数
    missed_num = 0
    i = 0
    title = 'screenPoint' + str(evaluate.screenX) + '_' + str(evaluate.screenY)
    path = '../image/prediction/' + present_time + '/' + title
    path1 = '../image/raw_prediction/' + present_time + '/' + title
    mkdir(path)
    mkdir(path1)
    for lbl in label_list:
        lbl.destroy()
    label_list = []
    while i < num:
        temp = video.caculatePointAndDisplay(A, B)
        if temp:
            point, result = temp
        else:
            point = None
            result = None

        if point and result:
            x_predict, y_predict = point
            eccg, frame, pre_frame = result
            cv2.imwrite(path + '/' + str(i) + str(eccg) + '_' + str(point) + '.bmp', frame)
            cv2.imwrite(path1 + '/' + str(i) + str(eccg) + '_' + str(point) + '.bmp', pre_frame)
            cv2.imshow('frame_prediction', frame)
            cv2.waitKey(1)

            points_list.append((x_predict, y_predict))
            points_list_X.append((x_predict, y_predict))
            points_list_Y.append((x_predict, y_predict))
            dist = math.sqrt(math.pow((x_predict - evaluate.screenX), 2) + math.pow((y_predict - evaluate.screenY), 2))
            dist_list.append(dist)
            dist_X = math.fabs(x_predict - evaluate.screenX)
            dist_list_X.append(dist_X)
            dist_Y = math.fabs(y_predict - evaluate.screenY)
            dist_list_Y.append(dist_Y)
            # color 代表的是预测点的接受状态，红色为不接受，绿色为接受，由于同一测试过程中同时评估X,Y,(X,Y)三种方案，
            # 所以暂时以(X,Y)方案来决定预测点的显示状态,即下面距离用的是dist,而不是dist_X或者dist_Y
            color = 'red' if dist > critical else 'green'
            # if len(label_list) == num:
            #     label_list[i]['bg'] = color
            #     label_list[i].place(x=int(x_predict), y=int(y_predict))
            # else:
            label = Label(canvas, width=2, height=1, bg=color)
            label_list.append(label)
            label.place(x=int(x_predict), y=int(y_predict))
        else:
            missed_num += 1
        i += 1
        '''
        point = [video.geteccg()]
        if point[0]:

            print("point :", point)
            x_predict = clf_x.predict(np.array(point))
            y_predict = clf_y.predict(np.array(point))
            
            points_list.append((x_predict, y_predict))
            dist_list.append(math.sqrt(math.pow((x_predict - evaluate.screenX), 2) + math.pow((y_predict - evaluate.screenY), 2)))
            label.place(x=int(x_predict), y=int(y_predict))
        else:
            missed_num += 1
        '''

    evaluate.missed_num = missed_num
    evaluate.num = num

    if len(dist_list) > 2:
        # (X,Y)同时考虑
        with open('../res/pointsList.txt', 'a+') as fo:
            fo.write('XY\n')
        ma = max(dist_list)
        del points_list[dist_list.index(ma)]
        dist_list.remove(ma)

        mi = min(dist_list)
        del points_list[dist_list.index(mi)]
        dist_list.remove(mi)

        print('critical=', critical)
        valid_points = []
        valid_dists = []
        with open('../res/pointsList.txt', 'a+') as fo:
            fo.write('realPoint:%s' % '(' + str(evaluate.screenX) + ',' + str(evaluate.screenY) + ')\n')
            fo.write('critical distance is %.2f\n' % critical)
        for i in range(len(dist_list)):
            with open('../res/pointsList.txt', 'a+') as fo:
                fo.write('point:')
                fo.write(str(points_list[i]))
                fo.write('   distance=')
                fo.write(str(dist_list[i]))
            if dist_list[i] <= critical:
                with open('../res/pointsList.txt', 'a+') as fo:
                    fo.write('     √\n')
                valid_dists.append(dist_list[i])
                valid_points.append(points_list[i])
            else:
                with open('../res/pointsList.txt', 'a+') as fo:
                    fo.write('\n')
        evaluate.accept_num = len(valid_dists)

        # 只考虑X
        with open('../res/pointsList.txt', 'a+') as fo:
            fo.write('X\n')
        ma = max(dist_list_X)
        del points_list_X[dist_list_X.index(ma)]
        dist_list_X.remove(ma)

        mi = min(dist_list_X)
        del points_list_X[dist_list_X.index(mi)]
        dist_list_X.remove(mi)

        valid_points_X = []
        valid_dists_X = []
        for i in range(len(dist_list_X)):
            with open('../res/pointsList.txt', 'a+') as fo:
                fo.write('point:')
                fo.write(str(points_list_X[i]))
                fo.write('   distance=')
                fo.write(str(dist_list_X[i]))
            if dist_list_X[i] <= critical:
                with open('../res/pointsList.txt', 'a+') as fo:
                    fo.write('     √\n')
                valid_dists_X.append(dist_list_X[i])
                valid_points_X.append(points_list_X[i])
            else:
                with open('../res/pointsList.txt', 'a+') as fo:
                    fo.write('\n')
        evaluate.accept_num_X = len(valid_dists_X)

        # 只考虑Y
        with open('../res/pointsList.txt', 'a+') as fo:
            fo.write('Y\n')
        ma = max(dist_list_Y)
        del points_list_Y[dist_list_Y.index(ma)]
        dist_list_Y.remove(ma)

        mi = min(dist_list_Y)
        del points_list_Y[dist_list_Y.index(mi)]
        dist_list_Y.remove(mi)

        valid_points_Y = []
        valid_dists_Y = []
        for i in range(len(dist_list_Y)):
            with open('../res/pointsList.txt', 'a+') as fo:
                fo.write('point:')
                fo.write(str(points_list_Y[i]))
                fo.write('   distance=')
                fo.write(str(dist_list_Y[i]))
            if dist_list_Y[i] <= critical:
                with open('../res/pointsList.txt', 'a+') as fo:
                    fo.write('     √\n')
                valid_dists_Y.append(dist_list_Y[i])
                valid_points_Y.append(points_list_Y[i])
            else:
                with open('../res/pointsList.txt', 'a+') as fo:
                    fo.write('\n')
        evaluate.accept_num_Y = len(valid_dists_Y)

        evaluate.caculate()
        average_accuracy += evaluate.acceptRatio
        average_accuracy_X += evaluate.acceptRatio_X
        average_accuracy_Y += evaluate.acceptRatio_Y

        output_predictInfo(evaluate, valid_points, valid_points_X, valid_points_Y)

        print('missRatio=', evaluate.missRatio)
        print('acceptRatio=', evaluate.acceptRatio)
        print('acceptRatio_X=', evaluate.acceptRatio_X)
        print('acceptRatio_Y=', evaluate.acceptRatio_Y)

        # 成功录入btn_index才会+1
        btn_list[btn_index_main].destroy()  # delete button which has been clicked just now
        btn_index_main += 1
        if btn_index_main < BTN_ALL:
            create()
        else:
            average_accuracy /= BTN_ALL
            average_accuracy_X /= BTN_ALL
            average_accuracy_Y /= BTN_ALL
            fo = open("../res/predictInfo.txt", "a+")
            msg = '平均准确率=' + str(round(average_accuracy, 2)) + '\nX平均准确率=' + str(round(average_accuracy_X, 2)) + '\nY平均准确率=' + str(round(average_accuracy_Y, 2))
            fo.write('%s\n' % msg)
            # 关闭打开的文件
            fo.close()
            tkinter.messagebox.showinfo('提示', '预测结束!\n' + msg)
            for lbl in label_list:
                lbl.destroy()
    else:
        tkinter.messagebox.showinfo('提示', '预测失败，请重新预测!')


def displayTitle():
    fo = open("../res/predictInfo.txt", "a+")

    fo.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % ('屏幕坐标'.ljust(12), '错失率'.ljust(8), '接受率'.ljust(8), 'X接受率'.ljust(8), 'Y接受率'.ljust(8), '错失数'.ljust(8), '接受数'.ljust(8), 'X接受数'.ljust(8), 'Y接受数'.ljust(8), '总数'.ljust(8), '接受预测点坐标', 'X接受预测点坐标', 'Y接受预测点坐标'))
    # 关闭打开的文件
    fo.close()


def output_predictInfo(evaluate, points_list, points_list_X, points_list_Y):
    """将预测结果写入文件

    :param evaluate: 评估结果
    :param points_list: 注视点坐标列表
    :return: None
    """
    points = ''
    for row in range(len(points_list)):
        points += ',('+str(points_list[row][0])+','+str(points_list[row][1])+')'
    points_X = ''
    for row in range(len(points_list_X)):
        points_X += ',('+str(points_list_X[row][0])+','+str(points_list_X[row][1])+')'
    points_Y = ''
    for row in range(len(points_list_Y)):
        points_Y += ',(' + str(points_list_Y[row][0]) + ',' + str(points_list_Y[row][1]) + ')'
    # input dict into txt
    fo = open("../res/predictInfo.txt", "a+")

    fo.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % (('('+str(evaluate.screenX)+','+str(evaluate.screenY)+')').ljust(15), str(evaluate.missRatio).ljust(8), str(evaluate.acceptRatio).ljust(8), str(evaluate.acceptRatio_X).ljust(8), str(evaluate.acceptRatio_Y).ljust(8), str(evaluate.missed_num).ljust(8), str(evaluate.accept_num).ljust(8), str(evaluate.accept_num_X).ljust(8), str(evaluate.accept_num_Y).ljust(8), str(evaluate.num).ljust(8), points, points_X, points_Y))
    # 关闭打开的文件
    fo.close()


def change_demoflag():
    global demoflag
    demoflag = True

if __name__ == '__main__':
    displayTitle()
    # Distortion.calib('../image/distortion_img', 'png')
    print('fps=', Video.fps)
    root = Tk('eye_Tracking')
    # get screen information
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    screenhelper = ScreenHelper.ScreenHelper()
    print('PPI=', screenhelper.getPPI())
    # screen_width = screenhelper.getWResolution()
    # screen_height = screenhelper.getHResolution()
    print(screen_width, screen_height)
    canvas = Canvas(root, width=screen_width, height=screen_height)
    canvas.pack()
    root.geometry("%dx%d+0+0" % (screen_width, screen_height))
    root.resizable(width=False, height=False)
    # root.attributes("-topmost", True)
    root.overrideredirect(1)  # 去除标题栏

    del_files('../image/calibration/')
    del_files('../image/calibration_tag/')
    del_files('../image/raw_calibration/')
    mkdir('../image/prediction/' + present_time)

    flag = False

    A = []
    B = []
    clf_x, clf_y = None, None
    label = Label(canvas, width=2, height=1, bg='red')
    quitBtn = Button(canvas, width=4, height=1, text='quit', command=lambda: destroy())
    quitBtn.place(x=screen_width-50, y=5)

    predictBtn = Button(canvas, width=8, height=1, text='predict')
    predictBtn.place(x=screen_width - 150, y=5)

    demoBtn = Button(canvas, width=8, height=1, text='demo')
    demoBtn.place(x=screen_width - 250, y=5)

    adjust = False
    tkinter.messagebox.showinfo('提示', '瞳孔阈值调整开始...')
    interrupt = None
    first_createBtn = True
    while True:
        # # 将图像显示到画布
        _, pic = Video.video_capture.read()
        cov = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)  # 初始图像是RGB格式，转换成BGR即可正常显示了
        img = Image.fromarray(cov).resize((screen_width, screen_height), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        canvas.create_image(0, 0, anchor=NW, image=img)
        # anchor=NW是从西北角开始排列，显示一张图片时即为正常位置，调整好img的长和宽即可
        validate_x1 = screen_width / 6
        validate_y1 = screen_height / 6
        validate_x2 = screen_width / 6 * 5
        validate_y2 = screen_height / 6 * 5
        sel = canvas.create_rectangle(validate_x1, validate_y1, validate_x2, validate_y2, outline='black', width=5)

        if not gaze.calibration.is_complete():
            pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
            histogram_pic, cdf = CalibrationHelper.histeq(array(pic))
            adjust_threshold(uint8(histogram_pic))
        if gaze.calibration.is_complete() and adjust is False:
            adjust = True
            tkinter.messagebox.showinfo('提示', '瞳孔阈值调整完成!')
            print('best_threshold = ', gaze.calibration.best_thres)
            create_btn(Video.video_capture, canvas, screen_width, screen_height)

        if flag is False:
            # clf_x, clf_y = FixationPoint_Standardization.caculateCoeficiente_RF()
            A, B = caculateCoeficiente()

            if A:
                flag = True
                predictBtn['command'] = lambda: create()
                demoBtn['command'] = lambda: change_demoflag()
        # if clf_x is not None:
        if A:
            if demoflag:
                print("start predicting....")
                temp = video.caculatePointAndDisplay(A, B)
                if temp:
                    point, result = temp
                else:
                    point = None
                    result = None

                if point and result:
                    x_predict, y_predict = point
                    eccg, frame, pre_frame = result
                    label.place(x=int(x_predict), y=int(y_predict))
                    if isValidate(x_predict, y_predict):
                        validate = True
                        canvas.itemconfig(sel, outline='green')
                        text = ''
                        canvas.create_text(250, 100, text=text)
                    else:
                        if validate:
                            time_start = time.time()
                            time_end = time_start
                        else:
                            time_end = time.time()
                        time_interval = time_end - time_start
                        text = 'the time you leave: ' + str(round(time_interval, 2)) + ' seconds'
                        canvas.create_text(250, 100, text=text, fill='red', font=("Purisa", 30))
                        canvas.itemconfig(sel, outline='red')
                        if time_interval > validate_interval:
                            canvas.create_text(150, 150, text='Cheating...', fill='red', font=("Purisa", 30))
                        validate = False

                    print("predictPoint:(%.2f,%.2f)" % (x_predict, y_predict))
                else:
                    if validate:
                        time_start = time.time()
                        time_end = time_start
                    else:
                        time_end = time.time()
                    time_interval = time_end - time_start
                    text = 'the time you leave: ' + str(round(time_interval, 2)) + ' seconds'
                    canvas.create_text(250, 100, text=text, fill='red', font=("Purisa", 30))
                    canvas.itemconfig(sel, outline='red')
                    if time_interval > validate_interval:
                        canvas.create_text(150, 150, text='Cheating...', fill='red', font=("Purisa", 30))
                    validate = False


        # 更新界面
        root.update_idletasks()
        root.update()

    root.mainloop()
    # Release handle to the webcam
    video.video_capture.release()
    cv2.destroyAllWindows()
