import math

from core.Evaluate import Evaluate
from core.FixationPoint_Standardization import *
from tkinter import *
import cv2
from PIL import ImageTk, Image
from core import FixationPoint_Standardization, ScreenHelper
from core.video import Video
import time
import threading

validate_interval = Config.VALIDATE_INTERVAL  # the time your eyes can be tolerated to leave
time_start = 0
time_end = 0
time_interval = 0
validate = True
img_open_main = None
img_main = None
btn_index_main = 0
btn_list = []
gaze = GazeTracking()
video = Video()
average_accuracy = 0

def isValidate(x, y):
    validate_x1 = screen_width / 6
    validate_y1 = screen_height / 6
    validate_x2 = screen_width / 6 * 5
    validate_y2 = screen_height / 6 * 5
    if x >= validate_x1 and x <= validate_x2 and y >= validate_y1 and y <= validate_y2:
        return True
    return False


def adjust_threshold(frame):
    # histogram_dis = adaptive_histogram_equalization(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    cv2.imshow('hist', small_frame)
    cv2.waitKey(1)
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame)

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


def create():
    global img_open_main
    global img_main
    global canvas
    frame = canvas

    # 创建一个Canvas，设置其背景色为白色
    # cv = Canvas(root, bg='white', height=screen_height, width=screen_width)
    d = Config.CALIBRATION_POINTS_INTERVAL_EDGE
    w = Config.CALIBRATION_POINTS_WIDTH
    h = Config.CALIBRATION_POINTS_HEIGHT
    print('createBtn...')
    x = f(btn_index_main) * (screen_width - 2 * d) / (COL_POINT - 1) + d
    y = g(btn_index_main) * (screen_height - 2 * d) / (ROW_POINT - 1) + d
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
    global label
    global btn_index_main
    global average_accuracy
    points_list = []  # 所有预测成功的点
    dist_list = []
    num = Video.fps
    missed_num = 0
    i = 0
    while i < num:
        i += 1
        point = video.caculatePointAndDisplay(A, B)
        if point:
            x_predict, y_predict = point
            points_list.append((x_predict, y_predict))
            dist_list.append(math.sqrt(math.pow((x_predict - evaluate.screenX), 2) + math.pow((y_predict - evaluate.screenY), 2)))
            label.place(x=int(x_predict), y=int(y_predict))
        else:
            missed_num += 1
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
        ma = max(dist_list)
        del points_list[dist_list.index(ma)]
        dist_list.remove(ma)

        mi = min(dist_list)
        del points_list[dist_list.index(mi)]
        dist_list.remove(mi)

        critical = screen_height / 6
        print('critical=', critical)
        valid_points = []
        valid_dists = []
        for i in range(len(dist_list)):
            if dist_list[i] <= critical:
                valid_dists.append(dist_list[i])
                valid_points.append(points_list[i])
        evaluate.accept_num = len(valid_dists)

        evaluate.caculate()
        average_accuracy += evaluate.acceptRatio

        output_predictInfo(evaluate, valid_points)

        print('missRatio=', evaluate.missRatio)
        print('acceptRatio=', evaluate.acceptRatio)

        # 成功录入btn_index才会+1
        btn_list[btn_index_main].destroy()  # delete button which has been clicked just now
        btn_index_main += 1
        if btn_index_main < BTN_ALL:
            create()
        else:
            average_accuracy /= Config.CALIBRATION_POINTS_NUM
            fo = open("../res/predictInfo.txt", "a+")

            fo.write('%s=%s\n' % ('平均准确率', str(round(average_accuracy, 2))))
            # 关闭打开的文件
            fo.close()
            tkinter.messagebox.showinfo('提示', '预测结束!')
    else:
        tkinter.messagebox.showinfo('提示', '预测失败，请重新预测!')


def displayTitle():
    fo = open("../res/predictInfo.txt", "a+")

    fo.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % ('屏幕坐标'.ljust(12), '错失率'.ljust(8), '接受率'.ljust(8), '错失数'.ljust(8), '接受数'.ljust(8), '总数'.ljust(8), '接受预测点坐标'))
    # 关闭打开的文件
    fo.close()


def output_predictInfo(evaluate, points_list):
    points = ''
    for row in range(len(points_list)):
        points += ',('+str(points_list[row][0])+','+str(points_list[row][1])+')'
    # input dict into txt
    fo = open("../res/predictInfo.txt", "a+")

    fo.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t\n' % (('('+str(evaluate.screenX)+','+str(evaluate.screenY)+')').ljust(15), str(evaluate.missRatio).ljust(8), str(evaluate.acceptRatio).ljust(8), str(evaluate.missed_num).ljust(8), str(evaluate.accept_num).ljust(8), str(evaluate.num).ljust(8), points))
    # 关闭打开的文件
    fo.close()


if __name__ == '__main__':
    displayTitle()

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

    flag = False
    A = []
    B = []
    clf_x, clf_y = None, None
    label = Label(canvas, width=2, height=1, bg='red')
    quitBtn = Button(canvas, width=4, height=1, text='quit', command=lambda: destroy())
    quitBtn.place(x=screen_width-50, y=5)

    predictBtn = Button(canvas, width=8, height=1, text='predict')
    predictBtn.place(x=screen_width - 150, y=5)

    adjust = False
    tkinter.messagebox.showinfo('提示', '瞳孔阈值调整开始...')
    interrupt = None
    first_createBtn = True
    while True:

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
            adjust_threshold(pic)
        if gaze.calibration.is_complete() and adjust is False:
            adjust = True
            tkinter.messagebox.showinfo('提示', '瞳孔阈值调整完成!')
            print('best_threshold = ', gaze.calibration.best_thres)
            create_btn(Video.video_capture, canvas, screen_width, screen_height)

        if flag is False:
            # clf_x, clf_y = FixationPoint_Standardization.caculateCoeficiente_RF()
            A, B = caculateCoeficiente()
        # if clf_x is not None:
        if A:
            # flag = True
            predictBtn['command'] = lambda: create()
            print("start predicting....")
            flag = True
            # 随机森林
            # point = [video.geteccg()]
            # if point[0]:
            #
            #     print("point :", point)
            #     x_predict = clf_x.predict(np.array(point))
            #     y_predict = clf_y.predict(np.array(point))
            #     label.pack()
            point = video.caculatePointAndDisplay(A, B)
            if point:
                x_predict, y_predict = point

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
                        # result = tkinter.messagebox.showinfo('提示', '系统判定您在作弊，请停止作答!')
                        # if result:
                        #     video.video_capture.release()
                        #     cv2.destroyAllWindows()
                    validate = False

                print("predictPoint:(%.2f,%.2f)" % (x_predict, y_predict))
                label.place(x=int(x_predict), y=int(y_predict))
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
                    # result = tkinter.messagebox.showinfo('提示', '系统判定您在作弊，请停止作答!')
                    # if result:
                    #     video.video_capture.release()
                    #     cv2.destroyAllWindows()
                validate = False


        # 更新界面
        root.update_idletasks()
        root.update()

    root.mainloop()
    # Release handle to the webcam
    video.video_capture.release()
    cv2.destroyAllWindows()
