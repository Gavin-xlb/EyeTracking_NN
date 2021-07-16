from core.FixationPoint_Standardization import *
from tkinter import *
import cv2
from PIL import ImageTk, Image
from core import FixationPoint_Standardization, ScreenHelper
from core import video
import time

validate_interval = 5  # 5s your eyes can be tolerated to leave
time_start = 0
time_end = 0
time_interval = 0
validate = True
gaze = GazeTracking()


def isValidate(x, y):
    validate_x1 = screen_width / 6
    validate_y1 = screen_height / 6
    validate_x2 = screen_width / 6 * 5
    validate_y2 = screen_height / 6 * 5
    if x >= validate_x1 and x <= validate_x2 and y >= validate_y1 and y <= validate_y2:
        return True
    return False


def adjust_threshold(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
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


if __name__ == '__main__':
    # video_capture = cv2.VideoCapture(0)
    root = Tk()
    # get screen information
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    screenhelper = ScreenHelper.ScreenHelper()
    # screen_width = screenhelper.getWResolution()
    # screen_height = screenhelper.getHResolution()
    print(screen_width, screen_height)
    canvas = Canvas(root, width=screen_width, height=screen_height)
    canvas.pack()
    root.geometry("%dx%d+0+0" % (screen_width, screen_height))
    root.resizable(width=False, height=False)
    root.attributes("-topmost", True)


    # display_thread = myThread(1, 'display_thread-1', root, canvas, video_capture)
    # display_thread.start()
    # display_thread.join()
    flag = False
    A = []
    B = []
    clf_x, clf_y = None, None
    label = Label(canvas, width=2, height=1, bg='red')


    adjust = False
    tkinter.messagebox.showinfo('提示', '瞳孔阈值调整开始...')
    while True:
        # if not root.winfo_exists():
        #     break
        _, pic = video.video_capture.read()
        cov = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)  # 初始图像是RGB格式，转换成BGR即可正常显示了
        img = Image.fromarray(cov).resize((screen_width, screen_height - 20), Image.ANTIALIAS)
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
            point_list, btn_list = create_btn(video.video_capture, canvas, screen_width, screen_height)

        if flag is False:
            clf_x, clf_y = FixationPoint_Standardization.caculateCoeficiente_RF()
        if clf_x is not None:
            print("strat predicting....")
            flag = True
            point = [video.geteccg()]
            # point = video.geteccg()
            if point[0]:

                print("point :", point)
                x_predict = clf_x.predict(np.array(point))
                y_predict = clf_y.predict(np.array(point))
                label.pack()

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

                print("predict:(%.2f,%.2f)" % (x_predict, y_predict))
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
