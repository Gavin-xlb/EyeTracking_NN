from core.FixationPoint_Standardization import *
from tkinter import *
import cv2
from PIL import ImageTk, Image
from core import FixationPoint_Standardization, ScreenHelper
from core import video

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
    point_list, btn_list = create_btn(video.video_capture, canvas, screen_width, screen_height)

    # display_thread = myThread(1, 'display_thread-1', root, canvas, video_capture)
    # display_thread.start()
    # display_thread.join()
    flag = False
    A = []
    B = []
    clf_x, clf_y = None, None
    label = Label(canvas, width=2, height=1, bg='red')
    while True:
        # if not root.winfo_exists():
        #     break
        _, pic = video.video_capture.read()
        cov = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)  # 初始图像是RGB格式，转换成BGR即可正常显示了
        img = Image.fromarray(cov).resize((screen_width, screen_height - 20), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        canvas.create_image(0, 0, anchor=NW, image=img)
        # anchor=NW是从西北角开始排列，显示一张图片时即为正常位置，调整好img的长和宽即可
        # 更新界面
        root.update_idletasks()
        root.update()

        if flag == False:
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
                print("predict:(%.2f,%.2f)" % (x_predict, y_predict))
                label.place(x=int(x_predict), y=int(y_predict))

    root.mainloop()
    # Release handle to the webcam
    video.video_capture.release()
    cv2.destroyAllWindows()
