from core.FixationPoint_Standardization import *
from tkinter import *
import cv2
from PIL import ImageTk, Image
from core import FixationPoint_Standardization
from core import video



if __name__ == '__main__':
    # video_capture = cv2.VideoCapture(0)
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    canvas = Canvas(root, width=screen_width, height=screen_height)
    canvas.pack()
    # frame = Frame(
    #     master=canvas,  # 父容器
    #     bg='white',  # 背景颜色
    #     relief='groove',  # 边框的3D样式 flat、sunken、raised、groove、ridge、solid。
    #     bd=3,  # 边框的大小
    #     height=screen_height,  # 高度
    #     width=screen_width,  # 宽度
    #     padx=1,  # 内间距，字体与边框的X距离
    #     pady=1,  # 内间距，字体与边框的Y距离
    #     cursor='arrow',  # 鼠标移动时样式 arrow, circle, cross, plus...
    # )
    # frame.pack()

    # print(screen_width,screen_height)
    root.geometry("%dx%d+0+0" % (screen_width, screen_height))
    root.attributes("-topmost", True)
    point_list, btn_list = create_btn(video.video_capture, canvas, screen_width, screen_height)

    # display_thread = myThread(1, 'display_thread-1', root, canvas, video_capture)
    # display_thread.start()
    # display_thread.join()
    flag = False
    A = []
    B = []
    label = Label(canvas, width=2, height=1, bg='red')
    while True:
        if not root.winfo_exists():
            break
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
            A, B = FixationPoint_Standardization.caculateCoeficiente()
        if A:
            flag = True
            Point = video.caculatePointAndDisplay(A, B)  #()

            if Point:
                Screen_X = Point[0]
                Screen_Y = Point[1]
                label.pack()
                label.place(x=Screen_X, y=Screen_Y)

        # Hit 'Esc' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == 27:
            break

    root.mainloop()
    # Release handle to the webcam
    video.video_capture.release()
    cv2.destroyAllWindows()