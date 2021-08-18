import os
import tkinter
from PIL import Image, ImageTk
import cv2
import threading
from tkinter.messagebox import showerror

from core.Config import Config

num = 0

root = tkinter.Tk()

root.geometry('645x550')
# # 指定视频编解码方式为MJPG 需要解码器
codec = cv2.VideoWriter_fourcc(*'MJPG')
fps = 5  # 指定写入帧率为5
frameSize = (640, 480)  # 指定窗口大小
def kill(pid):
    # 本函数用于中止传入pid所对应的进程
    if os.name == 'nt':
        # Windows系统
        cmd = 'taskkill /pid ' + str(pid) + ' /f'
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)
    elif os.name == 'posix':
        # Linux系统
        cmd = 'kill ' + str(pid)
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)
    else:
        print('Undefined os.name')
def videoLoop():
    global cap
    global num
    cap = cv2.VideoCapture(Config.TYPE_CAMERA, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    global root
    global flag
    flag=False
    def paizhao():
        global flag
        flag=True
        print(flag)
    but = tkinter.Button(root,
                         text="拍照",
                         command=paizhao)
    but.grid(row=1,column=1)

    def ex():
        cap.release()
        pP = os.getpid()
        kill(pid=pP)
    but2 = tkinter.Button(root,
                          text='退出',
                          command=ex)
    but2.grid(row=1,column=3)
    vidLabel = tkinter.Label(root,
                             width=640,
                             height=400,
                             )
    vidLabel.place(x=0,y=100)
    while True:
        ret, frame = cap.read()
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame2 = Image.fromarray(frame1)
        frame3 = ImageTk.PhotoImage(frame2)
        vidLabel.configure(image=frame3)
        vidLabel.image = frame3
        if flag:
            cv2.imwrite('../image/distortion_img/' + 'distortion_' + str(num) + '.png', frame)
            num += 1
            flag=False

def closeWindow():
    tkinter.messagebox.showinfo(title='关闭错误', message='请点击退出按钮！')  # 错误消息框
    return
root.protocol('WM_DELETE_WINDOW', closeWindow)#如果关闭窗口的话，执行closeWindow函数
videoThread = threading.Thread(target=videoLoop, args=())
videoThread.start()
root.mainloop()
