import threading
from tkinter import NW

import cv2
from mttkinter import mtTkinter as tk
from PIL import ImageTk, Image


class myThread (threading.Thread):
    def __init__(self, threadID, name, root, canvas, video_capture):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.root = root
        self.canvas = canvas
        self.video_capture = video_capture

    def run(self):
        print ("开始线程：" + self.name)
        display(self.root, self.canvas, self.video_capture)
        print ("退出线程：" + self.name)

def display(root, canvas, video_capture):
    while True:
        _, pic = video_capture.read()
        cov = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)  # 初始图像是RGB格式，转换成BGR即可正常显示了
        img = Image.fromarray(cov).resize((root.winfo_screenwidth(), root.winfo_screenheight() - 20), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        canvas.create_image(0, 0, anchor=NW, image=img)
        # anchor=NW是从西北角开始排列，显示一张图片时即为正常位置，调整好img的长和宽即可
        # 更新界面
        root.update_idletasks()
        root.update()