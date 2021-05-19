from tkinter import *
import cv2
import os

from PIL import ImageTk, Image


video_capture = cv2.VideoCapture(0)
root = Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (screen_width, screen_height))
root.attributes("-topmost", True)
canvas = Canvas(root, width=screen_width, height=screen_height)
canvas.pack()
while True:
    _, frame = video_capture.read()
    cov = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 初始图像是RGB格式，转换成BGR即可正常显示了
    img = Image.fromarray(cov).resize((screen_width, screen_height-20), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)

    canvas.create_image(0, 0, anchor=NW, image=img)
    # anchor=NW是从西北角开始排列，显示一张图片时即为正常位置，调整好img的长和宽即可
    # 更新界面
    root.update_idletasks()
    root.update()
    btn = Button(root, width=2, height=1, text=0, bg='yellow')
    btn.pack()
    btn.place(x=50, y=50)