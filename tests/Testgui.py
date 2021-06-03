from core.FixationPoint_Standardization import *
from tkinter import *
import cv2
from PIL import ImageTk, Image
from core import FixationPoint_Standardization, ScreenHelper
from core import video

root = Tk()
# screenhelper = ScreenHelper.ScreenHelper()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
# screen_width = screenhelper.getWResolution()
# screen_height = screenhelper.getHResolution()
print(screen_width, screen_height)
canvas = Canvas(root, width=screen_width, height=screen_height)
canvas.pack()
img_open = Image.open('../res/button_img.jpg')
img_png = ImageTk.PhotoImage(img_open)
btn = Button(canvas, image=img_png, width=10, height=10)
btn.pack()
btn.place(x=960, y=540)
root.mainloop()

