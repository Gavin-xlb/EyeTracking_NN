from tkinter import *


def say_hi():
    print("hello ~ !")


def f(frame1, frame2):
    frame1.pack(padx=1, pady=1)
    frame2.pack(padx=10, pady=10)


root = Tk()

frame1 = Frame(root)
frame2 = Frame(root)
root.title("tkinter frame")

label = Label(frame1, text="Label", justify=LEFT)
label.pack(side=LEFT)

hi_there = Button(frame2, text="say hi~", command=say_hi)
hi_there.pack()

f(frame1, frame2)

root.mainloop()
