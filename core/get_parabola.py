import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
#from scipy.optimize import curve_fit
import math

# 二次函数的标准形式
def func(params, x):
    a, b, c = params
    return a * x * x + b * x + c

# 误差函数，即拟合曲线所求的值与实际值的差
def error(params, x, y):
    return func(params, x) - y

# 对参数求解
# leastsq方法
def slovePara(X, Y):
    p0 = [10, 10, 10]
    Para = leastsq(error, p0, args=(X, Y))
    return Para

def solution(X1, Y1, X2, Y2):
    Para1 = slovePara(X1, Y1)
    a1, b1, c1 = Para1[0]
    print("a1=", a1, " b1=", b1, " c1=", c1)
    print("cost:" + str(Para1[1]))
    print("求解的曲线是:")
    print("y=" + str(round(a1, 2)) + "x*x+" + str(round(b1, 2)) + "x+" + str(c1))

    '''plt.figure(figsize=(8, 6))
    ax1=plt.subplot(121)
    ax1.scatter(X1, Y1, color="green", label="sample data", linewidth=2)
    # 画拟合直线
    x = np.linspace(75, 200, 500)  ##在0-15直接画100个连续点
    y = a1 * x * x + b1 * x + c1  ##函数式
    ax1.plot(x, y, color="red", label="solution line", linewidth=2)
    ax1.legend()  # 绘制图例'''

    Para2 = slovePara(X2, Y2)
    a2, b2, c2 = Para2[0]
    print("a2=", a2, " b2=", b2, " c2=", c2)
    print("cost:" + str(Para2[1]))
    print("求解的曲线是:")
    print("y=" + str(round(a2, 2)) + "x*x+" + str(round(b2, 2)) + "x+" + str(c2))

    '''ax2 = plt.subplot(122)
    ax2.scatter(X2, Y2, color="blue", label="sample data", linewidth=2)
    # 画拟合直线
    x = np.linspace(75, 200, 500)  ##在0-15直接画100个连续点
    y = a2 * x * x + b2 * x + c2  ##函数式
    ax2.plot(x, y, color="red", label="solution line", linewidth=2)
    ax2.legend()  # 绘制图例
    plt.show()'''

    return Para1[0], Para2[0]