import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def fun(x):
    round(x, 2)
    if x >= 0:
        return '+'+str(x)
    else:
        return str(x)


def get_res(X, Y, Z, n):
    # 求方程系数
    sigma_x = 0
    for i in X: sigma_x += i
    sigma_y = 0
    for i in Y: sigma_y += i
    sigma_z = 0
    for i in Z: sigma_z += i
    sigma_x2 = 0
    for i in X: sigma_x2 += i * i
    sigma_y2 = 0
    for i in Y: sigma_y2 += i * i
    sigma_x3 = 0
    for i in X: sigma_x3 += i * i * i
    sigma_y3 = 0
    for i in Y: sigma_y3 += i * i * i
    sigma_x4 = 0
    for i in X: sigma_x4 += i * i * i * i
    sigma_y4 = 0
    for i in Y: sigma_y4 += i * i * i * i
    sigma_x_y = 0
    for i in range(n):
        sigma_x_y += X[i] * Y[i]
    # print(sigma_xy)
    sigma_x_y2 = 0
    for i in range(n): sigma_x_y2 += X[i] * Y[i] * Y[i]
    sigma_x_y3 = 0
    for i in range(n): sigma_x_y3 += X[i] * Y[i] * Y[i] * Y[i]
    sigma_x2_y = 0
    for i in range(n): sigma_x2_y += X[i] * X[i] * Y[i]
    sigma_x2_y2 = 0
    for i in range(n): sigma_x2_y2 += X[i] * X[i] * Y[i] * Y[i]
    sigma_x3_y = 0
    for i in range(n): sigma_x3_y += X[i] * X[i] * X[i] * Y[i]
    sigma_z_x2 = 0
    for i in range(n): sigma_z_x2 += Z[i] * X[i] * X[i]
    sigma_z_y2 = 0
    for i in range(n): sigma_z_y2 += Z[i] * Y[i] * Y[i]
    sigma_z_x_y = 0
    for i in range(n): sigma_z_x_y += Z[i] * X[i] * Y[i]
    sigma_z_x = 0
    for i in range(n): sigma_z_x += Z[i] * X[i]
    sigma_z_y = 0
    for i in range(n): sigma_z_y += Z[i] * Y[i]
    # print("-----------------------")
    # 给出对应方程的矩阵形式
    a = np.array([[sigma_x4, sigma_x3_y, sigma_x2_y2, sigma_x3, sigma_x2_y, sigma_x2],
                  [sigma_x3_y, sigma_x2_y2, sigma_x_y3, sigma_x2_y, sigma_x_y2, sigma_x_y],
                  [sigma_x2_y2, sigma_x_y3, sigma_y4, sigma_x_y2, sigma_y3, sigma_y2],
                  [sigma_x3, sigma_x2_y, sigma_x_y2, sigma_x2, sigma_x_y, sigma_x],
                  [sigma_x2_y, sigma_x_y2, sigma_y3, sigma_x_y, sigma_y2, sigma_y],
                  [sigma_x2, sigma_x_y, sigma_y2, sigma_x, sigma_y, n]])
    b = np.array([sigma_z_x2, sigma_z_x_y, sigma_z_y2, sigma_z_x, sigma_z_y, sigma_z])
    # 高斯消元解线性方程
    res = np.linalg.solve(a, b)
    return res

def matching_3D(X, Y, Z):
    n = len(X)
    res = get_res(X, Y, Z, n)
    # 输出方程形式
    print("z=%.6s*x^2%.6s*xy%.6s*y^2%.6s*x%.6s*y%.6s" % (
    fun(res[0]), fun(res[1]), fun(res[2]), fun(res[3]), fun(res[4]), fun(res[5])))
    # 画曲面图和离散点
    fig = plt.figure()  # 建立一个空间
    ax = fig.add_subplot(111, projection='3d')  # 3D坐标

    n = 256
    u = np.linspace(-20, 20, n)  # 创建一个等差数列
    x, y = np.meshgrid(u, u)  # 转化成矩阵

    # 给出方程
    z = res[0] * x * x + res[1] * x * y + res[2] * y * y + res[3] * x + res[4] * y + res[5]
    # 画出曲面
    ax.plot_surface(x, y, z, rstride=3, cstride=3, cmap=cm.jet)
    # 画出点
    ax.scatter(X, Y, Z, c='r')
    plt.show()

if __name__ == '__main__':
    '''
    screen_x = a0 + a1 * x + a2 * y + a3 * x * y + a4 * x ^ 2 + a5 * y ^ 2
    screen_y = b0 + b1 * x + b2 * y + b3 * x * y + b4 * x ^ 2 + b5 * y ^ 2
    (EC, CG):(screen_x, screen_y)
    (3, -2):(50.0, 50.0)
    (-2, -3):(768.0, 50.0)
    (-5, -3):(1486.0, 50.0)
    (4, -1):(50.0, 432.0)
    (0, -1):(768.0, 432.0)
    (-5, -1):(1486.0, 432.0)
    (2, -1):(50.0, 814.0)
    (-2, -1):(768.0, 814.0)
    (-4, -1):(1486.0, 814.0)
    '''
    X = [3, -2, -5, 4, 0, -5, 2, -2, -4]
    Y = [-2, -3, -3, -1, -1, -1, -1, -1, -1]
    Z = [50, 768, 1486, 50, 768, 1486, 50, 768, 1486]
    matching_3D(X, Y, Z)