from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from core import Surface_fitting
from core.Config import Config

class Draw3D:
    """该类用于绘制拟合函数图像

    """
    @staticmethod
    def drawMap(x, y, z, flag):
        """绘制拟合曲面和离散点

        :param x: (list)EC
        :param y: (list)CG
        :param z: 屏幕横坐标/屏幕纵坐标
        :param flag: x:屏幕横坐标 y:屏幕纵坐标
        :return:
        """
        # 定义坐标轴
        fig = plt.figure()
        sub = fig.gca(projection='3d')

        sub.scatter(x, y, z, alpha=1, c='r', s=10)
        sub.set_xlim(-4, 4)
        sub.set_ylim(-6, 6)
        sub.set_zlim(0, 2000)

        ec_x = np.linspace(-4, 4, 1000)
        ec_y = np.linspace(-6, 6, 1000)
        mesh_x, mesh_y = np.meshgrid(ec_x, ec_y, indexing='ij')
        param = Surface_fitting.matching_3D(x, y, z)
        z = param[0] * mesh_x * mesh_x + param[1] * mesh_x * mesh_y + param[2] * mesh_y * mesh_y + param[3] * mesh_x + param[4] * mesh_y + param[5]
        surf = sub.plot_surface(mesh_x, mesh_y, z, rstride=3, cstride=3, cmap=cm.jet)

        line_x = np.linspace(0, 0, 1000)
        z = param[2] * ec_y * ec_y + param[4] * ec_y + param[5]
        sub.plot(line_x, ec_y, z, c='r', label='EC=0')
        line_y = np.linspace(0, 0, 1000)
        z = param[0] * ec_x * ec_x + param[3] * ec_x + param[5]
        sub.plot(ec_x, line_y, z, c='b', label='CG=0')

        sub.set_xlabel('EC')
        sub.set_ylabel('CG')
        if flag == 'x':
            sub.set_zlabel('x')
        elif flag == 'y':
            sub.set_zlabel('y')
        else:
            sub.set_zlabel('?')
        fig.colorbar(surf, shrink=0.5, aspect=8)
        sub.legend()
        plt.show()


    @staticmethod
    def drawWireFrameMap(param, flag):
        fig = plt.figure()
        sub = fig.add_subplot(111, projection='3d')
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-3, 3, 100)
        mesh_x, mesh_y = np.meshgrid(x * Config.eccg_magnify_times, y * Config.eccg_magnify_times, indexing='ij')
        z = param[0] * mesh_x * mesh_x + param[1] * mesh_x * mesh_y + param[2] * mesh_y * mesh_y + param[3] * mesh_x + param[4] * mesh_y + param[5]

        sub.plot_wireframe(mesh_x, mesh_y, z, rstride=1, cstride=1, alpha=0.1)

        sub.contour(mesh_x, mesh_y, z, cmap=cm.Accent, linewidths=2)
        sub.set_xlim(-12 * Config.eccg_magnify_times, 12 * Config.eccg_magnify_times)
        sub.set_ylim(-6 * Config.eccg_magnify_times, 6 * Config.eccg_magnify_times)
        sub.set_zlim(0, 2000)
        sub.set_xlabel('EC')
        sub.set_ylabel('CG')
        if flag == 'x':
            sub.set_zlabel('Sx')
        elif flag == 'y':
            sub.set_zlabel('Sy')
        else:
            sub.set_zlabel('S?')
        plt.show()


