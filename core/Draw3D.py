from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from core.Config import Config


class Draw3D:
    @staticmethod
    def drawScatterMap(x, y, z, flag):
        # 定义坐标轴
        fig = plt.figure()
        sub = fig.add_subplot(111, projection='3d')

        sub.scatter(x, y, z, alpha=1, c='r', s=10)
        sub.set_xlim(-10*Config.eccg_magnify_times, 10*Config.eccg_magnify_times)
        sub.set_ylim(-7*Config.eccg_magnify_times, 7*Config.eccg_magnify_times)
        sub.set_zlim(-200, 2000)
        sub.set_xlabel('EC')
        sub.set_ylabel('CG')
        if flag == 'x':
            sub.set_zlabel('Sx')
        elif flag == 'y':
            sub.set_zlabel('Sy')
        else:
            sub.set_zlabel('S?')
        plt.show()

    @staticmethod
    def drawSurfaceMap(param, flag):
        fig = plt.figure()
        sub = fig.add_subplot(111, projection='3d')
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-3, 3, 100)
        mesh_x, mesh_y = np.meshgrid(x * Config.eccg_magnify_times, y * Config.eccg_magnify_times, indexing='ij')
        z = param[0] + param[1] * mesh_x + param[2] * mesh_y + param[3] * mesh_x * mesh_y + param[4] * mesh_x * mesh_x + param[5] * mesh_y * mesh_y

        surf = sub.plot_surface(mesh_x, mesh_y, z, cmap='rainbow')

        sub.set_xlim(-12*Config.eccg_magnify_times, 12*Config.eccg_magnify_times)
        sub.set_ylim(-6*Config.eccg_magnify_times, 6*Config.eccg_magnify_times)
        sub.set_zlim(-200, 2000)
        sub.set_xlabel('EC')
        sub.set_ylabel('CG')
        if flag == 'x':
            sub.set_zlabel('Sx')
        elif flag == 'y':
            sub.set_zlabel('Sy')
        else:
            sub.set_zlabel('S?')
        fig.colorbar(surf, shrink=0.5, aspect=8)
        plt.show()

    @staticmethod
    def drawWireFrameMap(param, flag):
        fig = plt.figure()
        sub = fig.add_subplot(111, projection='3d')
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-3, 3, 100)
        mesh_x, mesh_y = np.meshgrid(x * Config.eccg_magnify_times, y * Config.eccg_magnify_times, indexing='ij')
        z = param[0] + param[1] * mesh_x + param[2] * mesh_y + param[3] * mesh_x * mesh_y + param[4] * mesh_x * mesh_x + \
            param[5] * mesh_y * mesh_y

        sub.plot_wireframe(mesh_x, mesh_y, z, rstride=1, cstride=1, alpha=0.1)

        sub.contour(mesh_x, mesh_y, z, cmap=cm.Accent, linewidths=2)
        sub.set_xlim(-12 * Config.eccg_magnify_times, 12 * Config.eccg_magnify_times)
        sub.set_ylim(-6 * Config.eccg_magnify_times, 6 * Config.eccg_magnify_times)
        sub.set_zlim(-200, 2000)
        sub.set_xlabel('EC')
        sub.set_ylabel('CG')
        if flag == 'x':
            sub.set_zlabel('Sx')
        elif flag == 'y':
            sub.set_zlabel('Sy')
        else:
            sub.set_zlabel('S?')
        plt.show()


