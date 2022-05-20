import numpy as np
import math


class Hough_transform:
    def __init__(self, img, angle, step=5, threshold=135):
        '''
        img: 输入的边缘图像
        angle: 输入的梯度方向矩阵
        step: Hough 变换步长大小
        threshold: 筛选单元的阈值
        '''
        self.img = img
        self.angle = angle
        self.y, self.x = img.shape[0:2]
        self.radius = math.ceil(math.sqrt(self.y ** 2 + self.x ** 2))
        self.step = step
        self.vote_matrix = np.zeros(
            [math.ceil(self.y / self.step), math.ceil(self.x / self.step), math.ceil(self.radius / self.step)])
        # 投票矩阵，将原图的宽和高，半径分别除以步长，向上取整，
        self.threshold = threshold
        self.circles = []

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单元进行投票。每个点投出来结果为一折线。
        return:  投票矩阵
        '''
        print('Hough_transform_algorithm')

        for i in range(1, self.y - 1):
            for j in range(1, self.x - 1):
                if self.img[i][j] > 0:
                    # 沿梯度正方向投票
                    y = i
                    x = j
                    r = 0
                    while y < self.y and x < self.x and y >= 0 and x >= 0: # 保证圆心在图像内
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                            math.floor(r / self.step)] += 1
                        # 为了避免 y / self.step 向上取整会超出矩阵索引的情况，这里将该值向下取整
                        y = y + self.step * self.angle[i][j]
                        x = x + self.step
                        r = r + math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
                    # 沿梯度反方向投票
                    y = i - self.step * self.angle[i][j]
                    x = j - self.step
                    r = math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
                    while y < self.y and x < self.x and y >= 0 and x >= 0:
                        self.vote_matrix[math.floor(y / self.step)][math.floor(x / self.step)][
                            math.floor(r / self.step)] += 1
                        y = y - self.step * self.angle[i][j]
                        x = x - self.step
                        r = r + math.sqrt((self.step * self.angle[i][j]) ** 2 + self.step ** 2)
        return self.vote_matrix   # 返回投票矩阵

    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制，这里的非极大化抑制我采
        用的是邻近点结果取平均值的方法，而非单纯的取极大值。
        return: None
        '''
        print('Select_Circle')
		# 挑选投票数大于阈值的圆
        houxuanyuan = []
        for i in range(0, math.ceil(self.y / self.step)):
            for j in range(0, math.ceil(self.x / self.step)):
                for r in range(0, math.ceil(self.radius / self.step)):
                    if self.vote_matrix[i][j][r] >= self.threshold:
                        y = i * self.step + self.step / 2   # 通过投票矩阵中的点，恢复到原图中的点，self.step / 2为补偿值
                        x = j * self.step + self.step / 2
                        r = r * self.step + self.step / 2
                        houxuanyuan.append((math.ceil(x), math.ceil(y), math.ceil(r)))
        if len(houxuanyuan) == 0:
            print("No Circle in this threshold.")
            return
        x, y, r = houxuanyuan[0]
        possible = []
        middle = []
        for circle in houxuanyuan:
            if abs(x - circle[0]) <= 20 and abs(y - circle[1]) <= 20:
                # 设定一个误差范围（这里设定方圆20个像素以内，属于误差范围），在这个范围内的到圆心视为同一个圆心
                possible.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possible).mean(axis=0)  # 对同一范围内的圆心，半径取均值
                middle.append((result[0], result[1], result[2]))
                possible.clear()
                x, y, r = circle
                possible.append([x, y, r])
        result = np.array(possible).mean(axis=0)  # 将最后一组同一范围内的圆心，半径取均值
        middle.append((result[0], result[1], result[2]))  # 误差范围内的圆取均值后，放入其中

        def takeFirst(elem):
            return elem[0]

        middle.sort(key=takeFirst)  # 排序
        # 重复类似上述取均值的操作，并将圆逐个输出
        x, y, r = middle[0]
        possible = []
        for circle in middle:
            if abs(x - circle[0]) <= 20 and abs(y - circle[1]) <= 20:
                possible.append([circle[0], circle[1], circle[2]])
            else:
                result = np.array(possible).mean(axis=0)
                print("Circle core: (%f, %f)  Radius: %f" % (result[0], result[1], result[2]))
                self.circles.append((result[0], result[1], result[2]))
                possible.clear()
                x, y, r = circle
                possible.append([x, y, r])
        result = np.array(possible).mean(axis=0)
        print("Circle core: (%f, %f)  Radius: %f" % (result[0], result[1], result[2]))
        self.circles.append((result[0], result[1], result[2]))

    def Calculate(self):
        '''
        按照算法顺序调用以上成员函数
        return: 圆形拟合结果图，圆的坐标及半径集合
        '''
        self.Hough_transform_algorithm()
        self.Select_Circle()
        return self.circles
