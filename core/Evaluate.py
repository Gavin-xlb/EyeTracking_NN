import math


class Evaluate(object):
    """该类用于评估预测点的预测效果，包括是否可接受和接受率、错失率

    """

    def __init__(self, x, y):
        self.missed_num = 0
        self.accept_num = 0
        self.accept_num_X = 0
        self.accept_num_Y = 0
        self.num = 0
        self.screenX = x
        self.screenY = y
        self.missRatio = 0
        self.acceptRatio = 0
        self.acceptRatio_X = 0
        self.acceptRatio_Y = 0

    def evalute(self, critical, predict_point):
        """评估预测点是否可接受

        :param critical: 临界值
        :param predict_point: (tuple)预测点坐标
        :return: 是否在临界值之内(可接受)
        """
        x, y = predict_point
        return math.sqrt(math.pow((x - self.screenX), 2) + math.pow((y - self.screenY), 2)) <= critical

    def caculate(self):
        """计算错失率和接受率

        :return: None
        """
        self.missRatio = round(self.missed_num / self.num, 2)
        self.acceptRatio = round(self.accept_num / (self.num - 2), 2)
        self.acceptRatio_X = round(self.accept_num_X / (self.num - 2), 2)
        self.acceptRatio_Y = round(self.accept_num_Y / (self.num - 2), 2)
