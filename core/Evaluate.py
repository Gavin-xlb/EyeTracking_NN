import math


class Evaluate(object):

    def __init__(self, x, y):
        self.missed_num = 0
        self.accept_num = 0
        self.num = 0
        self.screenX = x
        self.screenY = y
        self.missRatio = 0
        self.acceptRatio = 0

    def evalute(self, critical, predict_point):
        x, y = predict_point
        return math.sqrt(math.pow((x - self.screenX), 2) + math.pow((y - self.screenY), 2)) <= critical

    def caculate(self):
        self.missRatio = round(self.missed_num / self.num, 2)
        self.acceptRatio = round(self.accept_num / (self.num - 2), 2)
