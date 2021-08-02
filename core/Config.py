import math


class Config(object):
    # 摄像头类型 0:自带摄像头;1:外设
    TYPE_CAMERA = 1
    # 平均瞳孔半径
    AVERAGE_PUPIL_RADIUS = 8
    # 标定点总数
    CALIBRATION_POINTS_NUM = 9
    # 标定点行数
    CALIBRATION_POINTS_ROW = int(math.sqrt(CALIBRATION_POINTS_NUM))
    # 标定点列数
    CALIBRATION_POINTS_COL = CALIBRATION_POINTS_ROW
    # 标定点宽度
    CALIBRATION_POINTS_WIDTH = 20
    # 标定点高度
    CALIBRATION_POINTS_HEIGHT = 20
    # 最外侧标定点与屏幕的间隔
    CALIBRATION_POINTS_INTERVAL_EDGE = 50
    # 视线可以离开合法区域的时间
    VALIDATE_INTERVAL = 5  # 5s your eyes can be tolerated to leave
    # EC_CG放大倍数
    eccg_magnify_times = 1
