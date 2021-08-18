class CalibrationHelper(object):
    """该类是标定帮助类,保存的是注视点在屏幕中心时的值

    """
    # 定义类变量
    ec_x = None  # (float):中心点离眼部区域左侧的距离
    ec_y = None  # (float):中心点离眼部区域下侧的距离
    top2bottomDist = None  # (float):上眼皮和下眼皮之间的距离
