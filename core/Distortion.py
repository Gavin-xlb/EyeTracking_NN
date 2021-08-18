import os
import numpy as np
import cv2
import glob


class Distortion(object):
    """该类用于相机标定，图像去畸变

    """
    inter_corner_shape = None  # tuple width*height
    size_per_grid = None  # int 现实世界里一格多少距离(单位:米)
    mat_inter = None  # 内参数矩阵
    coff_dis = None  # 畸变系数

    @staticmethod
    def calib(img_dir, img_type):
        """不返回任何值，但是会对类变量mat_inter和coff_dis赋值

        :param img_dir: 有畸变的图像的文件路径
        :param img_type: 图像格式(.jpg/.png...)
        :return: None
        """
        # criteria: only for subpix calibration, which is not used here.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        w, h = Distortion.inter_corner_shape
        # cp_int: corner point in int form, save the coordinate of corner points in world space in 'int' form
        # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
        cp_int = np.zeros((w * h, 3), np.float32)
        cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        # cp_world: corner point in world space, save the coordinate of corner points in world space.
        cp_world = cp_int * Distortion.size_per_grid

        obj_points = []  # the points in world space
        img_points = []  # the points in image space (relevant to obj_points)
        images = glob.glob(img_dir + os.sep + '**.' + img_type)
        for fname in images:
            img = cv2.imread(fname)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the corners, cp_img: corner points in pixel space.
            ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
            # if ret is True, save.
            if ret:
                cv2.cornerSubPix(gray_img, cp_img, (11, 11), (-1, -1), criteria)
                obj_points.append(cp_world)
                img_points.append(cp_img)
                # view the corners
                cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
                cv2.imshow('FoundCorners', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()
        # calibrate the camera
        ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1],
                                                                       None, None)
        print(("ret:"), ret)
        print(("internal matrix:\n"), mat_inter)  # 内参数矩阵
        # in the form of (k_1,k_2,p_1,p_2,k_3)
        print(("distortion cofficients:\n"), coff_dis)  # 畸变系数
        print(("rotation vectors:\n"), v_rot)  # 旋转向量
        print(("translation vectors:\n"), v_trans)  # 平移向量
        # calculate the error of reproject
        total_error = 0
        for i in range(len(obj_points)):
            img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
            error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
            total_error += error
        print(("Average Error of Reproject: "), total_error / len(obj_points))
        Distortion.mat_inter = mat_inter
        Distortion.coff_dis = coff_dis

    @staticmethod
    def dedistortion_validate(img_dir, img_type, save_dir):
        """对有畸变的图像进行去畸变，并进行存储

        :param img_dir: 图像来源的文件路径
        :param img_type: 图像格式(.jpg/.png...)
        :param save_dir: 图像保存的文件路径
        :return: None
        """
        if Distortion.mat_inter is None or Distortion.coff_dis is None:
            return None
        w, h = Distortion.inter_corner_shape
        images = glob.glob(img_dir + os.sep + '**.' + img_type)
        for fname in images:
            img_name = fname.split(os.sep)[-1]
            img = cv2.imread(fname)
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(Distortion.mat_inter, Distortion.coff_dis, (w, h), 0, (w, h))  # 自由比例参数
            dst = cv2.undistort(img, Distortion.mat_inter, Distortion.coff_dis, None, newcameramtx)
            cv2.imwrite(save_dir + os.sep + img_name, dst)
        print('Dedistorted images have been saved to: %s successfully.' % save_dir)

    @staticmethod
    def dedistortion(img):
        """图像去畸变

        :param img: 传入的单帧有畸变图像
        :return: 无畸变的图像
        """
        w, h = Distortion.inter_corner_shape
        if Distortion.mat_inter is None or Distortion.coff_dis is None:
            return None
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(Distortion.mat_inter, Distortion.coff_dis, (w, h), 0, (w, h))  # 自由比例参数
        dst = cv2.undistort(img, Distortion.mat_inter, Distortion.coff_dis, None, newcameramtx)
        return dst
