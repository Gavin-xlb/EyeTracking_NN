import math
import cmath
import numpy as np

def  get_args(circle_args_list,parabola_args_list):
    c_a, c_b, c_c, c_d, c_e = circle_args_list
    p_a, p_b, p_c = parabola_args_list
    A = c_b*(p_a**2)
    B = 2*c_b*p_a*p_b
    C = 2*c_b*p_a*p_c+c_b*(p_b**2)+c_d*p_a+c_a
    D = 2*c_b*p_b*p_c+c_d*p_b+c_c
    E = c_b*p_c**2+c_d*p_c+c_e
    args=[A, B, C, D, E]
    return args

def cal_quartic_ik(args_list):
    a, b, c, d, e = args_list

    D = 3 * pow(b, 2) - 8 * a * c
    E = -pow(b, 3) + 4 * a * b * c - 8 * pow(a, 2) * d
    F = 3 * pow(b, 4) + 16 * pow(a, 2) * pow(c, 2) - 16 * a * pow(b, 2) * c + 16 * pow(a, 2) * b * d - 64 * pow(a,3) * e

    A = D ** 2 - 3 * F
    B = D * F - 9 * pow(E, 2)
    C = F ** 2 - 3 * D * pow(E, 2)
    # print("ABCDEF", A, B, C, D, E, F)
    delta = B ** 2 - 4 * A * C  # 总判别式

    print('delta', delta)

    if (D == 0) & (E == 0) & (F == 0):
        """ 四重实根"""
        x = -b / (4 * a)
        return 1, [x]
    if (A == 0) & (B == 0) & (C == 0) & (D * E * F != 0):
        """ 两个实根，其中一个三重实根"""
        x1 = (-b * D + 9 * E) / (4 * a * D)
        x234 = (-b * D - 3 * E) / (4 * a * D)
        return 2, [x1, x234]
    if (E == 0) & (F == 0) & (D != 0):
        """ 一对二重根"""
        if D > 0:  # 根为实数
            x13 = (-b + math.sqrt(D)) / (4 * a)
            x24 = (-b - math.sqrt(D)) / (4 * a)
            return 2, [x13, x24]

        if D < 0:  # 根为虚数
            # x13 = (-b + cmath.sqrt(D))/(4*a)
            # x24 = (-b - cmath.sqrt(D)) / (4 * a)
            return 0, 0
    if (A * B * C != 0) & (delta == 0):
        """ 一对二重实根 """
        x3 = (-b - np.sign(A * B * E) * math.sqrt(D - B / A)) / (4 * a)
        x4 = (-b - np.sign(A * B * E) * math.sqrt(D - B / A)) / (4 * a)
        if A * B > 0:  # 其余两根为不等实根
            x1 = (-b + np.sign(A * B * E) * math.sqrt(D - B / A) + math.sqrt(2 * B / A)) / (4 * a)
            x2 = (-b + np.sign(A * B * E) * math.sqrt(D - B / A) - math.sqrt(2 * B / A)) / (4 * a)
            return 4, [x1, x2, x3, x4]
        if A * B < 0:  # 其余两根为共轭虚根
            # x1 = (-b + np.sign(A * B * E) * math.sqrt(D - B / A) + cmath.sqrt(2 * B / A)) / (4 * a)
            # x2 = (-b + np.sign(A * B * E) * math.sqrt(D - B / A) - cmath.sqrt(2 * B / A)) / (4 * a)
            return 2, [x3, x4]
    if delta > 0:
        """" 两个不等实根和一对共轭虚根"""
        z1 = A * D + 3 * ((-B + math.sqrt(delta)) / 2.0)
        z2 = A * D + 3 * ((-B - math.sqrt(delta)) / 2.0)

        # print """ z1 =  """, z1
        # print """ z2 =  """, z2
        # print """ abs(z1) =  """, abs(z1)
        # print """ abs(z2) =  """, abs(z2)

        print('z1', z1)
        print('z2', z2)

        print('D', D)
        print('a', a)
        print('b', b)
        print('E', E)

        z = D ** 2 - D * (np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) + \
            (np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) ** 2 - 3 * A
        print('z', z)
        x1 = (-b + np.sign(E) * math.sqrt(
            (D + np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) / 3.0)
            + math.sqrt((2 * D - np.sign(z1) * pow(abs(z1), 1.0 / 3.0) - np.sign(z2) * pow(abs(z2), 1.0 / 3.0)
            + 2 * math.sqrt(z)) / 3.0)) / (4 * a)
        x2 = (-b + np.sign(E) * math.sqrt(
            (D + np.sign(z1) * pow(abs(z1), 1.0 / 3.0) + np.sign(z2) * pow(abs(z2), 1.0 / 3.0)) / 3.0)
              - math.sqrt((2 * D - np.sign(z1) * pow(abs(z1), 1.0 / 3.0) - np.sign(z2) * pow(abs(z2), 1.0 / 3.0)
                           + 2 * math.sqrt(z)) / 3.0)) / (4 * a)

        # 虚根忽略
        return 2, [x1, x2]
    if delta < 0:
        if E == 0:
            if (D > 0) & (F > 0):
                """ 四个不等实根 """
                x1 = (-b + math.sqrt(D + 2 * math.sqrt(F))) / (4 * a)
                x2 = (-b - math.sqrt(D + 2 * math.sqrt(F))) / (4 * a)
                x3 = (-b + math.sqrt(D - 2 * math.sqrt(F))) / (4 * a)
                x4 = (-b - math.sqrt(D - 2 * math.sqrt(F))) / (4 * a)
                return 4, [x1, x2, x3, x4]
            else:
                """ 两对不等共轭虚根 """
                # 虚根忽略
                print("null")
                " 两对不等共轭虚根 "
                return 0, 0
        else:
            if (D > 0) & (F > 0):
                """ 四个不等实根 """
                theta = math.acos((3 * B - 2 * A * D) / (2 * A * math.sqrt(A)))
                y1 = (D - 2 * math.sqrt(A) * math.cos(theta / 3.0)) / 3.0
                y2 = (D + math.sqrt(A) * (math.cos(theta / 3.0) + math.sqrt(3) * math.sin(theta / 3.0))) / 3.0
                y3 = (D + math.sqrt(A) * (math.cos(theta / 3.0) - math.sqrt(3) * math.sin(theta / 3.0))) / 3.0

                x1 = (-b + np.sign(E) * math.sqrt(y1) + (math.sqrt(y2) + math.sqrt(y3))) / (4 * a)
                x2 = (-b + np.sign(E) * math.sqrt(y1) - (math.sqrt(y2) + math.sqrt(y3))) / (4 * a)
                x3 = (-b - np.sign(E) * math.sqrt(y1) + (math.sqrt(y2) - math.sqrt(y3))) / (4 * a)
                x4 = (-b - np.sign(E) * math.sqrt(y1) - (math.sqrt(y2) - math.sqrt(y3))) / (4 * a)

                return 4, [x1, x2, x3, x4]
            else:
                """ 两对不等共轭虚根 """
                # 虚根忽略
                print("null")
                " 两对不等共轭虚根 "
                return 0, 0
def get_result(args_list, x):
    result = np.zeros((len(x), 2))
    for i in range(len(x)):
        result[i][0] = x[i]
        y = args_list[0]*(x[i]**4)+args_list[1]*(x[i]**3)+args_list[2]*(x[i]**2)+args_list[3]*x[i]+args_list[4]
        result[i][1] = y
    return result
'''
# 0.0003237315595627103, -0.1867841838765627, 39.162065866342765, -3265.738829238023, 97196.05934308255
# 0.0007952820277722974, -0.4631092407615735, 105.07041271121713, -10700.284455623947, 422251.50739980396
result =[]
args_list = [0.0003237315595627103, -0.1867841838765627, -39.162065866342765, -3265.738829238023, 97196.05934308255]
num, x = cal_quartic_ik(args_list)
result = get_result(args_list, x)
print("num: ", num)
print("x: ", x)
print("result: ", result)'''