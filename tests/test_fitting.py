import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt



# ECCG(x,y)  屏幕真实坐标（X，Y）
x = np.array([7,6,5,4,3,2,1])
X = np.array([1,2,3,4,5,6,7])

plt.scatter(x, X, s=5, color='black', alpha=0.8)
# 拟合函数的阶数。2次函数，10次函数。。。自己选的
degrees = [2, 10]
for degree in degrees:
    clf = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                    ('linear', linear_model.RidgeCV())])
    # 自变量需要二维数组
    clf.fit(x[:, np.newaxis], X)
    predict_y = clf.predict(x[:, np.newaxis])
    plt.plot(x, predict_y, linewidth=2, label=degree)

plt.legend()
plt.show()
