import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import r2_score

x = np.array([(3, -2), (-2, -3), (-5, -3), (4, -1), (0, -1), (-5, -1), (2, -1), (-2, -1), (-4, -1)])

y = np.array([50.0, 768.0, 1486.0, 50.0, 768.0, 1486.0, 50.0, 768.0, 1486.0])

clf = SVR(kernel='poly', degree=4, gamma="auto", coef0=0.0, tol=0.001, C=1.0,
          epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
clf.fit(x, y)
y_hat = clf.predict(x)
print(y_hat)
print("得分:", r2_score(y, y_hat))
