import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt
import numpy as np



n = 1000
X = np.linspace(-5,5,n)
y = np.exp(X) # + np.random.normal(0, .1, X.size)


regr = RandomForestRegressor(n_estimators=10)
regr.fit(X.reshape(X.size,1), y)


X2 = np.linspace(-6,6,2*n)
y2 = np.exp(X2)

z = []
for x in X2:
    z.append(regr.predict(x))

plt.plot(X2,y2)
plt.plot(X2,z)

plt.show()
