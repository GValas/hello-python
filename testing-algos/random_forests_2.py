# https://chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

import multiprocessing

headers = ['A' + str(i) for i in range(1, 62)]

filename = 'sonar.all-data.csv'
df = pd.read_csv(filename, sep=',', names=headers)
df['A61'] = pd.factorize(df['A61'])[0]
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

train, test = df[df['is_train'] == True], df[df['is_train'] == False]

features = df.columns[0:60]
labels = df.columns[60:61]

clf = RandomForestClassifier(n_estimators=100,n_jobs=-1, verbose=2)
clf.fit(train[features], train[labels])
z = clf.predict(test[features])
y = test[labels]


print(np.average(z== y.values.flatten()))


# print(list(zip(train[features], clf.feature_importances_)))
