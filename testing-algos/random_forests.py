# https://chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

import multiprocessing


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train'] == True], df[df['is_train'] == False]
features = df.columns[:4]
y = pd.factorize(train['species'])[0]

clf = RandomForestClassifier(n_estimators=100, n_jobs=8)
clf.fit(train[features], y)
z = clf.predict(test[features])
t = clf.predict_proba(test[features])
preds = iris.target_names[clf.predict(test[features])]

print(
pd.crosstab(test['species'], preds,
            rownames=['Actual Species'],
            colnames=['Predicted Species'])



)

print(list(zip(train[features], clf.feature_importances_)))
