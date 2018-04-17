import pandas as pd
import numpy as np
from sklearn import preprocessing

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns


# Training dataset
X = np.loadtxt('features_motox.csv', delimiter=",", skiprows=1, usecols=(
    3, 7
))
y = np.loadtxt('features_motox.csv', delimiter=",", skiprows=1, usecols=(39))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


logreg = LogisticRegression()
results = logreg.fit(X_train, y_train)  # type: object

print logreg.predict([[0.6819192346, 0.7091948608]])

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


print X.unique()