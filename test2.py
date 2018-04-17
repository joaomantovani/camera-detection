# coding=utf-8
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('features.csv', header=0)
# data = data.dropna()

# print data.shape
# print list(data.columns)
#
# print data.head()

# print data['tirado_pela_moto_X'].value_counts()
#
# print data.groupby('tirado_pela_moto_X').mean()

sns.countplot(x="tirado_pela_moto_X", data=data, palette="hls")
# plt.show()


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

data_final_vars = data.columns.values.tolist()
y = ["tirado_pela_moto_X"]
ignore = ["tirado_pela_moto_X", "photo_by", "filename", "fullpath", "tirado_pelo_iphoe"]

X = [i for i in data_final_vars if i not in ignore]

rfe = RFE(logreg, 18)
rfe = rfe.fit(data[X], data[y])
print(rfe.support_)
print(rfe.ranking_)

cols = ["r_vertical_mean", "r_vertical_skewness", "g_vertical_mean", "g_vertical_skewness", "b_vertical_mean",
        "r_horizontal_mean", "r_horizontal_skewness", "g_horizontal_mean", "g_horizontal_skewness", "b_horizontal_mean",
        "b_horizontal_skewness", "r_diagonal_variance", "r_diagonal_skewness", "g_diagonal_mean",
        "g_diagonal_variance", "g_diagonal_skewness", "b_diagonal_variance", "b_diagonal_skewness"]

X = data[cols]
y = data['tirado_pela_moto_X']

#
# Implementing the model
#
import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

logit_model = sm.Logit(y, X)
result = logit_model.fit()

print(result.summary())


#
# Logistic Regression Model Fitting
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
results = logreg.fit(X_train, y_train)

print ""
print results
print ""


# Predicting the test set results and calculating the accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


#
# Cross Validation
#

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)

print ""
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

#
# Confusion Matrix
#
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print ""
print(confusion_matrix)


#
# Compute precision, recall, F-measure and support
#
from sklearn.metrics import classification_report
print ""
print(classification_report(y_test, y_pred))


#
# ROC Curve
#

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
# plt.show()

globvar = 0

def set_globvar_to_one():
    global globvar    # Needed to modify global copy of globvar
    globvar += 1

def print_globvar():
    print(globvar)

def calc(arr):
    y_pred = logreg.predict([arr])

    if y_pred == 0:
        print "Resultado: Não é motoG"
    else:
        print "Resultado: é motoG"
        set_globvar_to_one()


print_globvar()