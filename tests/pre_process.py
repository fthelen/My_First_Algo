import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.plotting import scatter_matrix
from sklearn import model_selection, preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

sp500 = pd.read_csv(r'tests\SP500.csv', parse_dates=True, index_col=0)
sp500['log_ret'] = np.log(sp500['Adj Close']/sp500['Adj Close'].shift(1))

sunspot = pd.read_csv(r'tests\mean_sunspots.csv', parse_dates=True, index_col=0)
moon = pd.read_csv(r'tests\lunar_eclipse.csv', parse_dates=True, index_col=0)
solar = pd.read_csv(r'tests\solar_eclipse.csv', parse_dates=True, index_col=0)

df_full = pd.concat([sunspot['mean_sunspots'],moon['gamma_lunar_eclipse'],solar['gamma_solar_eclipse'],sp500['log_ret']],axis=1,join='inner')
df_full['log_ret'] = df_full['log_ret'].shift(-4)
df_full.dropna(inplace=True)

# Shape (instances, attributes)
print(df_full.shape)

# # Head
# print(df_full.head())
# print(df_full.tail())

# # Descriptions
# print(df_full.describe())

# # Class distribution
# print(df_full.groupby('class').size())

# # Box and whisker plots
# df_full.plot(kind='box',subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# # Histograms
# df_full.hist()
# plt.show()

# # Scatter plot matrix
scatter_matrix(df_full)
plt.show()

# Split-out validation df_full
array = df_full.values
X, Y = array[:,0:3], array[:,-1]

validation_size = 0.20
seed = 5

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

clf = svm.SVR()
clf.fit(X_train,Y_train)
accuracy = clf.score(X_validation,Y_validation)

print(accuracy)


# Test options and evaluation metric
scoring = 'accuracy'

# # Spot Check Algorithms
# models = []
# models.append(('LR', LinearRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))

# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold = model_selection.KFold(n_splits=10, random_state=seed)
# 	cv_results = model_selection.cross_val_score(model, X_train, Y_train, scoring=scoring)
# 	results.append(cv_results)
# 	names.append(name)
# 	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	print(msg)

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# # Make predictions on validation df_full
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))
