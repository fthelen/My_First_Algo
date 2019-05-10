import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

sp500 = pd.read_csv(r'tests\SP500.csv', parse_dates=True, index_col=0)
sp500['log_ret'] = np.log(sp500['Adj Close']/sp500['Adj Close'].shift(1))

sun = pd.read_csv(r'tests\mean_sunspots.csv', sep=';')
sun['date'] = sun['year'].map(str) + str('-') + sun['month'].map(str) + str('-') + str('01')
sun.set_index('date',inplace=True)
sun.drop(columns=['year','month','year_frac'],inplace=True)

df_full = pd.concat([sp500['log_ret'],sun['mean']],axis=1,join='inner')
df_full['log_ret'] = df_full['log_ret'].shift(-1)
df_full.dropna(inplace=True)

# # Shape (instances, attributes)
# print(df_full.shape)

# Head
print(df_full.head())
print(df_full.tail())

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

# Scatter plot matrix
scatter_matrix(df_full)
plt.show()

# Split-out validation df_full
array = df_full.values
# print(array)
X, Y = array[:,1], array[:,0]
# print(X)
# print(Y)

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# # Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# # plt.show()

# # Make predictions on validation df_full
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))


# style.use('ggplot')
# df['Adj Close'].plot()
# plt.show()
