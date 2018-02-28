import pandas as pd
import math, quandl
import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn.linear_model import LinearRegression

data_frame = quandl.get('WIKI/GOOGL')

data_frame = data_frame[['Adj. Open', 'Adj. High', 'Adj. Close', 'Adj. Low', 'Adj. Volume']]

data_frame['HL_PTC'] = (data_frame['Adj. High'] - data_frame['Adj. Close']) / data_frame['Adj. Close'] * 100

data_frame['PTC_change'] = (data_frame['Adj. Close'] - data_frame['Adj. Open']) / data_frame['Adj. Open'] * 100

data_frame = data_frame[['Adj. Close', 'HL_PTC', 'PTC_change', 'Adj. Volume']]

forcast_col = 'Adj. Close'
data_frame.fillna(-99999, inplace=True)
forcast_out = int(math.ceil(0.01 * len(data_frame)))
data_frame['label'] = data_frame[forcast_col].shift(-forcast_out)
data_frame.dropna(inplace=True)
X = np.array(data_frame.drop(['label'], 1))
y = np.array(data_frame['label'])
X = preprocessing.scale(X)
# # X = X[:-forcast_out + 1]
# data_frame.dropna(inplace=True)
# y = np.array(data_frame['label'])
# print(len(X), len(y))
X_train, X_test, y_train, y_test= cross_validation.train_test_split(X,y,test_size=0.2)
classifer = LinearRegression()
classifer.fit(X_train,y_train)
classifer.score(X_test,y_test)
accuracy = classifer.score(X_test,y_test)
print(forcast_out)
print(accuracy)
