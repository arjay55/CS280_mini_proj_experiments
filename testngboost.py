from ngboost import NGBRegressor

from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from pandas import read_csv
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np

series = read_csv('ProcessedBioEngineeringPV.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
#series = read_csv('ProcessedBioEngineeringPV.csv', header=0, parse_dates=True)
series_value = series.values
series_index = series.index
print("series_value: ", series_value)
print("series_index: ", series_index)
# x, y = series_value[:, :-1], series_value[:, -1]
# print("x: ", x)
# print("y: ", y)

plt.plot(series_index, series_value, '--bo', label='Training Average Sum of Squared Errors')
#plt.show()
#print("\nseries:\n", series)
#series.plot()

plt.show()
print("load boston: ", np.array(load_boston()))

X, Y = load_boston(return_X_y=True)
print("X: ", X)
print("Y: ", Y)
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ngb = NGBRegressor().fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test).mean()
print('Test NLL', test_NLL)


