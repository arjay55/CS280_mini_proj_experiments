from ngboost import NGBRegressor
from scipy.stats import expon
from scipy.stats import multivariate_normal
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ngboost.scores import CRPScore, LogScore
from ngboost.distns import Exponential
from ngboost.distns import MultivariateNormal
from pandas import read_csv
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd

series = read_csv('ProcessedDataCenterHall.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
#series = read_csv('ProcessedBioEngineeringPV.csv', header=0, parse_dates=True)

x_series_index = series.index
y_series_value = series.values

print("series_index dates: ", x_series_index)
print("series_value power values: ", y_series_value)
print("series_index dates shape: ", x_series_index.shape)
print("series_value power values shape: ", y_series_value.shape)


# plt.plot(series_index, series_value, '--bo', label='Training Average Sum of Squared Errors')
# plt.show()

x_date_numerical = series.index.to_julian_date()
print("date_numerical: ", x_date_numerical)

# Diabetes
# X, Y = load_diabetes(return_X_y=True)
# print("X: ", X)
# print("Y: ", Y)
# print("X shape: ", X.shape)
# print("Y shape: ", Y.shape)

#print("y_date_new: ", y_date_new)

x_date_new = x_date_numerical.values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(x_date_new, y_series_value, test_size=0.2, shuffle=False)
#X_train, X_test, Y_train, Y_test = train_test_split(x_series_value, y_date_numerical, test_size=0.2)
print("X_test: ", X_test)
print("Y_actual: ", Y_test)

ngb = NGBRegressor(Dist=MultivariateNormal(k=2), Score=LogScore, verbose=True).fit(X_train, Y_train)
Y_preds = ngb.predict(X_test)
Y_dists = ngb.pred_dist(X_test)

print("Y_preds: ", Y_preds)
print("Y_dist: ", Y_dists)

mean = Y_dists.mean
sample = Y_dists.rv()
scipy_list = Y_dists.scipy_distribution()

print("mean: ", mean)
print("sample: ", sample)

# test Mean Squared Error
test_MSE = mean_squared_error(Y_preds, Y_test)
print('Test MSE', test_MSE)

# test Negative Log Likelihood
test_NLL = -Y_dists.logpdf(Y_test).mean()
print('Test NLL', test_NLL)
