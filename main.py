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
#x, y = series_value[:, :-1], series_value[:, -1]
#print("x: ", x)
#print("y: ", y)

plt.plot(series_index, series_value, '--bo', label='Training Average Sum of Squared Errors')
#plt.show()
print("test")
print("\nseries:\n", series)
#series.plot()
plt.show()
