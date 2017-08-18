'''
cs542: rainlevel prediction
iter_dataset_LSTM.py
Run this file to compute 190 blocks of
coordinates of training error, testing error,
guessing error and save them in result.csv.
'''

import itertools
from netCDF4 import Dataset
import numpy

import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from math import sqrt
from single_feature_LSTM import compute_RMSE,create_data,find_max

nc = Dataset('pnwrain.nc',mode='r')
lats = nc.variables['lat'][:]
lons = nc.variables['lon'][:]
time = nc.variables['time'][:]
data = nc.variables['data'][:]

data=data[:16800]
combo = []
result = []
combo=[[2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11],
       [2, 12], [2, 13], [2, 14], [3, 5], [3, 6], [3, 7], [3, 8],
       [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15],
       [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9],
       [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [5, 3], [5, 4],
       [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12],
       [5, 13], [5, 14], [5, 15], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7],
       [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15],
       [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10],
       [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [8, 3], [8, 4], [8, 5],
       [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13],
       [8, 14], [8, 15], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8],
       [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [10, 3],
       [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10],
       [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [11, 3], [11, 4],
       [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11],
       [11, 12], [11, 13], [11, 14], [11, 15], [12, 3], [12, 4], [12, 5],
       [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12],
       [12, 13], [12, 14], [12, 15], [13, 3], [13, 4], [13, 5], [13, 6],
       [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13],
       [13, 14], [13, 15], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6],
       [14, 7], [14, 8], [14, 9], [14, 10],
       [14, 11], [14, 12], [14, 13], [14, 14],
       [14, 15], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6],
       [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12],
       [15, 13], [15, 14], [16, 3], [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [16, 13]]


for i in range(len(combo)):
    la, lon = combo[i]
    trainScore,testScore,rms,trainPredict,testPredict,d,s =compute_RMSE(la,lon,data,30,30)
    value = [combo[i][0],combo[i][1],trainScore,testScore,rms]
    result.append([y for y in value])
    numpy.savetxt("result.csv", result, delimiter=",")






