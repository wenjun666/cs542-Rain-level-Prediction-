'''
cs542: rainlevel prediction
this file contains function plot_fig(latitude, longitude)
both latitude and longitude are indexs.
The function will plot the prediction of last 100 days of rain agains the actual rain level
'''
from netCDF4 import Dataset
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from single_feature_LSTM import compute_RMSE,create_data,find_max

nc = Dataset('pnwrain.nc',mode='r')
#print (dataset.dimensions.keys())
lats = nc.variables['lat'][:]
lons = nc.variables['lon'][:]
time = nc.variables['time'][:]
data = nc.variables['data'][:]

data=data[:16800]

def plot_fig(lat,lon):
    
    trainScore30, testScore30,rms,trainPredict30,testPredict30,dataset,scaler=compute_RMSE(lat,lon,data,30,30)
    trainPredictPlot30 = trainPredict30
    testPredictPlot30 = testPredict30
    
    trainScore3, testScore3,rms,trainPredict3,testPredict3,dataset,scaler=compute_RMSE(lat,lon,data,3,3)
    trainPredictPlot3 = trainPredict3
    testPredictPlot3 = testPredict3

    plt.figure(figsize=(10,4))
    plt.title('Last 100 Predictions')
    datasetPlot = dataset[len(dataset) - 100:len(dataset),:]
    plt.plot(scaler.inverse_transform(datasetPlot), color='b', label='Actual')
    
    testPredictPlot30 = testPredictPlot30[len(testPredictPlot30) - 100:len(testPredictPlot30),:]
    plt.plot(testPredictPlot30, color='r', label='Prediction look_back =30')
    print('Testing set root mean squared error(30 time_steps): %.2f' % (trainScore30))
    testPredictPlot3 = testPredictPlot3[len(testPredictPlot3) - 100:len(testPredictPlot3),:]
    print('Testing set root mean squared error(3 time_steps): %.2f' % (trainScore3))
    plt.plot(testPredictPlot3, color='g', label='Prediction look_back =3')

    plt.grid(True)
    plt.legend()
    plt.show()


plot_fig(4,3)
