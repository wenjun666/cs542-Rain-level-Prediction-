'''
cs542: rainlevel prediction
user_interface.py
This file is a user interface that prompt user for a latiutde and longitude, and retrun the training error and
testing error.
Run this file.
'''

import itertools
from netCDF4 import Dataset
import numpy
from single_feature_LSTM import compute_RMSE,create_data,find_max

dataset = Dataset('pnwrain.nc',mode='r')
lats = dataset.variables['lat'][:]
lons = dataset.variables['lon'][:]
time = dataset.variables['time'][:]
data = dataset.variables['data'][:]

lons=abs(lons)

lon=0
lat=0

def start_program():
    print("Please enter latitude bettween 41.90 and 50.00")
    print("Please enter longitude bettween -125.94 and -115.94")
    
    
    lat = float(input('Enter latitude: '))
    lon = abs(float (input('Enter longitude: ')))
    while(lat<41.9 or lat >50):
        lat = float(input('Enter valid latitude: '))
    
    while(lon<115.94 or lon > 125.94):
        lon = abs(float (input('Enter valid longitude: ')))
        
    lon_diff=[]
    lat_diff=[]
    for i in range(len(lons)):
        lon_diff+=[abs(lon-lons[i])]
    lon_pick =lons[lon_diff.index(min(lon_diff))]
    lon_index=lon_diff.index(min(lon_diff))

    for i in range(len(lats)):
        lat_diff+=[abs(lat-lats[i])]
    
    lat_pick = lats[lat_diff.index(min(lat_diff))]
    lat_index=lat_diff.index(min(lat_diff))

    trainScore, testScore,rms,trainPredict,testPredict,d,s =compute_RMSE(lat_index,lon_index,data,30,30)
    print('Training set: %.2f Root mean squared error' % (trainScore))
    print('Testing set: %.2f Root mean squared error' % testScore)

start_program()
    
