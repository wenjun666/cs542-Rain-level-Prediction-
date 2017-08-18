'''
cs542: rainlevel prediction
multi_features_LSTM.py
This file is the optimaztion of the single feature LSTM model.
Run this file to make a prediction at latitude index 7 and longitude at
index 7. This file computes the prediction with 5 features.
index [7,6],[7,5],[7,7],[8,6],[6,6]
'''
import itertools
from netCDF4 import Dataset
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#define all the coordinates we are going to use later
lat = 7
lon = 6
lat2 = 7
lon2 = 5
lat3 = 7
lon3 = 7
lat4 = 8
lon4 =6
lat5 =6
lon5 =6

#open the ncnet file. 
nc = Dataset('pnwrain.nc',mode='r')


lats = nc.variables['lat'][:]
lons = nc.variables['lon'][:]
time = nc.variables['time'][:]
data = nc.variables['data'][:]
data=data[:16800]

#set time_step and batch size
time_step = 30
batch_size = 30


'''
function to separate data into input and output.
For example, there is a sequence of data recording 5 days
of rain level: [1,2,3,4,5].The time step is 3, create_data() outputs
the following array:
X:[1,2,3] , Y:[4]
X:[2,3,4], Y:[5]
'''
def create_data(dataset, dataset2, dataset3,dataset4,dataset5, time_step=1):
    dataX, dataY = [], []#len(dataset)-time_step
    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 0]
        b = dataset2[i:(i+time_step), 0]
        c = dataset3[i:(i+time_step), 0]
        d = dataset4[i:(i+time_step), 0]
        e = dataset5[i:(i+time_step), 0]
        tmp=[]
        for x in range(len(a)):
            tmp.append(a[x])
            tmp.append(b[x])
            tmp.append(c[x])
            tmp.append(d[x])
            tmp.append(e[x])
        #a=numpy.append(a,b)
        dataX.append(tmp)
        dataY.append(dataset[i + time_step, 0])
    
    #print("in dataset",dataX[:1])
    return numpy.array(dataX), numpy.array(dataY)


'''
find the max number of the data array that can be divisible by batch size
'''
def find_max(batch_szie,time_step,length):
    while(length>0):
        train=int(length*0.7)
        train= train-time_step
        test=int(length*0.3)
        test = test - time_step
        if train%batch_szie==0 and test % batch_szie==0:
                
                return length
        
        length=length-1
    
dataset=[]
dataset2=[]
dataset3=[]
dataset4=[]
dataset5=[]

# for loop to iterate data array and get all the values into dataset array
for x1 in range (len(data)):
    value = data[x1][lat][lon]
    value2= data[x1][lat2][lon2]
    value3 = data[x1][lat3][lon3]
    value4 = data[x1][lat4][lon4]
    value5 = data[x1][lat5][lon5]
    
    if (type(value).__name__=='float32'):
        value=[value]
        dataset.append([y for y in value])
    
    if (type(value2).__name__=='float32'):
        value2=[value2]
        dataset2.append([y for y in value2])
        
    if (type(value3).__name__=='float32'):
        value3=[value3]
        dataset3.append([y for y in value3])
    if (type(value4).__name__=='float32'):
        value4=[value4]
        dataset4.append([y for y in value4])
    if (type(value5).__name__=='float32'):
        value5=[value5]
        dataset5.append([y for y in value5])    

#truncate the dataset so that it can be divisible by batch size
truncat= find_max(batch_size,time_step,len(dataset))
if truncat is None:
    truncat = len(dataset)
    
dataset=dataset[(len(dataset)-truncat):]

total = dataset+dataset2+dataset3+dataset4+dataset5
#normalize the data so it fits between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
total = scaler.fit_transform(total)
dataset=(total[0:16800])
dataset2=(total[16800:16800*2])
dataset3=(total[16800*2:16800*3])
dataset4=(total[16800*3:16800*4])
dataset5=(total[16800*4:])

train_size = int(len(dataset) * 0.7)
train, test = dataset[0:train_size,:], dataset[train_size:,:]
train2, test2 = dataset2[0:train_size,:], dataset2[train_size:,:]
train3, test3 =dataset3[0:train_size,:], dataset3[train_size:,:]
train4, test4 = dataset4[0:train_size,:], dataset4[train_size:,:]
train5, test5 =dataset5[0:train_size,:], dataset5[train_size:,:]

# reshape into X=t and Y=t+time_step
trainX, trainY = create_data(train,train2,train3,train4,train5,time_step)
testX, testY = create_data(test,test2,test3,test4,test5, time_step)

# reshape input to be samples*time steps*number of features
trainX = numpy.reshape(trainX, (trainX.shape[0], time_step,5))
testX = numpy.reshape(testX, (testX.shape[0], time_step,5))

#initialize stateful LSTM model
model = Sequential()
model.add(LSTM(20, batch_input_shape=(batch_size, time_step, 5),stateful=True,  return_sequences=True))
model.add(Activation('relu'))
model.add(LSTM(20, batch_input_shape=(batch_size, time_step, 5),stateful=True,return_sequences=True))
model.add(Activation('relu'))
model.add(LSTM(20, batch_input_shape=(batch_size, time_step, 5),stateful=True))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
model.fit(trainX, trainY, epochs=10, batch_size=batch_size,  verbose=2, shuffle=False)

# predict of training set
trainPredict = model.predict(trainX, batch_size=batch_size)
# predict of testing set
testPredict = model.predict(testX, batch_size=batch_size)
# converts predictions back to acutal vlaues from 0 to 1.
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Training set root mean squared error: %.2f ' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Testing set root mean squared error: %.2f' % (testScore))
trainPredictPlot = trainPredict
testPredictPlot = testPredict


# plot last 100 predictions
plt.figure(figsize=(10,4))
plt.title('Last 100 Predictions')
datasetPlot = dataset[len(dataset) - 100:len(dataset),:]
plt.plot(scaler.inverse_transform(datasetPlot), color='b', label='Actual')
testPredictPlot = testPredictPlot[len(testPredictPlot) - 100:len(testPredictPlot),:]
plt.plot(testPredictPlot, color='r', label='Prediction')
plt.grid(True)
plt.legend()
plt.show()



