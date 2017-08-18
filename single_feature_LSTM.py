'''
cs542: rainlevel prediction
this file contains three functions:
compute_RMSE
create_data
find_max
'''
import itertools
import numpy
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt



'''
input parameters:(latitude,longitude,data,time_step,batch_size)
data is a 16*17*16800 array containning all the data.

return parameters (trainScore, testScore,randomError,trainPredict,testPredict)
trainScore: root mean squared error for training set.
testScore: root mean squared error for testing set.
randomError: root mean squared error for guessing with average value.
trainPredict: a array with prediction of rainlevel at each day in trainig set.
testPredict: a array with prediction of rainlevel at each day in testing set.
'''
def compute_RMSE(lat,lon,data,look_back,batch_size):
    dataset=[]
    total =0
    avg_arr=[]

    #sum all the rainlevel at speific location
    for x1 in range (len(data)):
        value = data[x1][lat][lon]
        if (type(value).__name__=='float32'):
            total+= value
            value=[value]
            dataset.append([y for y in value])
    # get the average of that speific location
    avg = total/ len(data)
    
    # find the root mean squared error for predicting with average rain level
    for x1 in range (len(data)):
        value = data[x1][lat][lon]
        if (type(value).__name__=='float32'):
            value=[avg]
            avg_arr.append([y for y in value])
        
    rms = sqrt(mean_squared_error(dataset, avg_arr))


    #truncate the dataset so that it can be divisible by batch size
    truncat= find_max(batch_size,look_back,len(dataset))
    if truncat is None:
        truncat = len(dataset)
    dataset=dataset[(len(dataset)-truncat):]

    #normalize the data so it fits between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)


    #split the data into 70% training set and 30% testing set.
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[:train_size,:], dataset[train_size:,:]

    # reshape into X=t and Y=t+time_step
    trainX, trainY = create_data(train, look_back)
    testX, testY = create_data(test, look_back)
    
    # reshape input to be samples*time steps*number of features
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    #initialize stateful LSTM model
    model = Sequential()
    model.add(LSTM(30, batch_input_shape=(batch_size, look_back, 1),stateful=True,  return_sequences=True,kernel_initializer='random_uniform'))
    #model.add(Activation('relu'))
    model.add(LSTM(10, batch_input_shape=(batch_size, look_back, 1),stateful=True,kernel_initializer='random_uniform'))
    #model.add(Activation('relu'))
    model.add(Dense(1))
    #model.add(Activation('relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=10, batch_size=batch_size,  verbose=2, shuffle=False)
    
    # make predictions with model
    trainPredict = model.predict(trainX, batch_size=batch_size)
    testPredict = model.predict(testX, batch_size=batch_size)
    # converts predictions back to acutal vlaues from 0 to 1.
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    #print('Train Error: %.2f Root mean squared error' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    #print('Test Error: %.2f Root mean squared error' % (testScore))


    trainPredictPlot = trainPredict
    testPredictPlot = testPredict

    return (trainScore, testScore,rms,trainPredict,testPredict,dataset,scaler)

'''
function to separate data into input and output.
For example, there is a sequence of data recording 5 days
of rain level: [1,2,3,4,5].The time step is 3, create_data() outputs
the following array:
X:[1,2,3] , Y:[4]
X:[2,3,4], Y:[5]
'''
def create_data(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

    
'''
find the max number of the data array that can be divisible by batch size
'''
def find_max(batch_szie,look_back,length):
    while(length>0):
        train=length*0.7
        train= train-look_back
        test=length*0.3
        test = test - look_back
        if train%batch_szie==0 and test % batch_szie==0:
                return length
        
        length=length-1

