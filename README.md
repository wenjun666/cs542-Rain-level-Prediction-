# cs542 Rain level Prediction

# Project Ideas
The purpose of this paper is to report the results of rain prediction using a Long1 TermShortMemorynetworkinanefforttoimprovetheaccuracyofrainprediction.2 The main feature to be optimized with this method is that of resolution. Whereas3 previous methods achieved a resolution of 200 km by 500 km, this method is able4 to reﬁne the resolution to 50 km by 50 km grid size for each of the 272 regions5 analyzed. Various versions of the model were run to insure optimal results, by6 varying parameters such as the amount of historical data for a particular location7 and the number of hidden layers in the network. In the end, the method proposed8 succeeds in achieving an average testing error of 4.9 mm for a 50 km by 509 km region. For further improvements, this model can serve as a good basis for10 additional considerations such as topography.


# Instruction
All codes are written in python 3.5 environment. Side packages are needed to successfully configure the programs: Keras,panda, sklearn, netCDF4, tensorflow.

iter_dataset_LSTM.py:  
	Run this file to compute 190 blocks of coordinates of training error, testing error, guessing error and save them in result.csv.

multi_features_LSTM.py: 
	This file is the optimization version of the single feature LSTM model. Run this file to make a prediction of rain level at latitude index 7 and longitude at index 7. This file computes the prediction with 5 features at indexes: [7,6],[7,5],[7,7],[8,6],[6,6].

plot_figure.py: 
	this file contains function plot_fig(latitude, longitude). Both latitude and longitude are indexes of the array not actual latitude and longitude. This function will plot the prediction of last 100 days of rain level against the actual rain level by comparing 30 time_steps and 3 time_steps.

single_feature_LSTM.py: 
	This file contains three functions: compute_RMSE(), create_data() and find_max().
	compute_RMSE():input parameters:(latitude,longitude,data,time_step,batch_size)
	data is a 16*17*16800 array containing all the data.
	return parameters (trainScore, testScore,randomError,trainPredict,testPredict,dataset,scaler)
	trainScore: root mean squared error for training set.
	testScore: root mean squared error for testing set.
	randomError: root mean squared error for guessing with average value.
	trainPredict: an array with prediction of rain level at each day in training set.
	testPredict: an array with prediction of rain level at each day in testing set.
	dataset: the original data array.
	scaler: the scaler that normalize the dataset between 0 and 1.  

	create_data():
	function to separate data into input and output.
	For example, there is a sequence of data recording 5 days
	of rain level: [1,2,3,4,5]. The time step is 3, create_data() outputs
	the following array:
	X:[1,2,3] , Y:[4]
	X:[2,3,4], Y:[5]

	find_max(): find the max number of the data array that can be divisible by batch size.

user_interface.py:
	Run this file as  a user interface that prompt user for a latitude and longitude, and return the training error and testing error.

