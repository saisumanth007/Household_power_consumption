import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras, math
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

def Basic_LSTM(in_dim, out_dim):

	model = Sequential()
	model.add(LSTM(50, input_shape=(in_dim, 1), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(100,return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(out_dim))
	model.compile(loss='mse', optimizer='rmsprop')

	return model

def RandomForest():

	regr = RandomForestRegressor(max_depth=6,random_state=0)
	return regr

def Conv1d_test_1(in_dim, out_dim):

	inp = Input(shape = (in_dim,1))
	hour = Input(shape = (1,))
	month = Input(shape = (1,))
	week_day = Input(shape = (1,))
	month_day = Input(shape = (1,))
	weekend_flag = Input(shape = (1,))

	conv_1 = Conv1D(4,10,activation='relu')(inp)
	pool1 = MaxPooling1D(2)(conv_1)
	flatten = Flatten()(pool1)

	conc = Concatenate(1)([flatten, hour, month, week_day, month_day, weekend_flag])
	dense1 = Dense(20, activation = 'relu')(conc)
	dense2 = Dense(20, activation = 'relu')(dense1)
	out = Dense(out_dim)(dense2)

	input_layers = [inp, hour, month, week_day, month_day, weekend_flag]
	model = Model(inputs = input_layers,outputs = out)
	model.compile(optimizer="adam",loss="mae")

	return model

