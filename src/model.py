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