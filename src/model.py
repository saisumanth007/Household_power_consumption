import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras, math
from keras.models import Sequential, Model
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

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

def squeeze_arch(in_dim, out_dim):
	ip = Input(shape=(in_dim, 1))

	# x = Masking()(ip)
	# x = LSTM(60)(ip)
	# x = Dropout(0.8)(x)
	x = LSTM(60, input_shape=(in_dim, 1), return_sequences=True)(ip)
	x = Dropout(0.2)(x)
	x = LSTM(100,return_sequences=False)(x)
	x = Dropout(0.2)(x)

	# y = Permute((2, 1))(ip)
	y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
	y = BatchNormalization()(y)
	y = Activation('relu')(y)
	y = squeeze_excite_block(y)
	y_1 = add([y, ip])

	y = Conv1D(128, 5, padding='same', kernel_initializer='he_uniform')(y_1)
	y = BatchNormalization()(y)
	y = Activation('relu')(y)
	y = squeeze_excite_block(y)
	y_2 = add([y, y_1])

	y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y_2)
	y = BatchNormalization()(y)
	y = Activation('relu')(y)
	y = add([y, y_2])

	y = GlobalAveragePooling1D()(y)

	x = concatenate([x, y])
	dense1 = Dense(100, activation = 'relu')(x)
	dense2 = Dense(80, activation = 'relu')(dense1)
	# dense3 = Dense(80, activation = 'relu')(dense2)
	out = Dense(out_dim)(dense2)

	model = Model(inputs = ip,outputs = out)
	model.compile(optimizer="adam",loss="mean_squared_error")

	return model