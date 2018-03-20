import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import math
from tqdm import tqdm

def preprocess():
	df = pd.read_csv("../data/household_power_consumption.txt", sep = ";")
	df = df[(df.Global_active_power!="?")]
	df["Timestamp"] = df["Date"].astype(str)+" "+df["Time"]
	df.Timestamp = pd.to_datetime(df.Timestamp, format = "%d/%m/%Y %H:%M:%S")
	df.Date = pd.to_datetime(df.Date, format = "%d/%m/%Y")

	df.Global_active_power = df.Global_active_power.astype(float)
	df.Global_reactive_power = df.Global_reactive_power.astype(float)
	df.Voltage = df.Voltage.astype(float)
	df.Sub_metering_1 = df.Sub_metering_1.astype(float)
	df.Sub_metering_2 = df.Sub_metering_2.astype(float)
	df.Sub_metering_3 = df.Sub_metering_3.astype(float)

	return df

def day_plot(date):
	day = df[(df.Date == date)]
	plt.plot(day['Timestamp'], day['Global_active_power'])
	plt.xticks(rotation='vertical')

def extract_data(df, in_dim, out_dim):

	# seq_len = 50
	# batch_size = 32

	dataset = df.Global_active_power
	dataset = np.asarray(dataset)
	
	sequence_length = in_dim + out_dim
	result = []
	for index in range(len(dataset) - sequence_length):
	    result.append(dataset[index: index + sequence_length])

	result = np.array(result)

	train_size = int(len(dataset) * 0.98)
	test_size = int((len(dataset) - train_size) * 0.5)
	valid_size = len(dataset) - train_size - test_size
	train = result[:int(train_size), :]
	x_train = train[:, :in_dim]
	y_train = train[:, in_dim:]

	x_valid = result[int(train_size):int(train_size) + valid_size, :in_dim]
	y_valid = result[int(train_size):int(train_size) + valid_size, in_dim]

	x_test = result[int(train_size) + valid_size:, :in_dim]
	y_test = result[int(train_size) + valid_size:, in_dim]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	return x_train, y_train, x_valid,  y_valid, x_test, y_test