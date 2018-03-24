import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import math
from tqdm import tqdm

def preprocess():
	df = pd.read_csv("../data/household_power_consumption.txt", sep = ";")
	df = df.replace("?", np.nan)
	df = df.ffill()               # Replace "?" with previous valid values

	df["Timestamp"] = df["Date"].astype(str)+" "+df["Time"]
	df.Timestamp = pd.to_datetime(df.Timestamp, format = "%d/%m/%Y %H:%M:%S")
	df.Date = pd.to_datetime(df.Date, format = "%d/%m/%Y")

	df.Global_active_power = df.Global_active_power.astype(float)
	df.Global_reactive_power = df.Global_reactive_power.astype(float)
	df.Voltage = df.Voltage.astype(float)
	df.Sub_metering_1 = df.Sub_metering_1.astype(float)
	df.Sub_metering_2 = df.Sub_metering_2.astype(float)
	df.Sub_metering_3 = df.Sub_metering_3.astype(float)

	df = df.groupby([df.Timestamp.dt.date, df.Timestamp.dt.hour]).mean()
	df.index = df.index.set_names(['Date', 'Hour'])
	df.reset_index(inplace=True)
	df['Date'] = pd.to_datetime(df['Date'])
	df['Month'] = df['Date'].dt.month
	df['DayofWeek'] = df['Date'].dt.dayofweek
	df['DayofMonth'] = df['Date'].dt.day
	df['WeekendFlag'] = [1 if x == 'Saturday' or x == 'Sunday' else 0 for x in df['Date'].dt.weekday_name]
	df.Hour = df.Hour.astype(float)

	return df

def day_plot(date):
	day = df[(df.Date == date)]
	plt.plot(day['Timestamp'], day['Global_active_power'])
	plt.xticks(rotation='vertical')

def extract_data(df, in_dim, out_dim):

	# seq_len = 50
	# batch_size = 32

	dataset = np.asarray(df.Global_active_power)
	
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
	y_valid = result[int(train_size):int(train_size) + valid_size, in_dim:]

	x_test = result[int(train_size) + valid_size:, :in_dim]
	y_test = result[int(train_size) + valid_size:, in_dim:]

	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

	return result, x_train, y_train, x_valid,  y_valid, x_test, y_test

def extract_data_hourly(df, in_dim, out_dim):

	sequence_length = in_dim + out_dim
	result = []
	hour_final = []
	month_final = []
	week_day_final = []
	month_day_final = []
	weekend_final = []

	power = np.array(df.Global_active_power)
	hour = np.array(df.Hour)
	month = np.array(df.Month)
	dayofweek = np.array(df.DayofWeek)
	dayofmonth = np.array(df.DayofMonth)
	weekendflag = np.array(df.WeekendFlag)


	for index in range(len(power) - sequence_length):
	    result.append(power[index: index + sequence_length])
	    hour_final.append(hour[index + in_dim + 1])
	    month_final.append(month[index + in_dim + 1])
	    week_day_final.append(dayofweek[index + in_dim + 1])
	    month_day_final.append(dayofmonth[index + in_dim + 1])
	    weekend_final.append(weekendflag[index + in_dim + 1])


	result = np.array(result)
	hour_final = np.array(hour_final)
	month_final = np.array(month_final)
	week_day_final = np.array(week_day_final)
	month_day_final = np.array(month_day_final)
	weekend_final = np.array(weekend_final)


	train_size = 20000
	valid_size = 6671

	x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []

	power_train = result[:train_size, :in_dim]
	hour_train = hour_final[:train_size]
	month_train = month_final[:train_size]
	week_day_train = week_day_final[:train_size]
	month_day_train = month_day_final[:train_size]
	weekend_train = weekend_final[:train_size]
	power_train = np.reshape(power_train, (power_train.shape[0], power_train.shape[1], 1))
	hour_train = np.reshape(hour_train, (hour_train.shape[0], 1))
	month_train = np.reshape(month_train, (month_train.shape[0], 1))
	week_day_train = np.reshape(week_day_train, (week_day_train.shape[0], 1))
	month_day_train = np.reshape(month_day_train, (month_day_train.shape[0], 1))
	weekend_train = np.reshape(weekend_train, (weekend_train.shape[0], 1))
	x_train = [power_train, hour_train, month_train, week_day_train, month_day_train, weekend_train]
	y_train = result[:train_size, in_dim:]

	power_valid = result[int(train_size):int(train_size) + valid_size, :in_dim]
	hour_valid = hour_final[int(train_size):int(train_size) + valid_size]
	month_valid = month_final[int(train_size):int(train_size) + valid_size]
	week_day_valid = week_day_final[int(train_size):int(train_size) + valid_size]
	month_day_valid = month_day_final[int(train_size):int(train_size) + valid_size]
	weekend_valid = weekend_final[int(train_size):int(train_size) + valid_size]
	power_valid = np.reshape(power_valid, (power_valid.shape[0], power_valid.shape[1], 1))
	hour_valid = np.reshape(hour_valid, (hour_valid.shape[0], 1))
	month_valid = np.reshape(month_valid, (month_valid.shape[0], 1))
	week_day_valid = np.reshape(week_day_valid, (week_day_valid.shape[0], 1))
	month_day_valid = np.reshape(month_day_valid, (month_day_valid.shape[0], 1))
	weekend_valid = np.reshape(weekend_valid, (weekend_valid.shape[0], 1))
	x_valid = [power_valid, hour_valid, month_valid, week_day_valid, month_day_valid, weekend_valid]
	y_valid = result[int(train_size):int(train_size) + valid_size, in_dim:]

	power_test = result[int(train_size) + valid_size:, :in_dim]
	hour_test = hour_final[int(train_size) + valid_size:]
	month_test = month_final[int(train_size) + valid_size:]
	week_day_test = week_day_final[int(train_size) + valid_size:]
	month_day_test = month_day_final[int(train_size) + valid_size:]
	weekend_test = weekend_final[int(train_size) + valid_size:]
	power_test = np.reshape(power_test, (power_test.shape[0], power_test.shape[1], 1))
	hour_test = np.reshape(hour_test, (hour_test.shape[0], 1))
	month_test = np.reshape(month_test, (month_test.shape[0], 1))
	week_day_test = np.reshape(week_day_test, (week_day_test.shape[0], 1))
	month_day_test = np.reshape(month_day_test, (month_day_test.shape[0], 1))
	weekend_test = np.reshape(weekend_test, (weekend_test.shape[0], 1))
	x_test = [power_test, hour_test, month_test, week_day_test, month_day_test, weekend_test]
	y_test = result[int(train_size) + valid_size:, in_dim:]

	return x_train, y_train, x_valid,  y_valid, x_test, y_test