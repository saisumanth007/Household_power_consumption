from utils import *
from model import *
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import math

df = preprocess()
in_dim, out_dim = 60, 60
x_train, y_train, x_valid,  y_valid, x_test, y_test = extract_data(df, in_dim, out_dim)

## Basic LSTM
# class DataGenerator(object):
#     def __init__(self, batch_size = 32, shuffle = True):
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#     def generate(self, x, y):
#         while 1:
        
#             imax = int(len(x)/self.batch_size)
#             for i in range(imax):
#                 inp = x[i*batch_size:(i+1)*batch_size]
#                 out = y[i*batch_size:(i+1)*batch_size]
#                 yield inp, out


# batch_size = 100
# params = {'batch_size': 100,
#           'shuffle': True}

# train = np.concatenate([x_train, x_valid])
# lab = np.concatenate([y_train, y_valid])

# training_generator = DataGenerator(**params).generate(train, lab)
# # validation_generator = DataGenerator(**params).generate(x_valid, y_valid)

# # earlyStopping = EarlyStopping(monitor='val_loss', patience = 5)

# model = Basic_LSTM(in_dim, out_dim)

# # model.fit_generator(generator = training_generator,
# #                     steps_per_epoch = len(x_train)//batch_size,
# #                     validation_data = validation_generator,
# #                     validation_steps = len(x_valid)//batch_size, epochs = 20, callbacks = [earlyStopping])
# model.fit_generator(generator = training_generator,
#                     steps_per_epoch = len(train)//batch_size, 
#                     epochs = 100)

##Random Forest
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
# x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

# regr = RandomForest()
# regr.fit(x_train,y_train)

##Conv1d

# class NewDataGenerator(object):
#     def __init__(self, batch_size = 32, shuffle = True):
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#     def generate(self, x, y):
#     	power, hour, month, dayofweek, dayofmonth, weekendflag = x[0], x[1], x[2], x[3], x[4], x[5]
#         while 1:
        
#             imax = int(len(x[0])/self.batch_size)
#             for i in range(imax):
#                 inp_power = x[0][i*batch_size:(i+1)*batch_size]
#                 inp_hour = x[1][i*batch_size:(i+1)*batch_size]
#                 inp_month = x[2][i*batch_size:(i+1)*batch_size]
#                 inp_dayofweek = x[3][i*batch_size:(i+1)*batch_size]
#                 inp_dayofmonth = x[4][i*batch_size:(i+1)*batch_size]
#                 inp_weekendflag = x[5][i*batch_size:(i+1)*batch_size]
#                 inp = [inp_power, inp_hour, inp_month, inp_dayofweek, inp_dayofmonth, inp_weekendflag]
#                 out = y[i*batch_size:(i+1)*batch_size]
#                 yield inp, out

# x_train, y_train, x_valid,  y_valid, x_test, y_test = extract_data_hourly(df, in_dim, out_dim)

# train = []
# x_1 = np.concatenate([x_train[0], x_valid[0]])
# x_2 = np.concatenate([x_train[1], x_valid[1]])
# x_3 = np.concatenate([x_train[2], x_valid[2]])
# x_4 = np.concatenate([x_train[3], x_valid[3]])
# x_5 = np.concatenate([x_train[4], x_valid[4]])
# x_6 = np.concatenate([x_train[5], x_valid[5]])

# train = [x_1, x_2, x_3, x_4, x_5, x_6]
# lab = np.concatenate([y_train, y_valid])

# batch_size = 100
# params = {'batch_size': 100,
#           'shuffle': True}

# training_generator = NewDataGenerator(**params).generate(train, lab)
# # training_generator = NewDataGenerator(**params).generate(x_train, y_train)
# # validation_generator = NewDataGenerator(**params).generate(x_valid, y_valid)

# # earlyStopping = EarlyStopping(monitor='val_loss', patience = 5)

# model = Conv1d_test_1(in_dim, out_dim)

# print len(x_train[0]), batch_size


# # model.fit_generator(generator = training_generator,
# #                     steps_per_epoch = len(x_train[0])//batch_size,
# #                     validation_data = validation_generator,
# #                     validation_steps = len(x_valid[0])//batch_size, epochs = 100, callbacks = [earlyStopping])
# model.fit_generator(generator = training_generator,
#                     steps_per_epoch = len(train[0])//batch_size, 
#                     epochs = 100)

# test = [np.reshape(x_train[0][0], (1, x_train[0][0].shape[0], 1)), np.reshape(x_train[1][0], (1, 1)), np.reshape(x_train[2][0], (1, 1)), np.reshape(x_train[3][0], (1, 1)), np.reshape(x_train[4][0], (1, 1)), np.reshape(x_train[5][0], (1, 1))]
# print mean_squared_error(model.predict(x_test), y_test)
# ans = model.predict(x_test)
# out = 0
# for i in xrange(len(ans)):
# 	for j in xrange(60):
# 		out += (ans[i][j] - y_test[i][j])**2
# print math.sqrt(out/(len(ans) * 60))

##Squeeze Arch 

class DataGenerator(object):
    def __init__(self, batch_size = 32, shuffle = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, x, y):
        while 1:
        
            imax = int(len(x)/self.batch_size)
            for i in range(imax):
                inp = x[i*batch_size:(i+1)*batch_size]
                out = y[i*batch_size:(i+1)*batch_size]
                yield inp, out


batch_size = 100
params = {'batch_size': 100,
          'shuffle': True}

train = np.concatenate([x_train, x_valid])
lab = np.concatenate([y_train, y_valid])

training_generator = DataGenerator(**params).generate(train, lab)
# validation_generator = DataGenerator(**params).generate(x_valid, y_valid)

# earlyStopping = EarlyStopping(monitor='val_loss', patience = 5)

model = squeeze_arch(in_dim, out_dim)

# model.fit_generator(generator = training_generator,
#                     steps_per_epoch = len(x_train)//batch_size,
#                     validation_data = validation_generator,
#                     validation_steps = len(x_valid)//batch_size, epochs = 20, callbacks = [earlyStopping])
model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(train)//batch_size, 
                    epochs = 50)

ans = model.predict(x_test)
out = 0
for i in xrange(len(ans)):
	for j in xrange(60):
		out += (ans[i][j] - y_test[i][j])**2
print math.sqrt(out/(len(ans) * 60))