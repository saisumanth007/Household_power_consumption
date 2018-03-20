from utils import *
from model import *
import numpy as np
from keras.callbacks import EarlyStopping

df = preprocess()
in_dim, out_dim = 60, 60
x_train, y_train, x_valid,  y_valid, x_test, y_test = extract_data(df, in_dim, out_dim)

#Basic LSTM
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

training_generator = DataGenerator(**params).generate(x_train, y_train)
validation_generator = DataGenerator(**params).generate(x_valid, y_valid)

earlyStopping = EarlyStopping(monitor='val_loss', patience = 5)

model = Basic_LSTM(in_dim, out_dim)

model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(x_train)//batch_size,
                    validation_data = validation_generator,
                    validation_steps = len(x_valid)//batch_size, epochs = 20, callbacks = [earlyStopping])

# Random Forest
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

regr = RandomForest()
regr.fit(x_train,y_train)