import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import  mean_squared_error, r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout     
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard 
from keras.models import load_model
from keras.optimizers import RMSprop
from keras import backend as K
from joblib import dump, load


def get_baseline_score(df,geo=None):
    y = df.y
    y_hat = y.shift(1)
    y_hat[0] = y[0]

    return np.sqrt(mean_squared_error(y,y_hat))

def root_mean_squared_error(y_true,y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

data =  np.load('data/train_lstm.npy')


backsteps = 24
forwardsteps = 3

X = data[:,:-forwardsteps]
Y = data[:,-forwardsteps:].reshape(data.shape[0],-1)

X.shape

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X = X.reshape(data.shape[0],backsteps,-1)

X.shape

model = Sequential()
model.add(LSTM(units=50,return_sequences=True, input_shape=(X.shape[1],X.shape[2]),dropout=0.2))
model.add(LSTM(units=30,return_sequences=True,dropout=0.1))
model.add(LSTM(units=20,return_sequences=True))
model.add(LSTM(units=10))
model.add(Dense(units=3))

model.summary()

opti = RMSprop()
model.compile(optimizer='rmsprop', loss='mean_squared_error',metrics=['mse'])



filepath="models/LSTM-1wk-weights.{epoch:02d}-{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',period=1)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

te = TensorBoard(log_dir='/res_logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0)

callbacks_list = [checkpoint,es,te]

EPOCHS = 6
hist = model.fit(X, Y, epochs=EPOCHS, batch_size=64,validation_split=0.2,callbacks=callbacks_list)

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend(loc='best')
plt.savefig('plots/train_curve-complex.png')


dump('models/lstm_scaler.pkl')

