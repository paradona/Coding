# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:57:51 2020

@author: paras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = pd.read_csv('NewFile1a.csv',header=None,skiprows = 2)
# f[2].plot()
f_a = np.array(f[2][:589824]).reshape(-1,36)
ff = pd.read_csv('NewFile1b.csv',header=None,skiprows = 2)
ff_b = np.array(ff[2][:589824]).reshape(-1,36)
coherent_thermal = np.stack([f_a,ff_b]).reshape(-1,36)
y_coherent_thermal = np.tile([1,0],32768).reshape(-1,2)
print (coherent_thermal.shape)
#print (y_coherent_thermal)



f_d = pd.read_csv('NewFile1d.csv',header=None,skiprows = 2)
# f[2].plot()
f_d = np.array(f_d[2][:589824]).reshape(-1,36)
ff_e = pd.read_csv('NewFile1e.csv',header=None,skiprows = 2)
ff_ee = np.array(ff_e[2][:589824]).reshape(-1,36)
thermal = np.stack([f_d,ff_ee]).reshape(-1,36)
y_thermal = np.tile([0,1],32768).reshape(-1,2)
print (thermal.shape)
#print (y_thermal)

x_train = np.stack((coherent_thermal[5000:],thermal[5000:])).reshape(-1,36)
y_train = np.stack((y_coherent_thermal[5000:],y_thermal[5000:])).reshape(-1,2)
print(x_train[0].shape)
x_valid = np.stack((coherent_thermal[:5000],thermal[:5000])).reshape(-1,36)
y_valid = np.stack((y_coherent_thermal[:5000],y_thermal[:5000])).reshape(-1,2)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Whereas if you specify the input shape, the model gets built
# continuously as you are adding layers:
def create_model (learn_rate, dropout_rate):
    
    model = Sequential()
    model.add(Dense(36, input_dim = x_train.shape[1], activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(108, activation='softmax'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(216, activation='softmax'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(36, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    
    adam = Adam(lr= learn_rate)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['precision'])
    
    return model

dropout_rate = 0.05
epochs = 15
batch_size = 100
learn_rate = 0.001    

model = create_model(learn_rate,dropout_rate)    

model_history = model.fit(x_train,y_train, batch_size = batch_size, epochs=epochs, validation_split = 0.2, verbose =1)

accuracies = model.evaluate(x_valid, y_valid, verbose=1)



