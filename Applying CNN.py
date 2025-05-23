# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 23:26:26 2020

@author: paras
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:57:51 2020

@author: paras
"""
import sklearn.preprocessing as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = pd.read_csv('NewFile2a.csv',header=None,skiprows = 2)
# f[2].plot()
f_a = np.array(f[2][:576000]).reshape(-1,360)
ff = pd.read_csv('NewFile2b.csv',header=None,skiprows = 2)
ff_b = np.array(ff[2][:576000]).reshape(-1,360)
coherent_thermal = np.stack([f_a,ff_b]).reshape(-1,360)
y_coherent_thermal = np.tile([1,0],3200).reshape(-1,2)
print (coherent_thermal.shape)
print()
#print (y_coherent_thermal)



f_d = pd.read_csv('NewFile2d.csv',header=None,skiprows = 2)
# f[2].plot()
f_d = np.array(f_d[2][:576000]).reshape(-1,360)
ff_e = pd.read_csv('NewFile2e.csv',header=None,skiprows = 2)
ff_ee = np.array(ff_e[2][:576000]).reshape(-1,360)
thermal = np.stack([f_d,ff_ee]).reshape(-1,360)
y_thermal = np.tile([0,1],3200).reshape(-1,2)
print (thermal.shape)
#print (y_thermal)
print(np.concatenate((coherent_thermal[:500],thermal[:500])).shape)
x_train =pp.scale(np.concatenate((coherent_thermal[500:],thermal[500:]))).reshape(-1,18,20,1)
y_train = np.stack((y_coherent_thermal[500:],y_thermal[500:])).reshape(-1,2)
print(y_train.shape)
x_test = pp.scale(np.concatenate((coherent_thermal[:500],thermal[:500]))).reshape(-1,18,20,1)
y_test = np.stack((y_coherent_thermal[:500],y_thermal[:500])).reshape(-1,2)
print(x_test.shape)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Whereas if you specify the input shape, the model gets built
# continuously as you are adding layers:
model = keras.Sequential(
    [ 
        keras.Input(shape=(18,20,1)),
        layers.Conv2D(32,3,padding='valid',activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu'),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(2)      
    ]    
)
#print(model.summary())

model.compile(
       loss = keras.losses.CategoricalCrossentropy(from_logits=True),
optimizer=keras.optimizers.Adam(lr=0.001),
metrics=["categorical_accuracy"],
)

model.fit(x_train,y_train,batch_size=10,epochs=10,verbose=2)
model.evaluate(x_test,y_test, batch_size=64,verbose=2)

import matplotlib.pyplot as plt
plt.plot(model_history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('val_accuracy')
plt.xlabel('epoch')
#plt.legend([train], loc='upper left')
plt.plot(model_history.history['accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


