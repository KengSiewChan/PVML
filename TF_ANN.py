# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:11:59 2019

@author: Keng Siew Chan
Artficial Neural Network model trained with Tensorflow.Keras for the prediction of the efficiency of Si solar cell

"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
import pydotplus

# Read data stored in excel file. Change 'Your directory' to the directory that you stored the datafile.xls
excel_file = 'Your directory/datafile.xls'

# Training set
sheet_name = 'single_train'
raw_data = pd.read_excel(excel_file, sheet_name)

# Assign data to their respective column in the excel sheet
raw_x = raw_data["SIN nm"]
raw_y = raw_data["CE"]

# convert data type from pandas to numpy
raw_x_np = raw_x.to_numpy()
raw_y_np = raw_y.to_numpy()
y = raw_y.to_numpy()

# Normalisation for input x
x_mean = np.mean(raw_x.to_numpy())
x_std = np.std(raw_x.to_numpy())
x = (raw_x_np-x_mean)/x_std

x0 = x.reshape(len(x),1)
x2 = np.square(x0)

x = np.concatenate((x0,x2), axis=1)

# number of data points
n = len(x)

# Plot training set for visualisation
plt.scatter(raw_x_np, y) 
plt.xlabel('SIN nm') 
plt.ylabel('Efficiency (%)') 
plt.title("Simulated Si solar cell efficiency vs front SiNx thickness") 
plt.show() 

# cross validation set
sheet_name_val = 'single_val'
raw_data_val = pd.read_excel(excel_file, sheet_name_val)

# Assign data to their respective column in the excel sheet
raw_x_val = raw_data_val["SIN nm"]
raw_y_val = raw_data_val["CE"]

# convert data type from pandas to numpy
raw_x_np_val = raw_x_val.to_numpy()
y_val = raw_y_val.to_numpy()

# Normalisation for input x
x_mean_val = np.mean(raw_x_val.to_numpy())
x_std_val = np.std(raw_x_val.to_numpy())
x_val = (raw_x_np_val-x_mean_val)/x_std_val

x0_val = x_val.reshape(len(x_val),1)
x2_val = np.square(x0_val)

x_val = np.concatenate((x0_val,x2_val), axis=1)

val_data = (x_val, y_val)

# Number of data points
n_val = len(x_val) 

#Plot validation set for visualisation
plt.scatter(raw_x_np_val, y_val) 
plt.xlabel('SiN nm') 
plt.ylabel('Efficiency (%)') 
plt.title("Validation Data") 
plt.show() 

data = x
labels = y

#The Neural Networks.
#2 layers with 30 and 10 nodes on the 1st and 2nd layers respectively
model = tf.keras.Sequential()
model.add(layers.Dense(30, input_dim=2, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss = 'mse', metrics=['mae'] )
history = model.fit(data, labels, validation_data = val_data, epochs=10000, batch_size=len(y), verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['mean_absolute_error'], label='Train')
plt.plot(history.history['val_mean_absolute_error'],label='Validation')
plt.title('Model accuracy')
plt.ylabel('MAE')
plt.xlabel('Iteration')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss. Learning rate = 0.001')
plt.ylabel('Cost')
plt.xlabel('Iteration')
plt.legend()
plt.show()

# model prediction for training data
y_pred = model.predict(x)

# Plot model prediction vs training data
plt.scatter(raw_x_np, y)
plt.plot(raw_x_np, y_pred)
plt.xlabel('SiN nm')
plt.ylabel('Effiiency(%)')
plt.title("Training data vs prediction") 
plt.show()
    
# model prediction for cross validation data
y_pred_val = model.predict(x_val)

# Plot model prediction vs cross validation data
plt.scatter(raw_x_np_val, y_val)
plt.plot(raw_x_np_val, y_pred_val)
plt.xlabel('SiN nm')
plt.ylabel('Effiiency(%)')
plt.title("Cross validation data vs prediction") 
plt.show()

