# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:37:29 2019

@author: Keng Siew Chan
Linear regression for prediction of the efficiency of Si solar cell
"""

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import pandas as pd

np.random.seed(51) 
tf.set_random_seed(51) 

# Read data stored in excel file
# Change 'Your directory' to the directory of your ML_QK.xls file
excel_file = 'Your directory/ML_QK.xls'

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

# number of data points
n = len(x)

# Plot training set for visualisation
plt.scatter(raw_x_np, y) 
plt.xlabel('SiN nm') 
plt.ylabel('Efficiency (%)') 
plt.title("Training Data") 
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

# Number of data points
n_val = len(x_val) 

#Plot validation set for visualisation
plt.scatter(raw_x_np_val, y_val) 
plt.xlabel('SiN nm') 
plt.ylabel('Efficiency (%)') 
plt.title("Cross val Data") 
plt.show() 

# Create placeholders for X and Y for tensorflow
X = tf.placeholder("float") 
Y = tf.placeholder("float")

# Parameters initilisation
W1 = tf.Variable(np.random.randn(), name = "W1") 
W2 = tf.Variable(np.random.randn(), name = "W2")
W3 = tf.Variable(np.random.randn(), name = "W3")
W4 = tf.Variable(np.random.randn(), name = "W4")
b = tf.Variable(np.random.randn(), name = "b") 

# learning_rate and number of epochs used
learning_rate = 0.0005
training_epochs = 10000

# Hypothesis 
y_pred = tf.multiply(X, W1)+ tf.multiply(X**2,W2)+ tf.multiply(X**3,W3)+ tf.multiply(X**4,W4)+b
# Loss Function (mean square error) 
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n) 

# Gradient decent with Adam optimiser
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Global variables initializer for tensorflow 
init = tf.global_variables_initializer() 

# Start Tensorflow Session 
with tf.Session() as sess: 
      
    # Initializing Variables 
    sess.run(init) 
    costs= []
    epoch_num=[]
    predictions=[]
    
    # Iterating through epochs 
    for epoch in range(training_epochs): 
          
        # Feed each data point into the optimizer
        for (_x, _y) in zip(x, y): 
            sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
        
        # Display result every 500 epochs 
        if (epoch+1) % 500 == 0: 
            # Cost calculation for every epoch 
            c = sess.run(cost, feed_dict = {X : x, Y : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W1 =", sess.run(W1), "W2 =", sess.run(W2), "b =", sess.run(b))
            costs.append(c)
            epoch_num.append(epoch)
      
    # Caching necessary values  
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
    weight1 = sess.run(W1)
    weight2 = sess.run(W2)
    weight3 = sess.run(W3)
    weight4 = sess.run(W4)
    predictions = sess.run(y_pred, feed_dict ={X: x, Y: y})
    predictions_val = sess.run(y_pred, feed_dict = {X: x_val, Y: y_val})
    bias = sess.run(b) 
        
    # Cost vs epoch num plot    
    plt.plot(epoch_num, costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate) + " with Adam optimizer")
    plt.show()
    
    # Scatter vs predictions plot for training set
    plt.scatter(raw_x_np,y)
    plt.plot(raw_x_np,predictions)
    plt.xlabel('SiN nm')
    plt.ylabel('Effiiency')
    plt.title("Training Data vs prediction") 
    plt.show()
    
    # Scatter vs predictions plot for cross validation set
    plt.scatter(raw_x_np_val,y_val)
    plt.plot(raw_x_np_val,predictions_val)
    plt.xlabel('SiN nm')
    plt.ylabel('Effiiency')
    plt.title("Cross validation data vs prediction") 
    plt.show()
    
    
    #accuracy analysis
    #I used "Mean Absolute Percent Error" (MAPE) here. <10% would be considered as pretty good result
    
    error_train = (1/n)*sum(abs(y-predictions)/abs(y))*100
    print("Mean Absolute Percent Error for training set is " + "%.2f" % error_train +"%")
 
    error_val = (1/n_val)*sum(abs(y_val-predictions_val)/abs(y_val))*100
    print("Mean Absolute Percent Error for cross validation set is  " + "%.2f" % error_val +"%")
    
    
    
    
    