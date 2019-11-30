# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:37:29 2019

@author: Keng Siew Chan
Linear regression model for the prediction of the efficiency of Si solar cell
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(51) 

# Read data stored in excel file
# Change 'Your directory' to the directory of your datafile.xls file
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

# instantiate and fit
lm2 = LinearRegression()

# Resphape the data
x = x.reshape(-1,1)
y = y.reshape(-1,1)
x_val= x_val.reshape(-1,1)
y_val = y_val.reshape(-1,1)

# Define the polynomial used
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)
x_val_poly = poly.fit_transform(x_val)

lm2.fit(x_poly, y)

# print the coefficients
print(lm2.intercept_)
print(lm2.coef_)

#output prediction for training
y_pred_train = lm2.coef_[0][1]*x + lm2.coef_[0][2]*(x**2) +lm2.coef_[0][3]*(x**3)+lm2.coef_[0][4]*(x**4)+ lm2.intercept_

#output prediction for cross validation
y_pred_val = lm2.coef_[0][1]*x_val + lm2.coef_[0][2]*(x_val**2) +lm2.coef_[0][3]*(x_val**3)+lm2.coef_[0][4]*(x_val**4)+ lm2.intercept_


#Scatter vs predictions plot for training set
plt.scatter(raw_x_np,y)
plt.plot(raw_x_np,y_pred_train)
plt.xlabel('SiN nm')
plt.ylabel('Effiiency')
plt.title("Training Data vs prediction") 
plt.show()

#Scatter vs predictions plot for cross validation set
plt.scatter(raw_x_np_val,y_val)
plt.plot(raw_x_np_val,y_pred_val)
plt.xlabel('SiN nm')
plt.ylabel('Effiiency')
plt.title("Cross validation Data vs prediction") 
plt.show()



#accuracy analysis
#I used "Mean Absolute Percent Error" (MAPE) here. <10% would be considered as pretty good result
    
error_train = (1/n)*sum(abs(y-y_pred_train)/abs(y))*100
print("Mean Absolute Percent Error for training set is " + "%.2f" % error_train +"%")
 
error_val = (1/n_val)*sum(abs(y_val-y_pred_val)/abs(y_val))*100
print("Mean Absolute Percent Error for cross validation set is  " + "%.2f" % error_val +"%")
    
    
    
    
    