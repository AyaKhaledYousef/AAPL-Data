# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:43:14 2018

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import matplotlib.pyplot as plt
import time

data=pd.read_csv("AAPL.csv")

data['Date'] = pd.to_datetime(data['Date'])  #add DateTime to my data to generate dt 

data['day_of_week'] = data['Date'].dt.weekday_name  #add the days standard to the date

#data['Open'] = data['Open'].astype(int) # Convert to int 
#data['Close'] = data['Close'].astype(int) 

data['same_day_delta'] = ((data['Close'] - data['Open']) / data['Close'] ) * 100 #percentage difference between 'open','close'

data['same_day_strategy'] = np.where(data['same_day_delta'] <= 0,'0' ,'1')


data['next_close_delta'] = 100 * (1 - data.iloc[0].Close / data.Close)
data['next_close_strategy'] = np.where(data['next_close_delta'] <= 1,'0' ,'1')


# information per month
keep_col = ['Date'] #write the Date column
new_f = data[keep_col]
new_f.to_csv("monthly_analysis.csv", index=False) #write csv file

monthly=pd.read_csv("monthly_analysis.csv")
monthly['year'] = pd.DatetimeIndex(monthly['Date']).year #extract year from Date
monthly['month'] = pd.DatetimeIndex(monthly['Date']).month # extract month from Date
                   

#plot
data['same_day_delta'].value_counts().plot() #plot the column
data['same_day_delta'].value_counts().plot.hist() #Plot the distribution of “same_day_delta”.

#the execution time
start_time = time.time()
print("--- %s data ---" % (time.time() - start_time))

#prediction model for “same_day_strategy”.
x=data.iloc[:,[1,4]].values
y=data.iloc[:,[9]].values

from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

#excution time for this model model

start_time = time.time()
print("--- %s y_pred ---" % (time.time() - start_time))

#prediction model for “next_close_strategy”

a=data.iloc[:,[4,4]].values
b=data.iloc[:,[11]].values

from sklearn.cross_validation import train_test_split

A_train,A_test,B_train,B_test=train_test_split(a,b,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(A_train,B_train)

b_pred=classifier.predict(A_test)

#excution time for this model model

start_time = time.time()
print("--- %s b_pred ---" % (time.time() - start_time))