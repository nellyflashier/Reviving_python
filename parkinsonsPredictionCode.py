#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('C:/Users/hp/Desktop/Reviving python/parkinsons.data') 
df


# In[3]:


#get features and  labels
#Get the features and labels from the DataFrame (dataset).
#The features are all the columns except ‘status’, and the labels are those in the ‘status’ column.
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[4]:


#Get the count of each label (0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
#147ones and 48 zeros


# In[6]:


# Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them.
#The MinMaxScaler transforms features by scaling them to a given range. 
#The fit_transform() method fits to the data and then transforms it. 
#Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels 


# In[7]:


# split the dataset into training and testing sets keeping 20% of the data for testing.
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# In[8]:


# Initialize an XGBClassifier and train the model. 
#This classifies using eXtreme Gradient Boosting- using gradient boosting algorithms for modern data science problems. 
#It falls under the category of Ensemble Learning in ML,
#where we train and predict using many models to produce one superior output.
model=XGBClassifier()
model.fit(x_train,y_train)


# In[9]:


#Finally, generate y_pred (predicted values for x_test) and calculate the accuracy for the model. Print it out.
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:


#In this Python machine learning project,
#we learned to detect the presence of Parkinson’s Disease in individuals using various factors.
#We used an XGBClassifier for this and made use of the sklearn library to prepare the dataset.
#This gives us an accuracy of 94.87%, which is great considering the number of lines of code in this python project.

