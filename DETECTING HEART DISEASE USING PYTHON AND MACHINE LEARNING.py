#!/usr/bin/env python
# coding: utf-8

# In[23]:


#in this program, i will use python and machine learning to detect heart disease. I will be using data obtained from 70000 patients
#visiting a cardiology clinic. 

#in the data, a patient with cardiovascular dx is categorized as 1 and 0 if they do not have the disease

#import the libraries
import numpy as np
import pandas as pd
import seaborn as sns

#Next we load the data set and define the separator
df = pd.read_csv("C:\\Users\\hp\\Desktop\\Reviving python\\cardio_train.csv", sep = ';')
df.head(7)
df.shape
#Next we peruse through our data to check for any missing values as these could affect our ouput
#we can use the isnull() or isna() functions in pandas which do the same things. 
df.isnull().sum() #you could also use df.isnull.values.any()

#Next we describe our data by getting a few stats
df.describe()

#Number of patients with cardiovascular disease and visualize
df['cardio'].value_counts()
sns.countplot(df['cardio'])

#next we compare to see the difference between people with the dx and those without in terms of their age in years
df['years'] = (df['age'] / 365).round(0)  # age was given in days so we convert to years
df["years"] = pd.to_numeric(df["years"],downcast = 'integer') #we convert the years to integer values


# In[9]:


#visualize data using te seaborn library
sns.countplot(x = 'years',hue = 'cardio',data = df, palette = "colorblind", edgecolor = sns.color_palette("dark",n_colors = 1));


# In[10]:


#get the column correlations
df.corr()


# In[12]:


#visualize the correlation
import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(),annot=True, fmt = '.0%')


# In[13]:


#here we prepare our data for the machine lerning
#drop the years column cz it doesnt add value really
df = df.drop('years',axis = 1)


# In[14]:


#remove the id column since it really does not give any pertinent info
df = df.drop('id', axis = 1)


# In[15]:


#split the data into independent /feature dataset and the target dataset
X = df.iloc[: , :-1]. values # contains all rows and all columns except the last one
Y = df.iloc[:, -1]. values #contains all rows of the last column


# In[16]:


#split the data into the test and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 1)


# In[17]:


#feature scaling so the data values are between 0 and 1 inclusive
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) #we already fit so no need to redo it


# In[18]:


#Create a machine learnig model using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1)
forest.fit(X_train, Y_train)


# In[19]:


#testing the acuracy of the model on the training data
model =  forest
model.score(X_train, Y_train)


# In[22]:


#test model acuracy on the test data using a confusion matrix and use the matrix to compute the accuracy score
from sklearn.metrics import confusion_matrix
cm =  confusion_matrix(Y_test, model.predict(X_test))

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
#Print confusion matrix
print (cm)

#print the models accuracy on the data 
print('Model Test Accuracy = {}'.format( (TP + TN) / (TP+ TN + FN + FP) ) )


# In[ ]:





# In[ ]:




