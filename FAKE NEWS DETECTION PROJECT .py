#!/usr/bin/env python
# coding: utf-8

# In[3]:


#in this project, i use python  to detect fake news. I will use SK learn to build a TfidfVectorizer 
#on the dataset. I will then initializea PassiveAgressive Classifier and fit the model. The accuracy score
#and the confusion matrix will give the indication to the efficiency of the model. 

#IMPORTS
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[9]:


#Next step is to Read the data and get the size and shape of the first five entiries
df = pd.read_csv ("C:\\Users\\hp\\Desktop\\Reviving python\\news.csv")

#get shape
df.shape
df.head()


# In[13]:


#get the lables
labels = df.label
labels.head()


# In[14]:


#now we split the data set into the test and training sets
x_train,x_test,y_train,y_test = train_test_split(df['text'], labels, test_size = 0.2, random_state = 7)


# In[15]:


#next we initalize a TfidfVectorizer with stop words(common english language words that will not be included)
#and a maximum document frequency of 0.7 therefore words with a frequency greater than 0.7 will be excluded

#INITIALIZE THE Tdidf Vectorizer
tfidf_vectorizer = TfidfVectorizer (stop_words = 'english',max_df = 0.7)

#next we transform the vectorizer on the test and train data sets. This is because the vectorizer 
#converts raw documents into a matrix of TF-IDF features. 
#FIT AND TRANSFORM TRAIN SET AND TRANSFORM THE TEST SET
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# In[22]:


#we’ll initialize a PassiveAggressiveClassifier. This is. We’ll fit this on tfidf_train and y_train.
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#, we’ll predict on the test set from the TfidfVectorizer and 
#calculate the accuracy with accuracy_score() from sklearn.metrics.
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)} %')
      


# In[23]:


#this model gives an accuracy 92.74% . 
#next we print out a confussion matrix to get insight on the number of false and true negatives and postives
confusion_matrix(y_test,y_pred,labels = ['FAKE', 'REAL'])


# In[ ]:


#from this model prediction we have 591 true positives, 584 true negatives, 45 false positives and 47 false negatives

