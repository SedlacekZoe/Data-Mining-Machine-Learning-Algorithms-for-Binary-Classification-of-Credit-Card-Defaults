#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import sklearn
import warnings
import tensorflow
import numpy as np
from sklearn import metrics as met
from time import process_time
import os.path


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

warnings.filterwarnings('ignore')


# In[133]:


#https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
if os.path.isfile('default of credit card clients.csv'):
    df = pd.read_csv('default of credit card clients.csv')
else:
    print('Input file does not exist')
    input("Press 'Enter' to exit")
    quit()
    
if (df.shape[1] < 2 or df.shape[0] < 10):
    print('Input is not large enough--Ensure there are at least 2 columns and at least 10 rows')
    input("Press 'Enter' to exit")
    quit()
    
x = df.copy()
y = df.copy()


# In[134]:


mod = str(input("Choose classification model: \n 1. Random Forest \n 2. Decision Tree \n 3. LSTM \n"))


# In[135]:


#Split data into features and labels
#drop last col
x.drop(x.iloc[:,len(x.columns) - 1: len(x.columns)], inplace=True, axis=1)

#drop all but last col
y.drop(y.iloc[:,0:len(y.columns) - 1], inplace=True, axis=1) 


# In[136]:


# Creating a correlation matrix and displaying it using a heatmap
#fig, axis = plt.subplots(figsize=(14, 14))
#correlation_matrix = x.corr()
#ns.heatmap(correlation_matrix, annot=True, linewidths=.5, fmt='.2f', ax=axis)
#lt.show()


# In[137]:


#negative_outcomes, positive_outcomes = y.value_counts()
#total_samples = y.count()
#print('----------Checking for Data Imbalance------------')
#print('Number of Positive Outcomes: ', positive_outcomes)
#print('Percentage of Positive Outcomes: {}%'.format(round((positive_outcomes /
#total_samples) * 100, 2)))
#print('Number of Negative Outcomes : ', negative_outcomes)
#print('Percentage of Negative Outcomes: {}%'.format(round((negative_outcomes /
#total_samples) * 100, 2)))
#print('\n')


# In[138]:


kf = KFold(n_splits=10, shuffle=True, random_state=42)
stime = process_time()

if (mod == '1' or mod == 'Random Forest' or mod == 'random forest'):
    mod = '1'
    model = RandomForestClassifier(n_estimators = 30)
    print("Random Forest\n")
elif (mod == '2' or mod == 'Decision Tree' or mod == 'decision tree'):
    mod = '2'
    model = DecisionTreeClassifier()
    print("Decision Tree\n")
elif (mod == '3' or mod == 'LSTM' or mod == 'lstm'):
    mod = '3'
    print("LSTM\n")
    tensorflow.random.set_seed(42)
    model = Sequential()
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    mod = '1'
    model = RandomForestClassifier(n_estimators = 30)
    print("Defaulted to Random Forest")
    
resultDF = pd.DataFrame()

for i, (train_index, test_index) in enumerate(kf.split(x), start=1):
    print("Fold " + str(i) + "\n")
    # Splitting the data
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for dataset in [x_train, x_test, y_train,
        y_test]:
        dataset.reset_index(drop=True, inplace=True)
    
    x_train = (x_train - x_train.mean()) / x_train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()

    if mod == '3':
        #Reshape data for LSTM
        x_train = x_train.to_numpy()
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.to_numpy()
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        
        model.fit(x_train, y_train, epochs=15, verbose=0)
        y_pred = (model.predict(x_test) >= 0.5).astype("int32")
    else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

    #individual fold metrics
    cm = met.confusion_matrix(y_test, y_pred, labels=[0, 1])     
    tn, fp, fn, tp = cm.ravel()
    
    #recall/TPR
    r = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    #precision
    p = tp / (tp + fp)
    #f1 measure
    f1 = 2 * (p * r)/(p + r)
    #accuracy
    acc = (tp + tn) / (tp + fp + fn + tn)
    #error rate
    err = (fp + fn) / (tp + fp + fn + tn)
    
    row  = pd.DataFrame({'TP': [tp], 'FP': [fp], 'TN': [tn], 'FN': [fn], 'Recall': [r], 'TNR': [tnr], 'FPR': [fpr],'FNR': [fnr],
                         'Precision': [p],'F1 Measure': [f1],'Accuracy': [acc],'Error Rate': [err]})
    resultDF = pd.concat([resultDF, row], ignore_index = True)
    print(row)
#average metrics
avgResultDF = resultDF.mean(axis=0)
etime = process_time()


# In[139]:


print("Results: \n")
print(resultDF)


# In[140]:


print("Average Metrics:\n")
print(avgResultDF)


# In[141]:


print("Time Taken: " + str(etime - stime) + " seconds\n")


# In[142]:


print('\n')
input("Press 'Enter' to exit")

