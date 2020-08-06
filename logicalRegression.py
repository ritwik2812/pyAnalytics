# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:37:35 2020

@author: RITWIK NASKAR
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_wine

wine = load_wine()
wine
wine.data

x=wine.data
x

y=wine.target
y

labels=wine.feature_names
labels

df=pd.DataFrame(wine.data)
df.columns = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium','total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280_OD315', 'proline']
df
df['target'] = pd.Series(wine.target) #add class column as target
df
df.target.value_counts()

x = df[['ash','flavanoids','hue']]
x
y=df['target']
y

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
x_train, x_test

logistic_regression= LogisticRegression()
logistic_regression.fit(x_train,y_train)

y_pred=logistic_regression.predict(x_test)

#print the Accuracy and plot the Confusion Matrix:
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
y_test
y_pred
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='coolwarm', annot_kws={'size':20}, cbar=False)
plt.show();

newdata = pd.DataFrame({'ash':[12,13], 'flavanoids':[72,100], 'hue':[300,1000]})
newdata
y_pred2 = logistic_regression.predict(newdata)
y_pred2
