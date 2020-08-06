# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:11:34 2020

@author: RITWIK NASKAR
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_wine
from sklearn import tree
from graphviz import Source

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
df.shape
data = df[df.target !=2]
data
data2= data[['ash','flavanoids','hue','target']]
data2.head()
data2.describe()
x = data[['ash','flavanoids','hue']]
x
y=data['target']
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 
# 70% training and 30% test : each for train and test (X & y)
x_train.head()

clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_train

y_pred = clf.predict(x_test)
y_pred

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

y_test.shape, y_pred.shape
y_test.head()
y_pred[0:6]

tree.plot_tree(decision_tree=clf)
tree.plot_tree(decision_tree=clf, feature_names=['ash', 'flavanoids', ' hue'], class_names=['0','1'], fontsize=12)
#not a good way to draw graphs.. other methods to be experimented
tree.plot_tree(decision_tree=clf, max_depth=2, feature_names=['ash', 'flavanoids', ' hue'], class_names=['0','1'], fontsize=12)

Source(tree.export_graphviz(clf))
Source(tree.export_graphviz(clf, max_depth=3))
dot_data1 = tree.export_graphviz(clf, max_depth=3, out_file=None, filled=True, rounded=True,  special_characters=True, feature_names=['ash', 'flavanoids', ' hue'], class_names=['0','1'])  
