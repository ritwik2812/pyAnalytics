# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:29:33 2020

@author: RITWIK NASKAR
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model 
from sklearn import metrics
import statsmodels.api as sm  
import matplotlib.pyplot as plt
import seaborn as sns

url='https://raw.githubusercontent.com/DUanalytics/datasets/master/R/marketing.csv'
marketing=pd.read_csv(url)
marketing.head()
x=marketing[['youtube','facebook','newspaper']]
print(x)
y=marketing['sales']
print(y)

%matplotlib inline
plt.scatter(marketing['youtube'], marketing['sales'], color='red')
plt.title('Sales Vs Interest Rate', fontsize=14)
plt.xlabel('Youtube', fontsize=14)
plt.ylabel('Sales', fontsize=14)
plt.grid(True)
plt.show();

plt.scatter(marketing['facebook'], marketing['sales'], color='red')
plt.title('Sales Vs Facebook', fontsize=14)
plt.xlabel('Facebook', fontsize=14)
plt.ylabel('Sales', fontsize=14)
plt.grid(True)
plt.show();

plt.scatter(marketing['newspaper'], marketing['sales'], color='red')
plt.title('Sales Vs Newspaper', fontsize=14)
plt.xlabel('Newspaper', fontsize=14)
plt.ylabel('Sales', fontsize=14)
plt.grid(True)
plt.show();

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
x_train, x_test

model= LinearRegression()
model.fit(x,y)
model= LinearRegression().fit(x,y)

y_pred=model.predict(x_test)

y_test
y_pred

print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

x = sm.add_constant(x)
x
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)

y=3.5267 + 0.0458*(youtube) + 0.1885*(facebook) - 0.0010*(newspaper)

newdata = pd.DataFrame({'youtube':[50,60,70], 'facebook':[20, 30, 40], 'newspaper':[70,75,80]})
newdata
y_pred2= model.predict(newdata)
y_pred2
model.intercept_
model.coef_
