# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:54:50 2020

@author: RITWIK NASKAR
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pydataset import data
import statsmodels.api as sm
%matplotlib inline

dataset_mtcars = sm.datasets.get_rdataset(dataname='mtcars', package='datasets')

dataset_mtcars.data.head()

mtcars = dataset_mtcars.data
data = mtcars
data
data.head()
data.columns

x=data['wt']
y=data['mpg']
x
y


scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)
scaled_features[:5]  #values between -3 to +3

kmeans = KMeans( init = 'random', n_clusters=2 , n_init=3, max_iter=300, random_state=42)
kmeans
kmeans.fit(scaled_features)
kmeans.inertia_
kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in 6 times, clusters stabilised
kmeans.labels_[:5]
kmeans.cluster_centers_.shape
kmeans.cluster_centers_[0:1]


plt.scatter(x, y, c='red')
print(kmeans.labels_)
print(kmeans.cluster_centers_)

plt.scatter(x,y, c=kmeans.labels_, cmap='rainbow')
