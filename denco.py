# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:50:43 2020

@author: RITWIK NASKAR
"""
#%%
import pandas as pd
data=pd.read_csv('denco.csv')
loyal_customers=data.groupby('custname',sort=True).size()
loyal_customers.sort_values(ascending=False)[:10]

#%%

highest_revenue=data.groupby('custname')['revenue'].sum()
highest_revenue.sort_values(ascending=False)

#%%

part_number_revenue=data.groupby('partnum')['revenue'].sum()
part_number_revenue.sort_values(ascending=False)

#%%

part_number_margin=data.groupby('partnum')['margin'].sum()
part_number_margin.sort_values(ascending=False)
