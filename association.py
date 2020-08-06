# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:02:35 2020

@author: RITWIK NASKAR
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#pip install apyori
from apyori import apriori


dataset = [['Apple', 'Beer', 'Rice', 'Chicken'],  ['Apple', 'Beer', 'Rice'], ['Apple', 'Beer'],  ['Apple', 'Bananas'], ['Milk', 'Beer', 'Rice', 'Chicken'], ['Milk', 'Beer', 'Rice'],  ['Milk', 'Beer'], ['Apple', 'Bananas']]

dataset

association_rules = apriori(dataset, min_support=0.03, min_confidence=0.2, min_lift=2, min_length=2)
association_results = list(association_rules)

print(association_results)

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")