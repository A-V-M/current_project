# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:24:28 2017

@author: andreas
"""

import pandas as pd
import numpy as np
import scipy as sp
from random import randint
from sklearn import linear_model, datasets


euroData = pd.read_excel('dalia_research_challenge_europulse.xlsx')

euroData_dummies = pd.DataFrame()

colIndex = pd.read_excel('wanted_fields.xlsx',header=None)
colIndex = colIndex[colIndex[1]==1]

c_labels = list(colIndex[0])

s = euroData['[dem] age']

labels = ['<18','18-35','35-51','51-78']
euroData['[dem] age'] = pd.cut(s,[0,18,35,51,78],right=False,labels=labels)
#euroData['[dem] age'] = euroData['[dem] age'].cat.codes


c_index = range(1,len(colIndex))

for c in c_index:
    
    c_label = c_labels[c]
    
    s=euroData[c_label].astype('category')
    
    s=pd.get_dummies(s)
    
    s=s.T

    DFindex = pd.MultiIndex.from_product([c_label,s.index.values], names=['question','answer'])

    s=s.set_index(DFindex)
    
    euroData_dummies = pd.concat([euroData_dummies, s], axis=0)
    
