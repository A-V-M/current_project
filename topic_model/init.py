# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 13:09:38 2017

@author: andreas
"""

#import all relevant modules and libraries

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score,precision_score, make_scorer,fbeta_score
from sklearn import svm, tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, ShuffleSplit,train_test_split
from wordcloud import WordCloud
from collections import Counter
from itertools import combinations
from gensim import models,corpora,similarities
from nltk.util import ngrams
import string
import matplotlib.pyplot as plt
import re
import nltk

#import main dataset for training / testing

def import_data():
    
    reviews = pd.read_csv('deceptive-opinion.csv')
    reviews_txt =  reviews['text']
    
    labels = reviews['deceptive'].values
    labels = (labels == 'truthful') * 1
    labels = pd.Series(labels)
    
    truthful_indices = labels[labels == 1].index.values
    deceptive_indices = labels[labels == 0].index.values
                              
    return reviews_txt, labels, truthful_indices, deceptive_indices


                      
    
    