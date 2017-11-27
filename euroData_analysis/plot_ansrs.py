# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:54:33 2017

@author: andreas
"""
import matplotlib.pyplot as plt

def plot_answers(label):
    
    v1=euroData_dummies.loc[label][np.concatenate((iRemain,iOther))].T.sum().values / len(np.concatenate((iRemain,iOther)))
    v2=euroData_dummies.loc[label][iLeave].T.sum().values / len(iLeave)
    cats=(euroData[label].astype('category')).cat.categories
    #v_sum = v1 + v2    
    
    #v1 = v1 / (v_sum)
    #v2 = v2 / (v_sum)

    v_indxs = np.argsort(MIn_ind.loc[label][cats].values)    
    
    ind = np.arange(len(v1))
    width = 0.75
    
    p1 = plt.barh(ind, v1[v_indxs], width, color='r')
    p2 = plt.barh(ind, v2[v_indxs], width,left=v1[v_indxs])
    
    labels=(euroData[label].astype('category')).cat.categories[v_indxs]

    ind_labels = np.arange(0.5,len(v1),1)
    plt.title(label)
    plt.yticks(ind_labels,labels)
    plt.tight_layout()
