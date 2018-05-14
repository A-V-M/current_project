# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:16:31 2018

@author: andreas
"""
import pandas as pd
import numpy as np
import scipy as sp
from sklearn import manifold
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import chi2_contingency
from random import randint

for modules in ['_DataPreparation','_ProbAnalysis','_FeatAnalysis','_CountryComparison']:
    
    with open(modules + '.py') as source_file:
    
        exec(source_file.read())  

global df

print('Loading file...')

df = pd.read_excel('dalia_research_challenge_europulse.xlsx')

colIndex = pd.read_excel('wanted_fields.xlsx',header=None)
colIndex = colIndex[colIndex[1]==1]

c_labels = list(colIndex[0])

s = df['[dem] age']

labels = ['<18','18-35','35-51','51-78']
df['[dem] age'] = pd.cut(s,[0,18,35,51,78],right=False,labels=labels)

cutoff = 1.15

print('Setting up data...')
data_prep.setupData(colIndex)

voterIndices = data_prep.getIndices()

probabilities = prob_analysis.extract_probabilities(voterIndices)

print('Extracting prevalence values...')
prevalence = prob_analysis.compute_prevalence(probabilities)

print('Forming null distribution...')
rand_ratio = prob_analysis.rand_distribution(probabilities)

print('Compute feature information and chi-statistic...')
MIn, MIn_ind = feat_analysis.MI_question(probabilities)

chi = feat_analysis.chi_question(voterIndices)

print('Computing country distance matrix and MDS plot...')
dists_countries = comp_countries.BC_dist()

mds_fit, likelihoods_country = comp_countries.MDS_plot(dists_countries)

#prevalence scores bar plot

country_ranks_uk=prevalence.sort_values(ascending=True)  
z_scores = (country_ranks_uk.values - np.mean(rand_ratio))/np.std(rand_ratio)
ind = np.arange(28)
width=0.75
pl = plt.barh(ind, z_scores, width)
ind_labels = np.arange(0.5,len(country_ranks_uk),1)
labels=country_ranks_uk.index.values
plt.yticks(ind_labels,labels)
plt.tight_layout()
axes = plt.gca()
axes.set_xlim([0,3.65])
plt.grid()

thresh05 = sp.stats.norm(np.mean(rand_ratio), np.std(rand_ratio)).ppf(0.95)
thresh01 = sp.stats.norm(np.mean(rand_ratio), np.std(rand_ratio)).ppf(0.99)

plt.plot([thresh05, thresh05], [0, 28], color='y', linestyle='-.', linewidth=2)
plt.plot([thresh01, thresh01], [0, 28], color='r', linestyle='-.', linewidth=2)

#plot countries

prev_plot = prevalence.values.astype('float')
prev_plot = (np.exp(prev_plot * 10)) / 1000

fig, ax = plt.subplots()

for i, txt in enumerate(country_names):
    ax.annotate(txt, (mds_fit[i,1],mds_fit[i,0]))

#plt.scatter(mds_fit[:, 1], mds_fit[:, 0], color='navy', s=prev2, lw=0)            

Z = hierarchy.linkage(dists_countries,method='average')
T = hierarchy.fcluster(Z, 1.15)

plt.scatter(mds_fit[:, 1], mds_fit[:, 0], c=T, s=prev_plot, lw=0)
plt.axis('off')



#plot MDS correlations with question answers
 
mds1_corr = likelihoods_country['mds1']
mds0_corr = likelihoods_country['mds0'] 


fig, ax = plt.subplots()
axes = plt.gca()
axes.set_xlim([-0.65,1.3])
axes.set_ylim([-1,1])
axes.axis('off')

for answer_i,answer in enumerate(likelihoods_country.index.values):
    
    txt = (':'.join(answer)).split('] ')[1]
    txt_size = np.exp(((mds1_corr[answer_i]**2 + mds0_corr[answer_i]**2)*4))/2.5
    
    ax.text(mds1_corr[answer_i],mds0_corr[answer_i], txt,size=txt_size,fontweight='bold')
    
plt.tight_layout()
       
#
