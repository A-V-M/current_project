# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:43:39 2017

@author: andreas
"""
import scipy as sp

def extract_question_values(label,dict_conversion,nominal):
    
    country_val = np.zeros(len(country_names))
    
    for country_index in range(0,len(country_names)):
        
        country = country_names[country_index]
    
        cats=(euroData[label].iloc[country].astype('category')).cat.categories
        
        val=int(sp.median((euroData[label].iloc[country].astype('category')).cat.codes.values))
        
        country_val[country_index] = dict_conversion[cats[val]]
        
likelihoods_country = pd.DataFrame(columns = country_names,index=euroData_dummies.index)

country_val = np.zeros(len(country_names))
    
for country_index in range(0,len(country_names)):
    
    country = country_names[country_index]

    country_indices = euroData[euroData['[dem] country_code']==country].index.values
       
    likelihoods_country[country]=np.sum(euroData_dummies[country_indices],axis=1)/len(country_indices)

mds0_corr = np.zeros(likelihoods_country.shape[0])
mds1_corr = np.zeros(likelihoods_country.shape[0])

for answer_i,answer in enumerate(likelihoods_country.index.values):
    
    vals=likelihoods_country.loc[answer].values
    
    mds0_corr[answer_i] = sp.stats.spearmanr(vals,mds_fit[:,0])[0]
    mds1_corr[answer_i] = sp.stats.spearmanr(vals,mds_fit[:,1])[0]
    
likelihoods_country['mds0'] = mds0_corr
likelihoods_country['mds1'] = mds1_corr

fig, ax = plt.subplots()
axes = plt.gca()
axes.set_xlim([-1,1.4])
axes.set_ylim([-1,1.4])

for answer_i,answer in enumerate(likelihoods_country.index.values):
    
    txt = (':'.join(answer)).split('] ')[1]
    txt_size = np.exp(((mds1_corr[answer_i]**2 + mds0_corr[answer_i]**2)*4))/3.5
    
    ax.text(mds1_corr[answer_i],mds0_corr[answer_i], txt,size=txt_size,fontweight='bold')
    
plt.tight_layout()
