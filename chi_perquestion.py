# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:28:37 2017

@author: andreas
"""
c_labels_no_country = c_labels[1:]
#indexing from 2nd element onwards to get around the country code label which is not needed

chi = np.zeros(len(c_labels_no_country)) 
p_vals = np.zeros(len(c_labels_no_country))
#pd.DataFrame(index = c_labels)

for c in range(0,len(c_labels_no_country)):
    
    
    t1=euroData_dummies.loc[c_labels_no_country[c]][np.concatenate((iRemain,iOther))].T.sum().values
    t2=euroData_dummies.loc[c_labels_no_country[c]][iLeave].T.sum().values
    
    ctable = [t1,t2]
    chi2, p, dof, ex = sp.stats.chi2_contingency(ctable, correction=False)

    chi[c] = chi2
    p_vals[c] = p
    
chiDF = pd.DataFrame(columns=['chi','p'],data=np.array([chi,p_vals]).T,index=c_labels_no_country)

pl = plt.barh(ind, chiDF['chi'].sort_values().values, width)
ind_labels = np.arange(0.5,42,1)
labels=chiDF['chi'].sort_values().index.values
plt.yticks(ind_labels,labels)
plt.tight_layout()

for i in np.arange(41,11,-1):
    plt.annotate('*',(chiDF['chi'].sort_values().values[i],i))