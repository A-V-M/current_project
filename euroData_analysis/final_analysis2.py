# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:01:22 2017

@author: andreas
"""

#iVoters = euroData[euroData['[dem] country_code']=='GB'].index.values   
#nVoters = len(iVoters)
#
#X = euroData_dummies[euroData['[dem] country_code']=='GB']
#X = np.array(X)

iGB = euroData[euroData['[dem] country_code']=='GB'].index.values
voteRef = euroData.iloc[iGB]['[question] vote_referendum']   

iLeave = voteRef[voteRef == 'For the UK to leave the EU'].index.values
#
iRemain = voteRef[voteRef == 'For the UK to stay in the EU'].index.values
#
iOther = voteRef[voteRef == 'I did not vote'].index.values

likelihoods = pd.DataFrame(columns = ['Leave','Remain','Other'],index=euroData_dummies.index)

likelihoods['Remain']=np.sum(euroData_dummies[iRemain],axis=1)/len(iRemain)
likelihoods['Leave']=np.sum(euroData_dummies[iLeave],axis=1)/len(iLeave)
likelihoods['Other']=np.sum(euroData_dummies[iOther],axis=1)/len(iOther)

pL = len(iLeave)/(len(iGB))
pR = len(iRemain)/(len(iGB))
pO = len(iOther)/(len(iGB))

uncond_probs_Brexit = np.sum(euroData_dummies[iGB],axis=1)/len(iGB)

cond_probs_leave = (pL * likelihoods['Leave']) / uncond_probs_Brexit
cond_probs_remain = (pR * likelihoods['Remain']) / uncond_probs_Brexit
cond_probs_other = (pO * likelihoods['Other']) / uncond_probs_Brexit

euroData_dummies.columns[np.argsort(cond_probs_leave)]
euroData_dummies.columns[np.argsort(cond_probs_remain)]
euroData_dummies.columns[np.argsort(cond_probs_other)]

country_names = euroData['[dem] country_code'].unique()
country_dists = pd.DataFrame(columns = ['Country','Leave','Remain','Other'])
country_dists['Country'] = country_names
country_dists = country_dists.set_index(['Country'])

dists_leave = np.zeros(len(c_labels))
dists_remain = np.zeros(len(c_labels))
dists_other = np.zeros(len(c_labels))


def BC(p,q):
    
    p = p/np.sum(p)
    
    q = q/np.sum(q)
    
    D = (p * q)**(1/2)
    
    D = -np.log(np.sum(D))
    
    return D

uncond_probs = pd.DataFrame(columns = country_names,index=euroData_dummies.index)
    
for c in range(0,len(country_names)):

     country_i=euroData_dummies.columns[euroData['[dem] country_code']==country_names[c]]     
     
     uncond_probs[country_names[c]] = np.sum(euroData_dummies[country_i],axis=1)/len(country_i)

     for l in range(1,len(c_labels)):
         
         q = uncond_probs[country_names[c]].loc[c_labels[l]]
         
         p = cond_probs_leave.loc[c_labels[l]]
         
         dists_leave[l-1] = BC(p,q)
         
         p = cond_probs_remain.loc[c_labels[l]]

         dists_remain[l-1] = BC(p,q)
         
         p = cond_probs_other.loc[c_labels[l]]
         
         dists_other[l-1] = BC(p,q)
         
         
     country_dists.loc[country_names[c],:] = [np.sum(dists_leave), np.sum(dists_remain),np.sum(dists_other)]
         

log_leave=cond_probs_leave * np.log(cond_probs_leave)

log_remain=cond_probs_remain * np.log(cond_probs_remain)

log_other=cond_probs_other * np.log(cond_probs_other)        

sum_log=log_leave+log_remain+log_other

partA=[pL,pR,pO]

partA = -np.sum(partA * np.log(partA))

partB = -(uncond_probs_Brexit * sum_log).groupby(level=0).sum()

MI = partA - partB
MIn = (MI.sort_values())/partA
        
country_ranks_uk=((country_dists['Remain'] + country_dists['Other']) / country_dists['Leave']).sort_values(ascending=True)  
            