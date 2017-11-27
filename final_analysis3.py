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

iNoLeave = np.concatenate([iRemain,iOther])

likelihoods = pd.DataFrame(columns = ['Leave','No Leave'],index=euroData_dummies.index)

likelihoods['No Leave']=np.sum(euroData_dummies[iNoLeave],axis=1)/len(iNoLeave)
likelihoods['Leave']=np.sum(euroData_dummies[iLeave],axis=1)/len(iLeave)

pL = len(iLeave)/(len(iGB))
pN = len(iNoLeave)/(len(iGB))

uncond_probs_Brexit = np.sum(euroData_dummies[iGB],axis=1)/len(iGB)

cond_probs_leave = (pL * likelihoods['Leave']) / uncond_probs_Brexit
cond_probs_noleave = (pN * likelihoods['No Leave']) / uncond_probs_Brexit

euroData_dummies.columns[np.argsort(cond_probs_leave)]
euroData_dummies.columns[np.argsort(cond_probs_noleave)]

country_names = euroData['[dem] country_code'].unique()
country_dists = pd.DataFrame(columns = ['Country','Leave','No Leave'])
country_dists['Country'] = country_names
country_dists = country_dists.set_index(['Country'])

dists_leave = np.zeros(len(c_labels))
dists_noleave = np.zeros(len(c_labels))


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
         
         p = cond_probs_noleave.loc[c_labels[l]]

         dists_noleave[l-1] = BC(p,q)
         
     country_dists.loc[country_names[c],:] = [np.sum(dists_leave), np.sum(dists_noleave)]
         

log_leave=cond_probs_leave * np.log(cond_probs_leave)

log_noleave=cond_probs_noleave * np.log(cond_probs_noleave)

sum_log=log_leave+log_noleave

partA=[pL,pN]

partA = -np.sum(partA * np.log(partA))

partB = -(uncond_probs_Brexit * sum_log).groupby(level=0).sum()

MI = partA - partB
MIn = (MI.sort_values())/partA
MIn_ind = (partA - -(uncond_probs_Brexit * sum_log)) / partA
       
country_ranks_uk=((country_dists['No Leave']) / country_dists['Leave']).sort_values(ascending=True)  
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
         