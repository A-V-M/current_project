# -*- coding: utf-8 -*-
"""
Created on Mon May 14 02:14:34 2018

@author: andreas
"""

class prob_analysis():
    
    def extract_probabilities(voterIndices):
        
        probabilities = pd.DataFrame(columns = ['Likelihoods Leave','Likelihoods No Leave','Unconditional Leave','Conditional Leave','Conditional No Leave'],index=df_encoded.index)
        
        probabilities['Likelihoods No Leave']=np.sum(df_encoded[voterIndices['No Leave']],axis=1)/len(voterIndices['No Leave'])
        probabilities['Likelihoods Leave']=np.sum(df_encoded[voterIndices['Leave']],axis=1)/len(voterIndices['Leave'])
        
        pL = len(voterIndices['Leave'])/(len(voterIndices['UK']))
        pN = len(voterIndices['No Leave'])/(len(voterIndices['UK']))
        
        probabilities['Unconditional Leave'] = np.sum(df_encoded[voterIndices['UK']],axis=1)/len(voterIndices['UK'])
        
        probabilities['Conditional Leave'] = (pL * probabilities['Likelihoods Leave']) /  probabilities['Unconditional Leave']
        probabilities['Conditional No Leave'] = (pN * probabilities['Likelihoods No Leave']) /  probabilities['Unconditional Leave']
                              
        return probabilities
        
    def BC(p,q):
            
        p = p/np.sum(p)
        
        q = q/np.sum(q)
        
        D = (p * q)**(1/2)
        
        D = -np.log(np.sum(D))
        
        return D
       
    def compute_prevalence(probabilities):
        
        
        country_dists = pd.DataFrame(columns = ['Country','Leave','No Leave'])
        country_dists['Country'] = country_names
        country_dists = country_dists.set_index(['Country'])
        
        dists_leave = np.zeros(len(c_labels))
        dists_noleave = np.zeros(len(c_labels))
        
        uncond_probs = pd.DataFrame(columns = country_names,index=df_encoded.index)
            
        for c in range(0,len(country_names)):
        
             country_i=df_encoded.columns[df['[dem] country_code']==country_names[c]]     
             
             uncond_probs[country_names[c]] = np.sum(df_encoded[country_i],axis=1)/len(country_i)
        
             for l in range(1,len(c_labels)):
                 
                 q = uncond_probs[country_names[c]].loc[c_labels[l]]
                 
                 p = probabilities['Conditional Leave'].loc[c_labels[l]]
                 
                 dists_leave[l-1] = prob_analysis.BC(p,q)
                 
                 p = probabilities['Conditional No Leave'].loc[c_labels[l]]
        
                 dists_noleave[l-1] = prob_analysis.BC(p,q)
                 
             country_dists.loc[country_names[c],:] = [np.sum(dists_leave), np.sum(dists_noleave)]
             
             prevalence=((country_dists['No Leave']) / country_dists['Leave'])
             
        return prevalence
        
    def rand_distribution(probabilities):
        
        cond_probs_leave = probabilities['Conditional Leave']
        cond_probs_noleave = probabilities['Conditional No Leave']
               
        uncond_probs_rand = pd.DataFrame(columns = ['rand'],index=df_encoded.index)
        rand_ratio = np.zeros(1500)
        
        distsRand1 = np.zeros(len(c_labels))
        distsRand2 = np.zeros(len(c_labels))
        
        for randI in range(len(rand_ratio)):
             
             uncond_probs_rand['rand'] = np.random.random(len(df_encoded))
             uncond_probs_rand['rand'] = uncond_probs_rand['rand']/uncond_probs_rand['rand'].groupby(level=0).sum()
             
             for l in range(1,len(c_labels)):
                 
                 q = uncond_probs_rand['rand'].loc[c_labels[l]]
                 
                 p = cond_probs_leave.loc[c_labels[l]]
                 
                 distsRand1[l] = prob_analysis.BC(p,q)
                 
                 p = cond_probs_noleave.loc[c_labels[l]]
        
                 distsRand2[l] = prob_analysis.BC(p,q)
                 
                 
             rand_ratio[randI] = np.sum(distsRand2)  / np.sum(distsRand1)
        
        return rand_ratio
                