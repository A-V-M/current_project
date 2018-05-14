# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:44:04 2018

@author: andreas
"""

class feat_analysis():
    
    def MI_question(probabilities):
        
        log_leave=probabilities['Conditional Leave'] * np.log(probabilities['Conditional Leave'])

        log_noleave=probabilities['Conditional No Leave'] * np.log(probabilities['Conditional No Leave'])
        
        sum_log=log_leave+log_noleave
        
        pL = len(voterIndices['Leave'])/(len(voterIndices['UK']))
        pN = len(voterIndices['No Leave'])/(len(voterIndices['UK']))
        
        partA=[pL,pN]
        
        partA = -np.sum(partA * np.log(partA))
        
        partB = -(probabilities['Unconditional Leave'] * sum_log).groupby(level=0).sum()
        
        MI = partA - partB
        MIn = MI/partA
        MIn_ind = (partA - -(probabilities['Unconditional Leave'] * sum_log)) / partA
        
        return MIn, MIn_ind
        
    def chi_question(voterIndices):
        
        c_labels_no_country = c_labels[1:]
#indexing from 2nd element onwards to get around the country code label which is not needed

        chi = np.zeros(len(c_labels_no_country)) 
        p_vals = np.zeros(len(c_labels_no_country))
        #pd.DataFrame(index = c_labels)
        
        iLeave = voterIndices['Leave']
        iNoLeave = voterIndices['No Leave']
        
        for c in range(0,len(c_labels_no_country)):
                   
            t1=df_encoded.loc[c_labels_no_country[c]][iNoLeave].T.sum().values
            t2=df_encoded.loc[c_labels_no_country[c]][iLeave].T.sum().values
            
            ctable = [t1,t2]
            chi2, p, dof, ex = chi2_contingency(ctable, correction=False)
        
            chi[c] = chi2
            p_vals[c] = p
            
        chiDF = pd.DataFrame(columns=['chi','p'],data=np.array([chi,p_vals]).T,index=c_labels_no_country)
        
        return chiDF
        
        
