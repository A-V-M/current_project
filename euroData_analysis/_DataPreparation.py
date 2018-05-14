# -*- coding: utf-8 -*-
"""
Created on Mon May 14 01:24:06 2018

@author: andreas
"""


class data_prep():
    
    def setupData(colIndex):
        
        global df_encoded
        global c_labels
        global country_names
        
        country_names = df['[dem] country_code'].unique()

        c_labels = list(colIndex[0])

        c_index = range(1,len(colIndex))
        
        #one-hot encode the dataset to binary features
        
        df_encoded = pd.DataFrame()
       
        for c in c_index:
            
            c_label = c_labels[c]
            
            s=df[c_label].astype('category')
            
            s=pd.get_dummies(s)
            
            s=s.T
        
            DFindex = pd.MultiIndex.from_product([c_label,s.index.values], names=['question','answer'])
        
            s=s.set_index(DFindex)
            
            df_encoded = pd.concat([df_encoded, s], axis=0)
            

                        
        #setup indices
    def getIndices():
                
        iGB = df[df['[dem] country_code']=='GB'].index.values
        voteRef = df.iloc[iGB]['[question] vote_referendum']   
        
        iLeave = voteRef[voteRef == 'For the UK to leave the EU'].index.values
        #
        iRemain = voteRef[voteRef == 'For the UK to stay in the EU'].index.values
        #
        iOther = voteRef[voteRef == 'I did not vote'].index.values
        
        iNoLeave = np.concatenate([iRemain,iOther])
        
        voterIndices = dict({'UK': iGB,
                             'Vote':voteRef,
                             'Leave':iLeave,
                             'Remain':iRemain,
                             'Other':iOther,
                             'No Leave':iNoLeave
                             })
            
        return voterIndices
    
        
        
            
            