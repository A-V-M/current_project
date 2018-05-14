# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:56:52 2018

@author: andreas
"""

class comp_countries():
    
    def BC_dist():
        
        dists_questions = np.zeros(len(c_labels))
        dists_countries = np.zeros([len(country_names),len(country_names)])
        
        for m in range(0,len(country_names)):
        
             country_a=df_encoded.columns[df['[dem] country_code']==country_names[m]]     
             
             uncond_probs_a = np.sum(df_encoded[country_a],axis=1)/len(country_a)
        
             for n in range(0,len(country_names)):
                 
                 for l in range(1,len(c_labels)):
                     
                     country_b=df_encoded.columns[df['[dem] country_code']==country_names[n]]     
             
                     uncond_probs_b = np.sum(df_encoded[country_b],axis=1)/len(country_b)
                 
                     q = uncond_probs_a.loc[c_labels[l]]
                     
                     p = uncond_probs_b.loc[c_labels[l]]
                     
                     dists_questions[l-1] = prob_analysis.BC(p,q)
                     
                 dists_countries[m,n] = np.sum(dists_questions)
                 
        return dists_countries
        
    def MDS_plot(dists_countries):
        
        mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
        mds_fit = mds.fit_transform(dists_countries)
        
        likelihoods_country = pd.DataFrame(columns = country_names,index=df_encoded.index)

        country_val = np.zeros(len(country_names))
            
        for country_index in range(0,len(country_names)):
            
            country = country_names[country_index]
        
            country_indices = df[df['[dem] country_code']==country].index.values
               
            likelihoods_country[country]=np.sum(df_encoded[country_indices],axis=1)/len(country_indices)
        
        mds0_corr = np.zeros(likelihoods_country.shape[0])
        mds1_corr = np.zeros(likelihoods_country.shape[0])
        
        for answer_i,answer in enumerate(likelihoods_country.index.values):
            
            vals=likelihoods_country.loc[answer].values
            
            mds0_corr[answer_i] = sp.stats.spearmanr(vals,mds_fit[:,0])[0]
            mds1_corr[answer_i] = sp.stats.spearmanr(vals,mds_fit[:,1])[0]
            
        likelihoods_country['mds0'] = mds0_corr
        likelihoods_country['mds1'] = mds1_corr
        
        return mds_fit, likelihoods_country
        

        
        