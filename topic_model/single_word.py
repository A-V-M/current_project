# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 13:26:13 2017

@author: andreas
"""
#produces a wordcloud given a block of text

def gen_wordcloud(txt,inds):

    txt = ' '.join(txt[inds].values).replace("\n"," ")

    wordcloud = WordCloud().generate(txt)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    return


#given a block of text and threshold frequency, this funtion will
#output a tag for each word (e.g. "NN" for noun etc) and an associated
#frequency. the 'freq' parameter refers to the frequency threshold. word
#occuring at a value less than 'freq' will not be included

def word_type(txt,inds,freq):
    
    txt = ' '.join(txt[inds].values).replace("\n"," ")
    
    tokens = nltk.word_tokenize(txt)
    tagged_tokens = nltk.pos_tag(tokens)

    tags = Counter(dict(tagged_tokens).values()).keys()
    freqs = Counter(dict(tagged_tokens).values()).values()
    
    inds=np.where(np.array(freqs) > freq)[0]
    tags=np.array(tags)[inds]
    freqs=np.array(freqs)[inds]
        
    return tags, freqs

def common_word_remove(txt):
    
    num_docs = txt.shape[0]
    txt_removed = [None] * num_docs
       
    for i in range(0,num_docs):
        
        common_words = 'a at for of the and to in & $'.split()
    
        words = txt[i].lower().split()
        
        txt_removed[i] = ' '.join([word for word in words if word not in common_words])
            
    return pd.Series(txt_removed)

#compute information gain for each word

def word_info(txt,inds1,inds2,freq,num_bins):
          
    num_docs = txt.shape[0]
    
    bins = np.zeros([num_docs,2])
  
    p_c1 = float(len(inds1)) / num_docs
    
    p_c2 = float(len(inds2)) / num_docs
    
    p_c = np.array([[p_c1],[p_c2]])
    
    type_entropy = -np.sum(p_c * np.log(p_c))
    
    ngram_counts = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1)
    word_counts = ngram_counts.fit_transform(txt)

    num_words = len(ngram_counts.vocabulary_.values())

    word_vocab = [None] * num_words

    for n in range(0,num_words-1):
        
        index = ngram_counts.vocabulary_.values()[n]
        
        word_vocab[index] = ngram_counts.vocabulary_.keys()[n]
           
    
    subset_indices = (np.where(np.sum(word_counts.toarray(),axis=0)>freq))[0]
    
    word_vocab_subset = [word_vocab[i] for i in subset_indices]

    word_counts_subset = word_counts[:,subset_indices]
    
    num_words_subset = np.shape(word_counts_subset)[1]
    
    info_gain = np.zeros(num_words_subset)
    
    bins = range(0,num_bins)
    
    for n in range(0,num_words_subset):
            
        p_wc = np.zeros((2,len(bins)-1))
        
        b, bin_edges = np.histogram(word_counts_subset[inds1,n].toarray(),bins=bins)
        
        p_wc[0,:] = b.astype('float') / len(inds1)
        
        b, bin_edges = np.histogram(word_counts_subset[inds2,n].toarray(),bins=bins)
        
        p_wc[1,:] = b.astype('float') / len(inds2)
        
        b, bin_edges = np.histogram(word_counts_subset[:,n].toarray(),bins=bins)
        
        p_w = b.astype('float') / num_docs
        
        p_cw = np.multiply(p_wc,p_c) / p_w     
        
        conditional_entropy = -np.nansum((p_w * p_cw) * np.log(p_cw))
        
        info_gain[n] = ((type_entropy - conditional_entropy) / type_entropy) * 100
        
        names = np.array(word_vocab_subset)    
        
        info_per_word = dict(zip(names,info_gain))
    
    return info_per_word
    
    

    
    
    
    
    
    
