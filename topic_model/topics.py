# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:13:51 2017

@author: andreas
"""

def remove_tags(txt):
    
    num_docs = txt.shape[0]
    
    bigrams_docs = [None] * num_docs
    
    acceptable_tags = ['CC','CD','DT','IN','EX','LS','MD','POS','PDT','RP','TO','WDT','WP','WRB','PRP$','PRP']               
    
    tag_combos = [item for n,item in enumerate(combinations(acceptable_tags[::-1],2))] + [item for n,item in enumerate(combinations(acceptable_tags,2))]
    
    for i in range(0,num_docs):
        
        text=txt[i].lower().translate(None, string.punctuation)
        
        token = nltk.word_tokenize(text)
        
        bigrams = ngrams(token,2)
        
        valid_bigrams = [bigram for bigram in bigrams if tuple(dict(nltk.pos_tag(bigram)).values()) not in tag_combos]
    
        bigrams_docs[i] = ['_'.join(valid_bigrams[n]) for n,item in enumerate(valid_bigrams)]
    
    return bigrams_docs

def construct_corpus(docs):
    
        num_docs = len(docs)
    
        dictionary=corpora.Dictionary(docs)

        corpus = [dictionary.doc2bow(text) for text in docs]
        
        return corpus,dictionary

def extract_topic_dists(corpus,dictionary,num_topics,chunksize,passes):
    
    num_docs = np.shape(corpus)[0]
    
    model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics,alpha = 'auto',eta='auto',random_state=0, chunksize=chunksize, passes=passes)
    
    topic_dists = np.zeros([num_docs,num_topics])
    
    for i,item in enumerate(corpus):
        
        dists = model.get_document_topics(item)
        
        indices = dict(dists).keys()
        
        vals = dict(dists).values()
        
        topic_dists[i,indices] = vals
                   
    return topic_dists


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    