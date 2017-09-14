# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:28:39 2017

@author: andreas
"""


"""
let's import the reviews first

"""

reviews_txt, labels, truthful_indices, deceptive_indices = import_data()


"""
generate some initial graphics to get a feel of word distribution
"""
gen_wordcloud(reviews_txt,truthful_indices)

gen_wordcloud(reviews_txt,deceptive_indices)


"""
are there specific types of words which prevail in truthful vs.
deceptive reviews?
"""

tags, freqs = word_type(reviews_txt,truthful_indices,50)
plt.pie(freqs, labels=tags)


tags, freqs = word_type(reviews_txt,deceptive_indices,50)
plt.pie(freqs, labels=tags)


"""
does the occurrence of specific words carry information
regarding whether a review is deceptive or not?
"""

reviews_txt_SansTags = common_word_remove(reviews_txt)

info_per_word = word_info(reviews_txt_SansTags,truthful_indices,deceptive_indices,50,50)

ranked_words = np.array(info_per_word.keys())[np.argsort(info_per_word.values())]
ranked_vals = np.array(info_per_word.values())[np.argsort(info_per_word.values())]

ranked_words_top10 = ranked_words[len(info_per_word)-10:len(info_per_word)]
ranked_vals_top10 = ranked_vals[len(info_per_word)-10:len(info_per_word)]

fig, ax = plt.subplots()
  
ax.barh(np.arange(10),ranked_vals_top10)
plt.rcdefaults()

ax.set_yticks(np.arange(10))
ax.set_yticklabels(ranked_words_top10)
plt.show()


"""
I'm using support vector machines as the benchmark model as used by Ott et al., 2013
with 5-fold cross-validation 
"""

ngram_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1)
tf = ngram_vectorizer.fit_transform(reviews_txt)

X_train, X_test, y_train, y_test = train_test_split(tf, labels, test_size = 0.2, random_state = 0)
scorer = make_scorer(fbeta_score,beta=0.5)

parameters_SVM = {"kernel": ['rbf','linear'], "C": [0.1,0.4,1,10,100], "gamma": np.logspace(-3,3,5)}

clf_SVM = svm.SVC(random_state=0)

best_clf_SVM,grid_fit_SVM = train_classifier(clf_SVM,parameters_SVM,scorer,X_train,y_train)

performance_SVM = test_classifier(best_clf_SVM,X_test,y_test)

"""
rank weight coefficients
"""

weights = (best_clf_SVM.coef_[0].toarray()**2)[0]

bigram_top10 = [None] * 10

for i,item in enumerate(range(10,0,-1)):
        
    n = np.where(ngram_vectorizer.vocabulary_.values() == np.argsort(weights)[len(weights)-item])[0]

    bigram_top10[i] = ngram_vectorizer.vocabulary_.keys()[n[0]]
    
names = bigram_top10    

vals = np.sort(weights)[len(weights)-10:len(weights)]
fig, ax = plt.subplots()
  
ax.barh(np.arange(10),vals)
plt.rcdefaults()

ax.set_yticks(np.arange(10))
ax.set_yticklabels(names)
plt.show()  

"""
reduce documents to distribution of topics
"""

#first remove tag combinations which convey no information

bigram_docs = remove_tags(reviews_txt)

corpus,dictionary = construct_corpus(bigram_docs)

num_topics = range(25,275,25)+[500,1000]
chunksize = [160,320,400,800]

passes = range(1,16)
fbetas = dict()

#run through a few parameter combinations to pick the most suitable
#topic model setup

clf_gridsearch = svm.SVC(kernel = 'linear',random_state=0)

for nTopics in num_topics:

    for nChunks in chunksize:
        
        for nPasses in passes:
        
            topic_dists = extract_topic_dists(corpus,dictionary,nTopics,nChunks,nPasses)
            
            X_train, X_test, y_train, y_test = train_test_split(topic_dists, labels, test_size = 0.2, random_state = 0)
            clf_fit = clf_gridsearch.fit(X_train,y_train)
            preds = clf_fit.predict(X_test)
            fbeta = fbeta_score(preds,y_test,beta=0.5)
            fbetas[tuple([nTopics,nChunks,nPasses])] = fbeta
                    
            print([nTopics,nChunks,nPasses,fbeta])

optTopics, optChunksize,optPasses = fbetas.keys()[np.argmax(fbetas.values())]

"""
optimise the classifier with the best performing topic model
"""

topic_dists = extract_topic_dists(corpus,dictionary,optTopics,optChunksize,optPasses)
     
X_train, X_test, y_train, y_test = train_test_split(topic_dists, labels, test_size = 0.2, random_state = 0)
scorer = make_scorer(fbeta_score,beta=0.5)

parameters_TM = {"kernel": ['rbf','linear'], "C": [0.1,0.2,0.4,0.6,0.8,1,10], "gamma": np.logspace(-1,1,9)}
#parameters_TM = {"n_estimators": [10,50,100,250,500], "max_depth": [2,4,8,16,32,64]}

best_clf_TM,grid_fit_TM = train_classifier(clf_gridsearch,parameters_TM,scorer,X_train,y_train)

performance_TM = test_classifier(best_clf_TM,X_test,y_test)  
        