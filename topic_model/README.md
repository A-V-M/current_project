Classifying hotel reviews using topic models
---

The following files are necessary to run through the whole pipeline

init.py -- loads necessary libraries: sklearn 0.19, nltk 3.2.4, gensim 2.3, wordcloud 1.3.1. Make sure all these libraries are pre-installed
before running.

single_word.py -- visualising and extracting properties of single word distributions within a supplied text corpus

classification.py -- contains basic functions for training and testing a classifier with 5-fold cross-validation 

topic.py -- cleans up text from unnecessary bigrams with minimal information, formulates dictionary and corpus, extracts topic distributions for each document

recipe.py -- contains the full pipeline for the project documented in report.pdf. running through the whole script should produce identical results with the reported ones.

AM/2017
