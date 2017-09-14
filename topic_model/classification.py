# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:07:05 2017

@author: andreas
"""

def train_classifier(clf,parameters,scorer,X,y):
        
    cv_sets = KFold(n_splits = 5, shuffle = True, random_state = 0)
    
    print "Parameter searching..."
    grid_obj = GridSearchCV(clf, parameters,cv=cv_sets,scoring=scorer)
    
    grid_fit = grid_obj.fit(X,y)
        
    best_clf = grid_fit.best_estimator_
      
    return best_clf,grid_fit
    

def test_classifier(best_clf,X_test,y_test):
 
    best_predictions = best_clf.predict(X_test)
    
    performance = {'accuracy': accuracy_score(best_predictions,y_test),
                   'recall': recall_score(best_predictions,y_test),
                   'precision': precision_score(best_predictions,y_test)
                   }
    
    return performance
