# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:01:57 2020

@author: Adam Pazdor
"""

'''
Advice From

Mantovani, R. G., Horváth, T., Cerri, R., Junior, S. B., Vanschoren, J., & 
de Carvalho, A. C. P. D. L. F. (2018). 
An empirical study on hyperparameter tuning of decision trees. 
arXiv preprint arXiv:1812.02207.
'''

from twitterScrape import readHuge

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
import os
from pprint import pprint, pformat

path = "T:\\COMP7890\\Full Datasets\\Trimmed"
dropcols = ["Unnamed: 0"]

def main(file,one,two,threes,params):
    filepath = os.path.join(path,file)
    df = readHuge(filepath).set_index("id").drop(dropcols,axis=1)
    
    outputs = []
    for idx, cv in enumerate(threes):
        #print(cv.get_params())
    
        pipe = Pipeline([
            ("vect",one),
            ("tfidf",two),
            ("clf",cv)
            ],
            verbose=True)
        
        parameters = {**params[0],**params[1],**params[idx+2]}
        
        xx = df.loc[:,"clean_text"]
        yy = df.loc[:,"userid"]
        xx, yy = shuffle(xx, yy)
        
        outputs.append(gridSearch(xx,yy,pipe,parameters,False))
        
    with open("gridSearchResults.txt","w") as outFile:
        for oo in outputs:
            outFile.write(oo)
            outFile.write("\n------------------------------------\n")
    

def gridSearch(xx,yy,pipe,params,random):
    output = ""
    print("Performing grid search...")
    print("parameters:")
    pprint(params)
    
    if random:
        print("Randomized Grid Search")
        gridSearch = RandomizedSearchCV(pipe,params,n_iter=500,cv=5,verbose=2)
    else:
        gridSearch = GridSearchCV(pipe,params,cv=5,verbose=2)
    gridSearch.fit(xx, yy)
    
    best ="Best score: {:0.3f}".format(gridSearch.best_score_) 
    bestSet = "Best parameters set:"
    
    output += pformat(params) + "\n\n"
    output += "{}\n{}\n".format(best,bestSet)

    print(best)
    print(bestSet)
    best_parameters = gridSearch.best_estimator_.get_params()
    for pName in sorted(params.keys()):
        myBest = "\t{:s}: {}".format(pName, best_parameters[pName])
        print(myBest)
        output += myBest+"\n"
    return output
        
vv = CountVectorizer(strip_accents="unicode")
tt = TfidfTransformer()
dt = DecisionTreeClassifier()
sv = LinearSVC()
        
cvParams = {'vect__max_df': (1.0,),
            'vect__max_features': (None, 10000),
            'vect__ngram_range': ((1, 1), (1, 2))}  # unigrams or bigrams

tfParams = {'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2')}

dtParams = {'clf__ccp_alpha': [0.0], 
            'clf__class_weight': ['balanced'], 
            'clf__criterion': ['gini'], 
            'clf__max_depth': [100,None], 
            'clf__max_features': [None,'log2','sqrt'], 
            'clf__min_impurity_decrease': [0.0], 
            'clf__min_samples_leaf': [1,2], 
            'clf__min_samples_split': [8,16], 
            'clf__min_weight_fraction_leaf': [0.0], 
            'clf__splitter': ['best']}

svParams = {'clf__C': [1.0], 
            'clf__class_weight': [None,'balanced'], 
            'clf__dual': [True,False], 
            'clf__loss': ['squared_hinge'], 
            'clf__max_iter': [2500,5000], 
            'clf__multi_class': ['ovr'], 
            'clf__penalty': ['l2'], 
            'clf__tol': [0.0001,0.001], 
            'clf__verbose': [2]}

'''
Randomized Results
DT
Best score: 0.241
Best parameters set:
	clf__ccp_alpha: 0.0
	clf__class_weight: balanced
	clf__criterion: gini
	clf__max_depth: None
	clf__max_features: None
	clf__min_impurity_decrease: 0.0
	clf__min_samples_leaf: 1
	clf__min_samples_split: 16
	clf__min_weight_fraction_leaf: 0.0
	clf__splitter: best
	tfidf__norm: l1
	tfidf__use_idf: True
	vect__max_df: 1.0
	vect__max_features: 10000
	vect__ngram_range: (1, 1)

SV
Best score: 0.399
Best parameters set:
	clf__C: 1.0
	clf__class_weight: None
	clf__dual: False
	clf__loss: squared_hinge
	clf__max_iter: 5000
	clf__multi_class: ovr
	clf__penalty: l2
	clf__tol: 0.001
	clf__verbose: 2
	tfidf__norm: l2
	tfidf__use_idf: False
	vect__max_df: 1.0
	vect__max_features: None
	vect__ngram_range: (1, 2)
    

EXHAUSTIVE RESULTS
DT
Best score: 0.260
Best parameters set:
	clf__ccp_alpha: 0.0
	clf__class_weight: balanced
	clf__criterion: gini
	clf__max_depth: None
	clf__max_features: None
	clf__min_impurity_decrease: 0.0
	clf__min_samples_leaf: 1
	clf__min_samples_split: 8
	clf__min_weight_fraction_leaf: 0.0
	clf__splitter: best
	tfidf__norm: l1
	tfidf__use_idf: False
	vect__max_df: 1.0
	vect__max_features: None
	vect__ngram_range: (1, 1)
    
SV
Best score: 0.405
Best parameters set:
	clf__C: 1.0
	clf__class_weight: None
	clf__dual: True
	clf__loss: squared_hinge
	clf__max_iter: 2500
	clf__multi_class: ovr
	clf__penalty: l2
	clf__tol: 0.0001
	clf__verbose: 2
	tfidf__norm: l2
	tfidf__use_idf: False
	vect__max_df: 1.0
	vect__max_features: None
	vect__ngram_range: (1, 2)
'''


main("TweetTiny.csv",vv,tt,[dt,sv],[cvParams,tfParams,dtParams,svParams])    