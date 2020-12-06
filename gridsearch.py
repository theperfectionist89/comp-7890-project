# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:01:57 2020

@author: User
"""
from twitterScrape import readHuge

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
import os
from pprint import pprint

path = "C:\\Users\\User\\Desktop\\Large Datasets\\twitter-events-2012-2016\\Full Datasets\\Trimmed"
dropcols = ["Unnamed: 0"]

def main(file):
    filepath = os.path.join(path,file)
    df = readHuge(filepath).set_index("id").drop(dropcols,axis=1)
    
    vv = CountVectorizer(strip_accents="unicode",
                         max_df=0.5,ngram_range=(1,2))
    tt = TfidfTransformer(use_idf=False)
    #mm = MultinomialNB(alpha=1e-05)
    rf = RandomForestClassifier(verbose=1,n_jobs=-1)
    #gb = GradientBoostingClassifier(verbose=1)
    #sv = SVC(verbose=1)
    
    print(rf.get_params())
    #print(gb.get_params())
    #print(hg.get_params())
    #print(sv.get_params())
    
    pipe = Pipeline([
        ("vect",vv),
        ("tfidf",tt),
        ("clf",rf)
        ],
        verbose=True)
    
    ''' These were the initial big random search
    param = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
         'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__n_estimators':[ii*100 for ii in range(1,10)],
        'clf__max_depth':(None,10,20,30,40,50,60,70,80,90,100),
        'clf__max_features':('auto','log2','sqrt'),
        'clf__bootstrap':(True,False),
        'clf__warm_start':(True,False),
        'clf__min_samples_leaf':(1,2,4),
        'clf__min_samples_split':(2,5,10),
        'clf__max_samples':(None,0.5)
    }
    Best parameters set:
	clf__bootstrap: False
	clf__max_depth: None
	clf__max_features: sqrt
	clf__max_samples: 0.5
	clf__min_samples_leaf: 1
	clf__min_samples_split: 2
	clf__n_estimators: 400
	clf__warm_start: False
	tfidf__use_idf: False
	vect__max_df: 0.5
	vect__ngram_range: (1, 1)
    '''
    
    ''' This is the tighter exhaustive search'''
    parameters = {
        'vect__max_df': (0.5, 0.75),
        'vect__ngram_range': ((1, 1),),  # unigrams or bigrams
         'tfidf__use_idf': (False,),
        'clf__n_estimators':[300,400,500],
        'clf__max_depth':(None,),
        'clf__max_features':('sqrt',),
        'clf__bootstrap':(False,),
        'clf__warm_start':(False,),
        'clf__min_samples_leaf':(1,2),
        'clf__min_samples_split':(2,3),
        'clf__max_samples':(None,0.5)
    }
    '''
    Best score: 0.397
    Best parameters set:
	clf__bootstrap: False
	clf__max_depth: None
	clf__max_features: sqrt
	clf__max_samples: 0.5
	clf__min_samples_leaf: 1
	clf__min_samples_split: 3
	clf__n_estimators: 500
	clf__warm_start: False
	tfidf__use_idf: False
	vect__max_df: 0.75
	vect__ngram_range: (1, 1)
    '''
    
    df = df.drop("display_name",axis=1)
    
    xx = df.loc[:,"clean_text"]
    yy = df.loc[:,"userid"]
    xx, yy = shuffle(xx, yy)
    
    #gridSearch(xx,yy,pipe,param,True)
    gridSearch(xx,yy,pipe,parameters,False)
    

def gridSearch(xx,yy,pipe,params,random=False):
    print("Performing grid search...")
    print("parameters:")
    pprint(params)
    
    if random:
        print("Randomized Grid Search")
        gridSearch = RandomizedSearchCV(pipe,params,n_iter=100,cv=3,verbose=2)
    else:
        gridSearch = GridSearchCV(pipe,params,cv=3,verbose=2)
    gridSearch.fit(xx, yy)

    print("Best score: {:0.3f}".format(gridSearch.best_score_))
    print("Best parameters set:")
    best_parameters = gridSearch.best_estimator_.get_params()
    for pName in sorted(params.keys()):
        print("\t{:s}: {}".format(pName, best_parameters[pName]))

main("TweetTiny.csv")    