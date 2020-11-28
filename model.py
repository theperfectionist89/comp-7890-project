# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:01:57 2020

@author: User
"""
from twitterScrape import readHuge

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
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
    mm = MultinomialNB(alpha=1e-05)
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    hg = HistGradientBoostingClassifier()
    sv = SVC()
    
    #print(rf.get_params())
    #print(gb.get_params())
    #print(hg.get_params())
    #print(sv.get_params())
    
    pipe = Pipeline([
        ("vect",vv),
        ("tfidf",tt),
        ("clf",rf)
        ],
        verbose=True)
    
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
         'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        #'clf__max_iter': (20,),
        'clf__alpha': (0.00001, 0.000001)
        #'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__max_iter': (10, 50, 80),
    }
    
    versionOne(df,pipe,parameters)
    
def versionOne(df,pp,param):
    #Classification with yy value being userid and no other bounds
    df = df.drop("display_name",axis=1)
    
    xx = df.loc[:,"clean_text"]
    yy = df.loc[:,"userid"]
    
    #gridSearch(xx,yy,pp,param)
    
    scores = cross_val_score(pp,xx,yy)
    print(scores)
    
def gridSearch(xx,yy,pipe,params):
    print("Performing grid search...")
    print("parameters:")
    pprint(params)
    
    gridSearch = GridSearchCV(pipe,params,n_jobs=-1,verbose=1)
    gridSearch.fit(xx, yy)

    print("Best score: {:0.3f}".format(gridSearch.best_score_))
    print("Best parameters set:")
    best_parameters = gridSearch.best_estimator_.get_params()
    for pName in sorted(params.keys()):
        print("\t{:s}: {}".format(pName, best_parameters[pName]))
    


main("TweetTiny.csv")    