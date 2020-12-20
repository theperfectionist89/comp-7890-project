# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:18:23 2020

@author: User
"""
from twitterScrape import readHuge
from modelPhase2 import checkPrint, baseDF

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import sklearn.cluster as cluster
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

import time

path = "T:\\COMP7890\\Full Datasets\Trimmed"

def main(file,one,two,three,params,output=None):    
    pipe = Pipeline([
        ("vect",one),
        ("tfidf",two),
        ("clf",three)
        ],
        verbose=True)
    
    pipe.set_params(**params)
    
    clustered(file,pipe,output)
    return time.time()

def clustered(df,pp,oo):
    df = df.drop("display_name",axis=1)
    
    xx = df.loc[:,"clean_text"]
    yy = df.loc[:,"userid"]
    xx, yy = shuffle(xx, yy)
    
    scores = cross_val_score(pp,xx,yy,scoring='v_measure_score')
    checkPrint(scores,oo)
    checkPrint(scores.mean(),oo)
    
def setup():
    vv = CountVectorizer(strip_accents="unicode")
    tt = TfidfTransformer()
    km = cluster.KMeans()
    mi = cluster.MiniBatchKMeans()
    
    baseParams = {
    	'tfidf__use_idf': False,
    	'vect__max_df': 0.5,
        'vect__ngram_range': (1, 1)
    }
    
    kmParams = {"clf__verbose":1}
    miParams = {"clf__verbose":1}
    
    models = [#("KMeans",km,kmParams),
              ("MiniBatch",mi,miParams)
              ]
    
    with open("resultsCluster.txt","w") as output:
        for suffix in [""]:#,"2","3"]:
            #dfName = "TweetTiny{}.csv".format(suffix)
            dfName = "TweetDatabaseMulti.csv".format(suffix)
            df = baseDF(dfName)
            clusters = len(df["userid"].unique())
            output.write("====={}=====\n".format(dfName))
            for name, clf, params in models:
                allParams = {}
                allParams.update(baseParams)
                allParams.update(params)
                allParams["clf__n_clusters"] = clusters
                
                output.write("{}\n".format(name))
                start = time.time()
                times = main(df,vv,tt,clf,allParams,output)    
                output.write("Time Elapsed (Model {}): {}\n".format(
                        name,times-start))
                output.write("-"*50+"\n\n")
            
if __name__ == "__main__":
    setup()