# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:01:57 2020

@author: User
"""
from twitterScrape import readHuge

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.feature_selection import chi2, GenericUnivariateSelect
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import os
from pprint import pprint
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
from sklearn.compose import ColumnTransformer
import time

path = "T:\\COMP7890\\Full Datasets\Trimmed"
dropcols = ["Unnamed: 0"]

lemma = nltk.stem.WordNetLemmatizer()
tagMap = defaultdict(lambda : wn.NOUN)
tagMap['J'] = wn.ADJ
tagMap['V'] = wn.VERB
tagMap['R'] = wn.ADV

def main(file,one,two,three,models,output=None):    
    times = []
    pipe = Pipeline([
        ("vect",one),
        ("tfidf",two),
        ("clf",three)
        ],
        verbose=True)
    
    params = {
        #'clf__bootstrap': False,
        #'clf__max_depth': None,
        #'clf__max_features': 'sqrt',
    	#'clf__max_samples': 0.5,
    	#'clf__min_samples_leaf': 1,
    	#j'clf__min_samples_split': 3,
    	#'clf__n_estimators': 500,
    	#'clf__warm_start': False,
    	'tfidf__use_idf': False,
    	'vect__max_df': 0.75,
        'vect__ngram_range': (1, 1)
    }
    pipe.set_params(**params)
    
    if 1 in models:
        versionOne(df,pipe,output)
        times.append(time.time())
    if 2 in models:
        versionTwo(df,pipe,output)
        times.append(time.time())
    return times
    
def versionOne(df,pp,oo):
    #Classification with yy value being userid and no other bounds
    df = df.drop("display_name",axis=1)
    
    xx = df.loc[:,"clean_text"]
    yy = df.loc[:,"userid"]
    xx, yy = shuffle(xx, yy)
    
    scores = cross_val_score(pp,xx,yy,scoring='f1_weighted')
    checkPrint(scores,oo)
    checkPrint(scores.mean(),oo)
    '''
    #Results on Tiny
    [0.48410175 0.48186647 0.47763799 0.48405212 0.47794783]
    0.4811212298411828
    '''
    
def lemmaTokens(taggedTokens):
    '''Lemmatize the sentences and get the counts of each part of speech'''
    invalid = "<>#'\"()"
    results = {}
    lemmaText = []
    for tok, tag in taggedTokens:
        if tok in invalid:
            continue
        newTag = tagMap[tag[0]]
        results[newTag] = results.get(newTag,0) + 1
        lem = lemma.lemmatize(tok,newTag)
        lemmaText.append(lem)
    output = [results.get('a',0)/len(lemmaText),results.get('n',0)/len(lemmaText),
              results.get('v',0)/len(lemmaText),results.get('r',0)/len(lemmaText)]
    return [" ".join(lemmaText), len(lemmaText), *output]
    
def makeFeatures(data):
    '''
    Adds info about the tweet to our data'''
    
    data["Tokens"] = data["clean_text"].progress_apply(nltk.word_tokenize)
    data["Tags"] = data["Tokens"].progress_apply(nltk.pos_tag)
    data[["Tweet","Length","A","N","V","R"]] = data.progress_apply(lambda row: lemmaTokens(row.Tags),axis='columns',
                                                                  result_type='expand')
    data.drop(["Tokens","Tags","clean_text"],axis=1,inplace=True)


def featureWeight(ct,xx,yy):
    '''Gets the weight of features'''
    print("Getting Feature Weight")
    zz = ct.fit_transform(xx,yy)
    ff = ct.get_feature_names()
    best = GenericUnivariateSelect(chi2,mode='fpr',param=0.01)
    bb = best.fit(zz,yy)
    newXX = best.transform(zz)
    bf = sorted([(ff[ii], bb.pvalues_[ii]) for ii in bb.get_support(True)],
                key=lambda x: x[1])
    print(len(bf),"features")
    print(bf[:15])
    print(bf[-15:])
    return newXX

def versionTwo(df,pp,oo):
    use = df.copy()
    ct = ColumnTransformer([("words",pp[0],'Tweet')],
                           remainder='passthrough',
                           transformer_weights={"words":1,
                                                "Length":1,
                                                "v":1,
                                                #"a":1,
                                                #"n":1,
                                                "r":1})
    
    pp.steps.insert(0,('column',ct))
    pp.steps.pop(1)
    makeFeatures(use)
    xx = use.loc[:,["Tweet","Length","V","R"]]
    yy = use.loc[:,"userid"]
    xx = featureWeight(ct,xx,yy)
    pp.steps.pop(0) #We already did this one
    runModel(xx,yy,pp,oo)
    '''
    No Feature Selection - Tiny
    CV [0.48465516 0.47083181 0.48422406 0.48730956 0.48257483]
    Avg 0.48191908528454536
    
    After Feature Selection - Tiny
    CV [0.47652993 0.48855923 0.48393948 0.48343486 0.48540567]
    Avg 0.4835738342612304
    
    After Feature Selection Tiny Pt. 2 - Max Depth 75
    [0.23399367 0.23391813 0.23235523 0.23197075 0.23225816]
    0.23289918852928548
    
    [0.33685847 0.33652573 0.33480521 0.33559836 0.32796942]
    0.3343514392456411
    '''
    
    
def runModel(xx,yy,pp,oo):
    xx, yy = shuffle(xx, yy)
    scores = cross_val_score(pp,xx,yy,scoring='f1_weighted')
    checkPrint(scores,oo)
    checkPrint(scores.mean(),oo)
    
def baseDF(file):
    filepath = os.path.join(path,file)
    df = readHuge(filepath).set_index("id")#.drop(dropcols,axis=1)
    checkPrint(df.shape)
    checkPrint(df.groupby("userid")["clean_text"].count())
    return df
    
def checkPrint(txt,file=None):
    if file is None:
        print(txt)
    else:
        file.write("{}\n".format(txt))
    
vv = CountVectorizer(strip_accents="unicode")
#vv = HashingVectorizer()
tt = TfidfTransformer()
nb = MultinomialNB()
rf = RandomForestClassifier(verbose=3,n_jobs=1)
dt = DecisionTreeClassifier()
lr = LogisticRegression(solver='saga')
sv = LinearSVC()

df = baseDF("TweetTiny.csv")

#("Random Forest",rf),("Logistic Regression",lr),

with open("results.txt","w") as output:
    for name, clf in [("Naive Bayes",nb),("Decision Tree",dt),("Random Forest",rf),("Logistic Regression",lr),("SVM",sv)]:
        output.write("{}\n".format(name))
        start = time.time()
        times = main(df,vv,tt,clf,[1,2],output)    
        end = time.time()
        times.insert(0,start)
        times.append(end)
        for ii in range(1,len(times)-1):
            output.write("Time Elapsed (Model {}): {}\n".format(
                ii,times[ii]-times[ii-1]))
        output.write("-"*50+"\n\n")