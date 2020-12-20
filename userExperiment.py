# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:48:59 2020

@author: User
"""

from twitterScrape import readHuge, outPath
from modelPhase2 import baseDF, versionOne, dtParams, svParams, dropcols
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import time
import pandas as pd
import os
from tqdm import tqdm

userBase = "TweetUser"

def buildUserFiles(mainFile,cut=0.5):
    '''
    Assemble the files for testing
    UserX = X tweets per user
    '''
    report = open("report.txt",'w')
    file = readHuge(mainFile)#.drop(dropcols,axis=1)
    index = file["userid"].unique().tolist()
    cutoff = int(len(index) * cut)
    
    report.write("Base: {} users with {} Tweets total\n".format(
        len(index),file.shape[0]))
    
    cnt = 0
    safe = True
    while cnt < 1000 and safe:
        cnt += 25
        dfs = []
        hold = []
        dropids = []
        dropuser = []
        
        for uid in tqdm(index):
            tweets = file[file["userid"]==uid]
            try:
                picked = tweets.sample(cnt)
                dfs.append(picked)
            except:
                hold.append("\tUser {} did not have {} Tweets\n".format(uid,cnt))
                dropids.extend(tweets.index.tolist())
                dropuser.append(uid)
            
        result = pd.concat(dfs)
        outfile = "{}{}.csv".format(userBase,cnt)
        out = os.path.join(outPath,outfile)
        report.write("{} has {} users and {} Tweets\n".format(outfile,
                                                     len(result["userid"].unique()),
                                                     result.shape[0]))
        for hh in hold:
            report.write(hh)
            
        file.drop(dropids,inplace=True)
        for dd in dropuser:
            index.remove(dd)
            
        if len(index) < cutoff: #Less than half the users now
            safe = False
        
        report.write("\n")
        result.to_csv(out)
        print("{} created".format(outfile))
    report.close()
    
def selectUsernames(sizes,numUsers):
    os.chdir(outPath)
    filenames = ["{}{}.csv".format(userBase,ii) for ii in sizes]
    maxFile = readHuge(filenames[-1])
    
    usernames = maxFile["userid"].sample(numUsers)
    #These users must be in all files
    out = os.path.join(outPath,"userSample{}.csv".format(numUsers))
    usernames.to_csv(out)   
    
def makeSmallFiles(sizes,numUsers,sampleSize=None):
    if sampleSize is None:
        sampleSize = numUsers
        
    os.chdir(outPath)
    nameFile = "userSample{}.csv".format(numUsers)
    names = pd.read_csv(nameFile,header=0,index_col=0)["userid"].sample(sampleSize)
    names = names.tolist()
    
    filenames = ["{}{}.csv".format(userBase,ii) for ii in sizes]
    for ff in filenames:
        df = readHuge(ff).drop(dropcols,axis=1).set_index("id")
        sample = df.loc[df["userid"].isin(names),:]
        outname = ff.replace(".csv","small.csv")
        sample.to_csv(outname)
    
def smallFileReport(sizes):
    os.chdir(outPath)
    filenames = ["{}{}small.csv".format(userBase,ii) for ii in sizes]
    for ff in filenames:
        df = readHuge(ff)#.drop(dropcols,axis=1).set_index("id")
        print(ff)
        print(df.shape)
        print(df.groupby("userid")["clean_text"].count())


def runModel(file,one,two,three,params,output=None):    
    pipe = Pipeline([
        ("vect",one),
        ("tfidf",two),
        ("clf",three)
        ],
        verbose=True)
    
    pipe.set_params(**params)
    
    versionOne(file,pipe,output)
    return time.time()

def main(sizes):
    vv = CountVectorizer(strip_accents="unicode")
    tt = TfidfTransformer()
    dt = DecisionTreeClassifier()
    sv = LinearSVC()
    
    params = {
        'tfidf__norm': 'l1',
    	'tfidf__use_idf': False,
    	'vect__max_df': 1.0,
    	'vect__max_features': None,
    	'vect__ngram_range': (1, 1)
        }
    
    models = [("Decision Tree",dt),("SVM",sv)]
    
    filenames = ["{}{}small.csv".format(userBase,ii) for ii in sizes]
    filenames.extend(["TweetTiny.csv","TweetTiny2.csv","TweetTiny3.csv"])
    
    for file in filenames:
        df = baseDF(file)
    
        with open("./Results/resultUser-{}.txt".format(file.replace(".csv","")),"w") as output:
            for name, clf in models:
                output.write("{}\n".format(name))
                output.write("{}\n".format(df.shape))
                start = time.time()
                times = runModel(df,vv,tt,clf,params,output)    
                output.write("Time Elapsed (Model {}): {}\n".format(name,times-start))
                output.write("-"*50+"\n\n")
                
def chk():
    os.chdir(outPath)
    output = []
    for file in os.listdir(outPath):
        df = readHuge(file)
        output.append((file,df.shape))
        
    for out in output:
        print("{}: {}".format(*out))

if __name__ == "__main__":
    #buildUserFiles("TweetDatabaseMulti.csv",0.25)
    #selectUsernames([25,50,75,100,125], 5000)
    #makeSmallFiles([25,50,75,100,125], 5000, 500)
    #smallFileReport([25,50,75,100,125])
    main([25,50,75,100,125])
    #chk()