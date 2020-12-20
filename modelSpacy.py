# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:58:36 2020

@author: User
"""

import spacy
from modelPhase2 import baseDF, checkPrint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score as f1
import time

def runSpacy(file,out):    
    nlp = spacy.load("en_core_web_md")
    
    df = baseDF(file)
    xx = df["clean_text"]
    yy = df["userid"]
    xx, yy = shuffle(xx, yy)
    
    trainX, testX, trainY, testY = train_test_split(xx,yy,test_size=0.2)
    
    start = time.time()
    
    docs = {}
    for user in tqdm(trainY.unique().tolist()):
        tweets = trainX.loc[trainY==user].tolist()
        tweetDoc = ". ".join(tweets).replace("..",".")
        docs[user] = nlp(tweetDoc)
        
    def getSimilarity(base,target):
        return docs[target].similarity(base)
    
    def getConfidence(row):
        if row["Best"] == 0:
            return 0
        diff = (row["Best"] - row["Second"]) / row["Best"]
        return diff
    
    evaluate = pd.concat([testX,testY],axis=1)
    evaluate["DOC"] = evaluate["clean_text"].progress_apply(lambda x: nlp(x))
        
    for col in tqdm(trainY.unique()):
        evaluate[col] = evaluate["DOC"].apply(getSimilarity,target=col)
    evaluate.drop(["DOC"],axis=1,inplace=True)
    evaluate["Guess"] = evaluate.loc[:,trainY.unique()].idxmax(axis=1)
    
    #Unweighted
    score = f1(evaluate["userid"],evaluate["Guess"],average='weighted')
    checkPrint("Unweighted F1: {}".format(score),out)
    mid = time.time()
    checkPrint("Time Taken: {} seconds".format(mid-start),out)
    
    #Weighted    
    evaluate["Best"] = evaluate.loc[:,trainY.unique()].max(axis=1)
    evaluate["Second"] = evaluate.loc[:,trainY.unique()].apply(lambda row: row.nlargest(2).values[-1],axis=1)
    evaluate["Confidence"] = evaluate.apply(getConfidence,axis=1)
    score = f1(evaluate["userid"],evaluate["Guess"],average='weighted',
               sample_weight=evaluate["Confidence"])
    checkPrint("Confidence Weighted F1: {}".format(score),out)
    
    end = time.time()
    checkPrint("Time Taken: {} seconds".format(end-start),out)
    
    #evaluate.to_csv("Results/SpacyTest.csv")
    
def main():
    with open("Results/spacyOutput.txt","w") as output:
        for file in ["TweetTiny.csv","TweetTiny2.csv",
                     "TweetTiny3.csv"]:
            output.write("{0}{1}{0}\n".format("="*20,file))
            runSpacy(file,output)
            output.write("="*30+"\n")

if __name__ == "__main__":
    main()    
