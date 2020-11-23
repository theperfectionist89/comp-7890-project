# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:29:14 2020

@author: User
"""


'''
Dataset used from
Zubiaga, A. (2018), A longitudinal assessment of the persistence of twitter datasets. 
Journal of the Association for Information Science and Technology, 69: 974-984. 
https://doi.org/10.1002/asi.24026
'''
import os
import pandas as pd

path = "C:\\Users\\User\\Desktop\\Large Datasets\\twitter-events-2012-2016\\Full Datasets"
outPath = os.path.join(path,"Trimmed")
mainFile = "TweetDatabase.csv"
smallFile = "TweetDatabaseSmall.csv" 

keepList = ["created_at","text","id","lang","user_screen_name","user_name"]

def processBigData():
    folder = os.chdir(path)
    for file in os.listdir(folder):
        if file[-3:] == "csv":
            df = pd.read_csv(file)
            droplist = [ii for ii in df.columns if ii not in keepList]
            df = df.set_index("id").drop(droplist,axis=1)
            
            df.to_csv(outPath+"\\"+file)
    
def mergeSmallData():
    folder = os.chdir(outPath)
    dfs = []
    for file in os.listdir(folder):
        if file == mainFile or file == smallFile:
            continue
        df = pd.read_csv(file)
        dfs.append(df)
    result = pd.concat(dfs).set_index("id").rename(
        columns={"user_name":"display_name","user_screen_name":"userid"})
    result["display_name"] = result["display_name"].apply(lambda x: str(x).strip())
    result["userid"] = result["userid"].apply(lambda x: str(x).strip())
    result.to_csv(mainFile)
    
def trimUnique():
    filepath = os.path.join(path,outPath,mainFile)
    df = pd.read_csv(filepath).set_index("id")
    df.columns = df.columns.str.strip()
    
    gp = df.groupby(["display_name"])["userid"].nunique() > 1
    names = df["display_name"].dropna().unique()
    names.sort()
    
    multinames = names[gp]
    keep = df.loc[df["display_name"].isin(multinames)].sort_values(
        by=["display_name","userid"])
    hashtag = keep["display_name"].str.startswith("#")
    output = keep[~hashtag]
    
    outpath = os.path.join(path,outPath,smallFile)
    output.to_csv(outpath)
    
def process():
    filepath = os.path.join(path,outPath,smallFile)
    df = pd.read_csv(filepath)
    pass
    
    
    
def main():
    #processBigData()
    #mergeSmallData()
    trimUnique()
    #process()
    
main()