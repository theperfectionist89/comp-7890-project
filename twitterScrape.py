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
import re
import numpy as np
from tqdm import tqdm
tqdm.pandas()

#path = "C:\\Users\\User\\Desktop\\Large Datasets\\twitter-events-2012-2016\\Full Datasets"
path = "T:\\COMP7890\\Full Datasets"
outPath = os.path.join(path,"Trimmed")
mainFile = "TweetDatabase.csv"
cleanFile = "TweetDatabaseClean.csv"
multiFile = "TweetDatabaseMulti.csv" 
dupsFile = "TweetDatabaseDuplicates.csv"

tinyFile = "TweetTiny.csv"
tinyFile2 = "TweetTiny2.csv"

#created_at, lang may also be useful
keepList = ["text","id","user_screen_name","user_name"]

#dates = ['created_at','user_created_at']
df_dtype = {'coordinates': str, 
            'created_at': str, 
            'hashtags': str, 
            'media': str, 
            'urls': str, 
            'favorite_count': np.int32, 
            'id': np.int64, 
            'in_reply_to_screen_name': str, 
            'in_reply_to_status_id': "Int64", 
            'in_reply_to_user_id': "Int64", 
            'lang': str, 'place': str, 
            'possibly_sensitive': 'boolean', 
            'retweet_count': np.int32, 
            'retweet_id': "Int64", 
            'retweet_screen_name': str, 
            'source': str, 
            'text': str, 
            'tweet_url': str, 
            'user_created_at': str, 
            'user_screen_name': str, 
            'user_default_profile_image': 'boolean', 
            'user_description': str, 
            'user_favourites_count': np.int32, 
            'user_followers_count': np.int32, 
            'user_friends_count': np.int32, 
            'user_listed_count': np.int32, 
            'user_location': str, 
            'user_name': str, 
            'user_screen_name.1': str, 
            'user_statuses_count': np.int32, 
            'user_time_zone': str, 
            'user_urls': str, 
            'user_verified': bool}

pat1 = re.compile("RT @\w+:")
pat2 = re.compile("@\w+")
pat3 = re.compile("http:\/\/\S+")
pat4 = re.compile("#\S+")
pats = [pat1,pat2,pat3,pat4]

def readHuge(file,chunk=10**5):
    dfs = []
    for chk in tqdm(pd.read_csv(file,chunksize=chunk,dtype=df_dtype)):
        #print("Processed {} items of {}".format(chunk,file))
        dfs.append(chk)
    print("\nRead {}".format(file))
    return pd.concat(dfs)

def processBigData():
    folder = os.chdir(path)
    for file in os.listdir(folder):
        if file[-3:] == "csv":
            print("Reading {}".format(file))
            df = readHuge(file)
            df = df.loc[df["lang"]=="en",keepList]
            #droplist = [ii for ii in df.columns if ii not in keepList]
            df = df.set_index("id")
            
            df.to_csv(outPath+"\\"+file)
            print("\nDone {}".format(file))
    
def mergeSmallData():
    folder = os.chdir(outPath)
    dfs = []
    for file in os.listdir(folder):
        if file[:5] == "Tweet": #My files. Ignore
            continue
        print("Reading {}".format(file))
        df = readHuge(file)
        dfs.append(df)
    print("Merging...")
    result = pd.concat(dfs).set_index("id").rename(
        columns={"user_name":"display_name","user_screen_name":"userid"})
    print("Writing...")
    print(result.columns)
    print(result.head())
    result.to_csv(mainFile)   
    
def trimUnique(file):
    filepath = os.path.join(path,outPath,file)
    df = readHuge(filepath).set_index("id").dropna()
    df.columns = df.columns.str.strip()
    
    hashtag = df["display_name"].str.startswith("#")
    df = df.loc[~hashtag,:]
    
    minSize = 50
    df = getMany(df,minSize)
    
    outpath = os.path.join(path,outPath,multiFile.replace(
        ".csv,","{}.csv".format(minSize)))
    df.to_csv(outpath)
    print("{} created".format(multiFile))
    
    #Eliminate display names with unique usernames
    gp = df.groupby(["display_name"])["userid"].nunique() > 1
    names = df["display_name"].dropna().unique()
    names.sort()
    
    multinames = names[gp]
    keep = df.loc[df["display_name"].isin(multinames)].sort_values(
        by=["display_name","userid"])
    
    outpath2 = os.path.join(path,outPath,dupsFile)
    keep.to_csv(outpath2)
    
def makeTiny(file,ids=100,rows=10000):
    filepath = os.path.join(path,outPath,file)
    df = readHuge(filepath)
    idx = df["userid"].sample(ids)
    result = df[df["userid"].isin(idx)]
    while result.size < rows:
        idx = df["userid"].sample(ids)
        result = df[df["userid"].isin(idx)]
        
    result = result.sort_values(by=["display_name","userid"])
    result.to_csv(os.path.join(path,outPath,tinyFile2))
    
def stripLinks(txt):
    #Remove mentions and links
    for pp in pats:
        txt = re.sub(pp,"",txt)
    return re.sub(re.compile("\s{2,}")," ",txt).strip()

def getMany(df,cutoff=50):
    ui = df["userid"].value_counts() > cutoff
    good = ui[ui].index
    return df.loc[df["userid"].isin(good),:]
    
def process(file):
    filepath = os.path.join(path,outPath,file)
    df = readHuge(filepath,10**5).set_index("id")#.drop(["Unnamed: 0"],axis=1)
    print("Editing...")
    df["display_name"] = df["display_name"].progress_apply(lambda x: str(x).strip())
    df["userid"] = df["userid"].progress_apply(lambda x: str(x).strip())
    df["clean_text"] = df["text"].progress_apply(stripLinks)
    
    outpath = os.path.join(path,outPath,cleanFile)
    df.drop("text",axis=1).to_csv(outpath)
    
    
def main():
    #processBigData()
    #mergeSmallData()
    #process(mainFile)
    #trimUnique(cleanFile)
    makeTiny(multiFile,250,20000)
    
if __name__ == "__main__":
    main()