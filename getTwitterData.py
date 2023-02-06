# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:48:11 2023

@author: Akhilesh
"""

import os
import time
from ast import literal_eval
from datetime import datetime
import pandas as pd
import snscrape.modules.twitter as sntwitter # the magic
import warnings
warnings.filterwarnings('ignore')
root_dir = os.path.abspath(os.path.join(os.path.dirname("."), '.'))
class ExtractTweets:
    
    def __init__(self, outpathname, outfilename,
                 minTweetCountPerDay=10, 
                 minRetweetCount=0,
                 minLikeCount=0, 
                 minFollowersCount=0, 
                 VerifiedStatus=None, 
                 saveBufferDuration=3600,
                 ):
        """
        Accepts basic input params each of integer datatype, except for VerifiedStatus which accepts boolean or None.
        """
        self.start_timer = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
        self.min_tweet_count_perDay = minTweetCountPerDay
        self.minRetweetCount = minRetweetCount
        self.minLikeCount = minLikeCount
        self.minFollowersCount = minFollowersCount
        self.VerifiedStatus = VerifiedStatus                           
        self.tweets_df = pd.DataFrame(columns=['date', 'tweet', 'lang', 'retweetCount', 'likeCount', 'replyCount', 
                                               'username', 'user_followersCount','user_friendsCount', 'verifiedStatus', 
                                               'tweet_url', 'hastags', 'chr_count', 'topic'])
        self.save_buffer_duration = saveBufferDuration
        self.outpathname = outpathname
        self.outfilename = outfilename
        return

    def save_copy(self):
        """
        Saves a temp copy for restoration and prevent API time limit exceed error.
        
        :return:
        pandas dataframe containing twitter record-data.
        """
        data = self.tweets_df.reset_index(drop=True)
        data['date'] = data['date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
        for filename in os.listdir("."):
            if filename.endswith('local.csv'):
                os.remove(filename)
        data.to_csv(self.outpathname+self.outfilename+".csv", index=False)
        return data
        
    def getTweets(self, start_date, end_date, keywords):
        """
        Extracts historical twitter data.
        
        :params:
        start_date - str in "YYYY-MM-DD" format
        end_date - str in "YYYY-MM-DD" format
        keywords - list of tuples, 
            e.g, [('recession'), ('football, 'worldcup', 'fifa'), ('war', 'ukraine')]
            e.g. ['recession']
        
        :return:
        pandas dataframe with features as:
         date: Tweet Timestamp
         tweet: tweet content
         lang: language classifer used by parent api
         retweetCount: tweet retweeted count
         likeCount: tweet like count
         replyCount: number of replies to original tweet
         username: user who tweeted
         user_followersCount: number of followers user has (tells you how popular the avg tweets are)
         user_friendsCount: number of friends user has
         verifiedStatus: If the user is Verified or not (i.e. pays 8 bucks every month!)
         tweet_url: Link of original tweet (click and see)
         hastags: If any hastags were used (hastags are important for search and info retrieval)
         chr_count: number of english characters in the original tweet
         topic: keywords you used for searching tweets (kind of labels)
        """
        
        if not(isinstance(keywords, list) or isinstance(keywords, tuple)):
            raise Exception("Incorrect Input Format! Please pass a list")
        
        for topic in keywords:
            # for saving local copies every buffer_hour
            st_time = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
            date = pd.to_datetime(start_date, format='%Y-%m-%d')
            e_date = pd.to_datetime(end_date, format='%Y-%m-%d') + pd.to_timedelta(1, unit='d')
            if isinstance(topic, tuple) or isinstance(topic, list):
                topic = " ".join(topic)
            search_query = topic
            print("search_query:", search_query)
        
            while date != e_date:
                nxt_date = date + pd.to_timedelta(1, unit='d')
                content = '{} since:{} until:{} near: United States'.format(search_query, date.strftime('%Y-%m-%d'), nxt_date.strftime('%Y-%m-%d'))
                print(content)
                
                # check for save buffer duration (set to 1 Hr by default)
                delta_buffer = (datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") - st_time).seconds
                if delta_buffer >= self.save_buffer_duration:
                    self.save_copy()
                    # reset buffer
                    st_time = datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

                lst_tweets = []
                for counter, tweet in enumerate(sntwitter.TwitterSearchScraper(content).get_items()):
                    if counter+1 > self.min_tweet_count_perDay: 
                        break
                    if tweet.likeCount >= self.minLikeCount \
                        or tweet.retweetCount >= self.minRetweetCount \
                        or tweet.user.followersCount >= self.minFollowersCount \
                        or (tweet.user.verified and isinstance(tweet.user.verified, bool) and tweet.user.verified == self.VerifiedStatus):
                        
                        # ----------------------------------------------------------------
                        # Potential custom preprocessing module here: 
                        # 1. Simple and short: https://www.kaggle.com/code/zenbird01/pranjalpathak-semantic-clustering-v1-0/notebook
                        # 2. Advanced: ./NLP_basics_preprocessing_vectorization_similarity.ipynb
                        # 3. Best: Check github - https://github.com/pranzell/NLP_Tools
                        # ----------------------------------------------------------------
                        
                        lst_tweets.append([
                            tweet.date, 
                            tweet.content, 
                            tweet.lang,
                            tweet.retweetCount,
                            tweet.likeCount,
                            tweet.replyCount,
                            tweet.user.username, 
                            tweet.user.followersCount, 
                            tweet.user.friendsCount, 
                            tweet.user.verified,
                            tweet.url,
                            tweet.hashtags,
                            len(str(tweet.content).strip()),
                            topic])
                
                self.tweets_df = self.tweets_df.append(pd.DataFrame(lst_tweets, columns=self.tweets_df.columns))
                date = nxt_date
        
        print("\n\nTOTAL TIME TAKEN {} minutes".format(((datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") - self.start_timer).seconds)/60.0))
        return self.save_copy()
    
    
    def preprocess_shortText(self, text_col):
        # refer to Preprocessing ipynb file
        # https://github.com/pranzell/NLP_Tools
        pass