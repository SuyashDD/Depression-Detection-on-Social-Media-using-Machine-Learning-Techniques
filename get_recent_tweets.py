# ! pip install tweepy
import tweepy
import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
import ast
import string
import datetime


# input your credentials here
consumer_key = 'EcwAltcOolBlwQlesRg2Ux5A8'
consumer_secret = 'tMltHbelqqDAWJhoTSPa2TDrB4LWL4FK9B40F7qGsU7wUEmN9J'
access_token = '1331612828281683973-9SYgUjN2pYpoTVpONPRh3SCsMsMbB1'
access_token_secret = 'n901ArAC0NHH1EExcCEn8BMApnGfjejHM6V0Cz3wtS5Mq'

#accessing data using twitter api via tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def get_tweets(userID):
    user_exist = False
    try:
        tweets = api.user_timeline(screen_name=userID, 
                                # 200 is the maximum allowed count
                                count=200,
                                include_rts = False,
                                # Necessary to keep full_text 
                                # otherwise only the first 140 words are extracted
                                tweet_mode = 'extended'
                                )

        user_exist = True
        raw_tweets = []
        for tweet in tweets:
            raw_tweets.append(tweet.full_text)

        tweet_details = []
        for tweet in tweets:
            tweet_details.append([tweet.created_at,tweet.full_text])

        return raw_tweets, tweet_details, user_exist

    except Exception:

        return list(),list(), user_exist
