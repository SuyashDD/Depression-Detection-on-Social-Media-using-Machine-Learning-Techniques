# importing all required libraries
import pandas as pd
import numpy as np
import sklearn
from textblob import TextBlob
import os
import re
import emoji
import ast
import string
import random
import nltk
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# function for replacing urls(starting from http) with an empty string.
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# funtion for replacing usernames(starting with @___) with an empty string
def remove_usernames(text):
    return re.sub('@[^\s]+','',text)

def remove_hashtags(text):
    return re.sub('[!@_^%*#$=-]', '', text)

def remove_emojis(text):
    allchars = [j for j in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([k for k in text.split() if not any(i in k for i in emoji_list)])
    return clean_text

def remove_digits(text):
    return re.sub(r'\b\d+\b', '', text)

def remove_digits_str(text):
    return re.sub(r'\d+', '', text)


#make n't as not
nt_words = {"isnt":"is not",
            "didn't":"did not", 
            "can't":"can not",
            "shouldn't":"should not",
            "couldn't":"could not",
            "don't":"do not",
            "hadn't":"had not",
            "haven't":"have not",
            "hasn't":"has not",
            "aren't":"are not",
            "didnt":"did not", 
            "cant":"can not",
            "shouldnt":"should not",
            "couldnt":"could not",
            "dont":"do not",
            "hadnt":"had not",
            "havent":"have not",
            "hasnt":"has not",
            "arent":"are not",
            "isnt":"is not"
            }
            
def nt_replacement(text):
  text = text.split(" ")
  for i in range(len(text)):
    if text[i] in nt_words:
      text[i] = nt_words[text[i]]

  return ' '.join(text)


stop_words = set(STOPWORDS)
stop_words.add("xa0")
stop_words.add("https")
stop_words.add("twitter")
stop_words.add("tweet")
stop_words.add("pic")
stop_words.add("i")
stop_words.add("com")


def text_preprocessing(raw_tweets):
    raw_tweets = filter(None, raw_tweets)
    raw_tweets = list(map(remove_urls,raw_tweets))
    raw_tweets = list(map(remove_usernames,raw_tweets))
    raw_tweets = list(map(nt_replacement,raw_tweets))
    raw_tweets = list(map(lambda x:''.join([i for i in x if i not in string.punctuation]),raw_tweets))
    raw_tweets = list(map(remove_emojis,raw_tweets))
    raw_tweets = list(map(remove_hashtags,raw_tweets))
    raw_tweets = list(map(remove_digits,raw_tweets))
    raw_tweets = list(map(remove_digits_str,raw_tweets))
    final_tweets = filter(None, raw_tweets)
    final_tweets = list(map(lambda x: x.encode('ascii', 'ignore').decode('ascii'),final_tweets))
    raw_tweets = [word.lower() for word in final_tweets]
    raw_tweets = [word_tokenize(word) for word in raw_tweets]

        # Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    finalize_list = []
    for index,entry in tqdm(enumerate(raw_tweets)):
        Final_words = []
        word_Lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        
        finalize_list.append(Final_words)


    output = []
    for i in finalize_list:
        if len(i) >= 1:
            output.append(i)

    return output