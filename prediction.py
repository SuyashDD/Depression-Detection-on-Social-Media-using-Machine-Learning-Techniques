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
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer



def predict_result(tweets_list):
    df = pd.DataFrame({'tweets':tweets_list})

    df = df.astype({'tweets':str})
    X = df['tweets']
    print("X :")
    print(X)
    transformer = TfidfTransformer()
    
    # countvectorizer saved model
    root = "saved_model/"
    c_name = root +"Feature.pkl"
    loaded_vec = CountVectorizer(decode_error="replace",lowercase=False,vocabulary=pickle.load(open(c_name, "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(X))
    
    # model name
    filename = root +"Pickle_Naive.pkl"
    loaded_model = joblib.load(open(filename, 'rb'))

    # result prediction
    result = loaded_model.predict(tfidf)
    u, counts = np.unique(result, return_counts=True)
    dictionary = dict(zip(u, counts))
    print(dictionary)
    # percentage calculation
    print(result)
    percentage = 0
    if 1 in dictionary:
        one_count = dictionary[1]
        percentage = 0
    if 0 in dictionary:
        zero_count = dictionary[0]
        percentage = (zero_count / len(result))*100
    

    return percentage


def predict_singletext_result(normal_text):

    df = pd.DataFrame({'tweets':normal_text})

    df = df.astype({'tweets':str})
    X = df['tweets']

    transformer = TfidfTransformer()
    # countvectorizer saved model
    root = "saved_model/"
    c_name = root +"Feature.pkl"
    loaded_vec = CountVectorizer(decode_error="replace",lowercase=False,vocabulary=pickle.load(open(c_name, "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(X))
    # model name
    filename = root +"Pickle_Naive.pkl"
    loaded_model = joblib.load(open(filename, 'rb'))

    # result prediction
    result = loaded_model.predict(tfidf)

    if result == 0:
        normal_text_result = '0'
    else:
        normal_text_result = '1'
    
    return normal_text_result