'''
code for webapp
'''
import os
from uuid import uuid4
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
import flask
# import cv2
from get_recent_tweets import get_tweets
from preprocessing_tweets import text_preprocessing
from prediction import predict_result
from prediction import predict_singletext_result


app = Flask(__name__)
root_dir = os.path.dirname(os.path.abspath(__file__))

#index.html file 
@app.route("/")
def index():
    return render_template("index.html")

#colour identification from uploaded file
@app.route("/checking_username", methods=["POST","GET"])
def search_username():
        if flask.request.method == 'POST':
            username = request.form['username']
            print(username)
            temp = 0
            if username != "":
                tweets_list,tweet_details,user_exist = get_tweets(username)
                if len(tweets_list) >=1:
                    preprocess_tweets = text_preprocessing(tweets_list)
                    temp = int(predict_result(preprocess_tweets))
                    percentage = str(round(predict_result(preprocess_tweets),2))
                    tweets_list = tweets_list[:5]
                    tweet_details = tweet_details[:5]
                else:
                    percentage = 0

                dep = False
                
                if temp > 50 :
                    dep = True
                else:
                    dep = False

                return render_template("index.html", dep = dep, username = username,user_exist=user_exist, tweets_list = tweets_list,tweet_details = tweet_details,percentage=percentage)
            else:
                return render_template("error.html")

@app.route("/checking_text", methods=["POST","GET"])
def search_text():
        if flask.request.method == 'POST':
            normal_text = request.form['normal_text']
            print(normal_text)
            if normal_text != "":
                preprocess_tweets = text_preprocessing([normal_text])
                normal_text_result = predict_singletext_result(preprocess_tweets)
                normal_text_result = [normal_text_result]

            return render_template("index.html", normal_text = normal_text, normal_text_result = normal_text_result)
        else:
            return render_template("error.html")

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/error')
def error():
    return render_template("error.html")

if __name__ == "__main__":
    app.run(debug=True)
