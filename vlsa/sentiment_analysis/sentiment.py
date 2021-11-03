"""Author:Howard Mabhugu
National University Of Science And Technology
File Name: sentiment.py
Version: 1.0
Function: This module is used for predicting the sentiment of a comment
Input: comment from the user.
Output: The sentiment of the comment (0 for negative and 1 for positive)
"""
# Importing the required libraries
import os
import sys
import time
import datetime
import re
import nltk
from django.utils import timezone
import joblib
import django
from webbrowser import get

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "vlsa.settings")
django.setup()
#sys.path.append(os.path.dirname(__file__) + "C:/Users/Bhugs Hardy/Desktop/VLSA/vlsa/sent_model")
from sentiment_analysis.models import Post
from django.views.generic import (ListView, DetailView, CreateView)
from django.utils.datetime_safe import date
import pytz

from django.shortcuts import render
from django.http import request
from django.template import Library
from django.contrib.auth.decorators import login_required
from users.forms import UserRegisterForm
from django.contrib import messages
from webbrowser import get


# Processing Sentiments

def preprocessSentiments(comment):

    # trim
    comment = comment.strip('\'"')

    # Repeating words like happyyyyyyyy
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    comment = rpt_regex.sub(r"\1\1", comment)

    # Emoticons
    emoticons = \
        [
            ('__positive__', [':-)', ':)', '(:', '(-:', \
                              ':-D', ':D', 'X-D', 'XD', 'xD', \
                              '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ]), \
            ('__negative__', [':-(', ':(', '(:', '(-:', ':,(', \
                              ':\'(', ':"(', ':((', ]), \
            ]

    def replace_parenth(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def regex_join(arr):
        return '(' + '|'.join(arr) + ')'

    emoticons_regex = [(repl, re.compile(regex_join(replace_parenth(regx)))) \
                       for (repl, regx) in emoticons]

    for (repl, regx) in emoticons_regex:
        comment = re.sub(regx, ' ' + repl + ' ', comment)

    # Convert to lower case
    comment = comment.lower()

    return comment


# Stemming of comments

def stem(comment):
    stemmer = nltk.stem.PorterStemmer()
    comment_stem = ''
    words = [word if (word[0:2] == '__') else word.lower() \
             for word in comment.split() \
             if len(word) >= 3]
    words = [stemmer.stem(w) for w in words]
    comment_stem = ' '.join(words)
    return comment_stem

# Predict the sentiment

def predict(comment, classifier):
    comment_processed = stem(preprocessSentiments(comment))

    if '__positive__' in comment_processed:
        sentiment = 1
        return sentiment

    elif '__negative__' in comment_processed:
        sentiment = 0
        return sentiment
    else:

        X = [comment_processed]
        sentiment = classifier.predict(X)
        return sentiment[0]

    
# Main function
def main():
    print('Loading the Classifier, please wait....')
    classifier = joblib.load('svmClassifier.pkl')
    print('READY')
    context = {
        'posts': Post.objects.all()
    }
    comment=str(context)
    print(predict(comment, classifier))


if __name__ == '__main__':
    main()

# Date Ranges Info
# context = {
#         'posts': Post.objects.filter(date_posted__gte=date(2011,1,1)).filter(date_posted__lte=date(2020,12,31))
#     }