from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import (ListView, DetailView, CreateView, UpdateView)


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
from django.views.decorators.csrf import csrf_exempt
from sentiment_analysis.models import Post
from django.utils.datetime_safe import date
import pytz
import pandas as pd

from django.shortcuts import render
from django.http import request
from django.contrib.auth.decorators import login_required
from users.forms import UserRegisterForm
from django.contrib import messages
from django.views.generic import TemplateView
from django.core import serializers
from rest_framework import views
from rest_framework import viewsets
from .serialize import PostSerializer
from rest_framework.decorators import api_view
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework import status
from django.urls import reverse_lazy
from bootstrap_modal_forms.generic import BSModalLoginView
from bootstrap_modal_forms.generic import BSModalCreateView
from users.forms import CustomUserCreationForm, CustomAuthenticationForm
from rest_framework.renderers import TemplateHTMLRenderer
from django.contrib.messages.api import success
from django.http import HttpResponseRedirect
from rest_framework.views import APIView
from datetime import datetime, timedelta, time

from io import StringIO

today = datetime.now()
start_week = today - timedelta(7)



def index(request):
    return render(request, 'sentiment_analysis/index.html')


def comment(request):
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'sentiment_analysis/comment.html', context)


class PostListView(ListView):
    model = Post
    template_name = 'sentiment_analysis/comment.html'
    context_object_name = 'posts'
    ordering = ['-date_posted']


class PostDetailView(DetailView):
    model = Post


def about(request):
    return render(request, 'sentiment_analysis/about.html')


def login(request):
    return render(request, 'sentiment_analysis/login.html')


def register(request):
    return render(request, 'sentiment_analysis/register.html')


def forgot_password(request):
    return render(request, 'sentiment_analysis/forgot_password.html')


def charts(request):
    return render(request, 'sentiment_analysis/charts.html')


def buttons(request):
    return render(request, 'sentiment_analysis/buttons.html')


@csrf_exempt
def report(request):

    return render(request, 'sentiment_analysis/report.html')

@csrf_exempt
def test(request):

    return render(request, 'sentiment_analysis/test.html')


class PostCreateView(CreateView):
    model = Post
    fields = ['sentiment']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)


class PostUpdateView(LoginRequiredMixin, UpdateView):
    model = Post
    fields = ['sentiment', 'polarity']

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)

# Processing Sentiments

def preprocessSentiments(comment):
    # Convert www.* or https?://* to URL
    comment = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', comment)

    # Convert @username to __HANDLE
    comment = re.sub('@[^\s]+', '__HANDLE', comment)

    # Replace #word with word
    comment = re.sub(r'#([^\s]+)', r'\1', comment)

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

class PostView(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

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


def dashboard(request): 
    return render(request, 'sentiment_analysis/dashboard.html')


def predictionAnnual(request):
    predictionAnnually(request)
    population_chart(request) 
    if request.method=='POST':
        answer=answerValue
        messages.success(request,'The sentiment value of this year is:',extra_tags=answer)
    return render(request, 'sentiment_analysis/dashboard.html')

@csrf_exempt  
def predictionAnnually(request):
        try:
            classifier = joblib.load('sent_model/model_src/sent_model/svmClassifier.pkl')

            global posOne
            global negTwo
            posOne=0
            negTwo=0
            sentiment_iterator = Post.objects.values_list('sentiment').filter(date_posted__year=today.year).iterator()
            for i in sentiment_iterator:
                comment=str(i)
                
                answerQ=predict(comment, classifier)
                if answerQ==0:
                    negTwo=negTwo+1
                else:
                    posOne=posOne+1


            context = {
                 'posts': Post.objects.values('sentiment').filter(date_posted__year=today.year)
            }
            comment=str(context)
            answer=predict(comment, classifier)
            sentiment_iterator = Post.objects.values('sentiment').filter(date_posted__year=today.year)
            global answerValue
            if answer==0:
                answerValue="Negative"
            else:
                answerValue="Positive"

            return answerValue
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

  

def predictionMonth(request):
    predictionMonthly(request) 
    population_chart(request)
    if request.method=='POST':
        answer=answerValue
        messages.success(request,'The sentiment value of this month is:',extra_tags=answer)
    return render(request, 'sentiment_analysis/dashboard.html')


@csrf_exempt
def predictionMonthly(request):
        try:
            classifier = joblib.load('sent_model/model_src/sent_model/svmClassifier.pkl')

            global posOne
            global negTwo
            posOne=0
            negTwo=0
            sentiment_iterator = Post.objects.values_list('sentiment').filter(date_posted__year=today.year, date_posted__month=today.month).iterator()
            for i in sentiment_iterator:
                comment=str(i)
                
                answerQ=predict(comment, classifier)
                if answerQ==0:
                    negTwo=negTwo+1
                else:
                    posOne=posOne+1

            context = {
                'posts': Post.objects.values('sentiment').filter(date_posted__year=today.year, date_posted__month=today.month)
            }
            comment=str(context)
            answer=predict(comment, classifier)
            global answerValue
            if answer==0:
                answerValue="Negative"
            else:
                answerValue="Positive"

            return answerValue
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)


def predictionWeek(request):
    predictionWeekly(request) 
    population_chart(request)
    if request.method=='POST':
        answer=answerValue
        # listArray= commentArray
        messages.success(request,'The sentiment value for this Week is:',extra_tags=answer)
    return render(request, 'sentiment_analysis/dashboard.html')
  
@csrf_exempt  
def predictionWeekly(request):
        try:
            classifier = joblib.load('sent_model/model_src/sent_model/svmClassifier.pkl')

            global posOne
            global negTwo
            posOne=0
            negTwo=0
            sentiment_iterator = Post.objects.values_list('sentiment').filter(date_posted__gte=start_week).filter(date_posted__lte=today).iterator()
            for i in sentiment_iterator:
                comment=str(i)
                
                answerQ=predict(comment, classifier)
                if answerQ==0:
                    negTwo=negTwo+1
                else:
                    posOne=posOne+1

            context = {
                 'posts': Post.objects.values('sentiment').filter(date_posted__gte=start_week).filter(date_posted__lte=today)
            }

            comment=str(context)
            global answerValue
            commentArray=comment
            answer=0 
            print(answer)
            if answer==0:
                answerValue="Negative"
            else:
                answerValue="Positive"

            return answerValue
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

def predictionToday(request):
    predictionTodaly(request) 
    if request.method=='POST':
        answer=answerValue
        messages.success(request,'The sentiment value for Today is:',extra_tags=answer)
    return render(request, 'sentiment_analysis/dashboard.html')


@csrf_exempt  
def predictionTodaly(request):
        try:
            classifier = joblib.load('sent_model/model_src/sent_model/svmClassifier.pkl')

            global posOne
            global negTwo
            posOne=0
            negTwo=0
            counter=0
            sentiment_iterator = Post.objects.values_list('sentiment').filter(date_posted__year=today.year, date_posted__month=today.month, date_posted__day=today.day).iterator()
            for i in sentiment_iterator:
                comment=str(i)
                counter=counter+1
                answerQ=predict(comment, classifier)
                if answerQ==0:
                    negTwo=negTwo+1
                else:
                    posOne=posOne+1

            context = {
                 'posts': Post.objects.values('sentiment').filter(date_posted__year=today.year, date_posted__month=today.month, date_posted__day=today.day)
            }

            comment=str(context)
            global answerValue
            commentArray=comment
            answer=predict(comment, classifier)
            sentiment_iterator = Post.objects.values('sentiment').filter(date_posted__year=today.year, date_posted__month=today.month, date_posted__day=today.day)
            if counter>0:
                answerValue="None"
                if answer==0:
                    answerValue="Negative"
                else:
                    answerValue="Positive"

            return answerValue, posOne
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

def predictionRange(request):
    predictionRangely(request) 
    if request.method=='POST':
        answer=answerValue
        messages.success(request,'The sentiment value for the selected range is:',extra_tags=answer)
    return render(request, 'sentiment_analysis/dashboard.html')
  
@csrf_exempt  
def predictionRangely(request):
        if request.GET:
            date_min=request.GET['startdate']
            date_max=request.GET['enddate']
        try:
            classifier = joblib.load('sent_model/model_src/sent_model/svmClassifier.pkl')

            global posOne
            global negTwo
            posOne=0
            negTwo=0
            counter=0
            sentiment_iterator = Post.objects.values_list('sentiment').filter(date_posted__gte=date_min).filter(date_posted__lte=date_max).iterator()
            for i in sentiment_iterator:
                comment=str(i)
                counter=counter+1
                answerQ=predict(comment, classifier)
                if answerQ==0:
                    negTwo=negTwo+1
                else:
                    posOne=posOne+1

            context = {
                 'posts': Post.objects.values_list('sentiment').filter(date_posted__gte=date_min).filter(date_posted__lte=date_max).iterator()
            }

            comment=str(context)
            global answerValue
            commentArray=comment
            answer=predict(comment, classifier)
            sentiment_iterator = Post.objects.values_list('sentiment').filter(date_posted__gte=date_min).filter(date_posted__lte=date_max).iterator()
            if counter>0:
                answerValue="None"
                if answer==0:
                    answerValue="Negative"
                else:
                    answerValue="Positive"

            return answerValue
        except ValueError as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)


def population_chart(request):
    labels = ['Positive', 'Negative']
    data = [posOne, negTwo]
    return JsonResponse(data={
        'labels': labels,
        'data': data,
    })
