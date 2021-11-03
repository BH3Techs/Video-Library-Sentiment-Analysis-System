"""Author: Howard Mabhugu
Name: training.py
Version: 1.0
Function: This module is used for defining, training
and testing the SVM classifier and then dumping it as a pickle
file which can be reused for sentiment prediction.
Input: The training datasets from various sources.
Output: The result of tesing(recall,precision, F1-score etc) and
the trained classifier dumped in svmClassifier.pkl
"""
# Import required libraries

import csv
import os
import re
import nltk
import scipy
import pandas as pd
import sklearn.metrics
import manage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Generating the Training and testing vectors

def getTrainingAndTestData():
        X = []
        y = []

        #Training data 1: Sentiment 140
        f=open(r'C:/Users/Bhugs Hardy/Downloads/trainingandtestdata/training.csv','r', encoding='ISO-8859-1')
        reader = csv.reader(f)

        for row in reader:
            X.append(row[5])
            y.append(1 if (row[0]=='4') else 0)

        
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.20, random_state=42)
        return X_train, X_test, y_train, y_test

#Process Sentiments (Stemming+Pre-processing)

def processSentiments(X_train, X_test):
        X_train = [manage.stem(manage.preprocessSentiments(comment)) for comment in X_train]
        X_test = [manage.stem(manage.preprocessSentiments(comment)) for comment in X_test]
        return X_train,X_test

# SVM classifier

def classifier(X_train,y_train):
        vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
        svm_clf =svm.LinearSVC(C=0.1)
        vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
        vec_clf.fit(X_train,y_train)
        # joblib.dump(vec_clf, 'svmClassifier.pkl', compress=3)
        return vec_clf

# Main function

def main():
        X_train, X_test, y_train, y_test = getTrainingAndTestData()
        X_train, X_test = processSentiments(X_train, X_test)
        vec_clf = classifier(X_train,y_train)
        y_pred = vec_clf.predict(X_test)
        print(sklearn.metrics.classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
