#!/usr/bin/env python3
# -*- coding: utf-8 -*
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss, roc_curve
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin



class Preprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        stemmer = nltk.stem.RSLPStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        return 

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        def tokenize(X):
            corpus = []
            for tweet in X:
              review = re.sub(r"@[A-Za-z0-9_]+", " ", tweet)
              review = re.sub('RT', ' ', review)
              review = re.sub(r"https?://[A-Za-z0-9./]+", " ", review)
              review = re.sub(r"https?", " ", review)
              review = re.sub('[^a-zA-Z]', ' ', review)
              review = review.lower()
              review = review.split()
              ps = PorterStemmer()
              review = [ps.stem(word) for word in review if not word in set(all_stopwords) if len(word) > 2]
              review = ' '.join(review)
              corpus.append(review)
            return np.array(corpus)
        return X 
