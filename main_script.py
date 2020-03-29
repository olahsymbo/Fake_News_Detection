# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:59:22 2019

@author: olahs
"""
import os
import inspect
app_path = inspect.getfile(inspect.currentframe())
directory = os.path.realpath(os.path.dirname(app_path))
import pandas as pd
#nltk.download()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import classifiers
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 

#### Get the data sets from Positive, Negative Tweets #####
data_train = pd.read_csv(os.path.join(directory, "train.csv"), encoding = 'iso-8859-1')
data_test = pd.read_csv(os.path.join(directory, "test.csv"), encoding = 'iso-8859-1')

## Splitting the dataset into train and test set
target_train = data_train['Label'] 
target_test = data_test['Label'] 
number = preprocessing.LabelEncoder() 
target_train = number.fit_transform(target_train.astype(str))
target_test = number.fit_transform(target_test.astype(str))

# Get the data for positive, negative, neutral
data_pos = data_train[ target_train == 1]  # find data that have positive tweets
data_pos = data_pos['Statement']
data_neg = data_train[ target_train == 0] # find data that have negative tweets
data_neg = data_neg['Statement']

# count vectorizer
vectorizer = CountVectorizer(max_features=4000, ngram_range=(1, 5), stop_words = 'english')
train_features = vectorizer.fit_transform(data_train['Statement'])
test_features = vectorizer.fit_transform(data_test['Statement'])
print(vectorizer.get_feature_names())

# instantiate a classifier object
classifier_obj = classifiers.classifiers(train_features, target_train, test_features)

# Prediction using Bayes Classifiers
bayes_model, y_pred = classifier_obj.bayesClassifier()
print("Test Accuracy of Naive Bayes Classifier :: ", accuracy_score(target_test, y_pred)) 

#Prediction using Random Forest Classifiers
random_model, predictions = classifier_obj.randomForestClassifier()
print("Test Accuracy of Random Forest Classifier :: ", accuracy_score(target_test, predictions))

#Prediction using Deep Neural Networks
deep_model, prediction = classifier_obj.deepNetworks()
print("Test Accuracy of Deep Neural Networks :: ", accuracy_score(target_test, predictions))
