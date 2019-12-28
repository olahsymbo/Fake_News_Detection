# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:59:22 2019

@author: olahs
"""
  
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
#nltk.download()
from nltk.corpus import stopwords  
from wordcloud import WordCloud,STOPWORDS  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB 
from sklearn.ensemble import RandomForestClassifier  
from sklearn import preprocessing
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 

##### Get the data sets from Positive, Negative Tweets #############
df = pd.read_csv(r"C:\Users\olahs\Documents\Python_scripts\Fake_News_Detection-master\train.csv", encoding = 'iso-8859-1')

data_train = df 


df1 = pd.read_csv(r"C:\Users\olahs\Documents\Python_scripts\Fake_News_Detection-master\test.csv", encoding = 'iso-8859-1')

data_test = df1

## Splitting the dataset into train and test set
#train, test = train_test_split(data,test_size = 0.3)

target_train = data_train['Label'] 
target_test = data_test['Label'] 
number = preprocessing.LabelEncoder() 
target_train = number.fit_transform(target_train.astype(str))
target_test = number.fit_transform(target_test.astype(str))


####### Get the data for positive, negative, neutral jsut for plotting word cloud
data_pos = data_train[ target_train == 1]  # find data that have positive tweets
data_pos = data_pos['Statement']
data_neg = data_train[ target_train == 0] # find data that have negative tweets
data_neg = data_neg['Statement']

### define the function for removing @ # RT from tweets
def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@') 
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    #### initialize word cloud by using the clean words 
    wordcloud = WordCloud(stopwords=STOPWORDS,             
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(5, 5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

#### View the word cloud for true and false new from training set
#print("True Words")
#wordcloud_draw(data_pos,'white')
#print("False words")
#wordcloud_draw(data_neg)

#########################################################################
#########################################################################
#tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
#train_features = tfidf.fit_transform(data_train['Statement'] ) 
#print(train_features.shape)
#test_features = tfidf.fit_transform(data_test['Statement'] ) 
#print(test_features.shape)

vectorizer = CountVectorizer(max_features=4000, ngram_range=(1, 5), stop_words = 'english')
train_features = vectorizer.fit_transform(data_train['Statement'])
print(train_features.shape)
test_features = vectorizer.fit_transform(data_test['Statement'] ) 
print(test_features.shape)
#print(vectorizer.get_feature_names())
#########################################################
#########################################################
###### Prediction using Naive Bayes Classifier ##########
#########################################################
#########################################################
 
gnb = BernoulliNB()
gnb.fit(train_features, target_train)

y_pred = gnb.predict(test_features)  

print("Test Accuracy of Naive Bayes Classifier :: ", accuracy_score(target_test, y_pred)) 


#########################################################
#########################################################
###### Prediction using Random Forest Classifier ########
#########################################################
#########################################################
 ## fit random forest classifier
clfr = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=10, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=70)

clfr.fit(train_features, target_train)

## predict random forest classifier
predictions = clfr.predict(test_features)

print("Test Accuracy of Random Forest Classifier :: ", accuracy_score(target_test, predictions))



############################################################################
############## Now Let's Perform Independent Test of some News ###########
##############                                           ###################
############## Prediction examples for a True News ####################
print("Positive (1) News Prediction")

Y = vectorizer.transform(["State revenue projections have missed the mark month after month."])
prediction = gnb.predict(Y)
print(prediction)


############## Prediction examples for a Negative Tweet  ####################
print("Negative (0) News Prediction")

Yn = vectorizer.transform(["Mark Sharpe has lowered property taxes by 17 percent."])
prediction = gnb.predict(Yn)
print(prediction)



############################################################################
############################################################################
############ BUILD DEEP NEURAL NETWORK (DNN) ###############################
############################################################################

from keras.models import Sequential
from keras.layers import Dense, Dropout 
import numpy
# fix random seed for reproducibility
numpy.random.seed(7000)


# create model
model = Sequential()
model.add(Dense(100, input_dim=4000, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))  
#model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#embedding_dim = 50
#vocab_size = 200
#maxlen = 400
## create model
#model = Sequential()
#model.add(layers.Embedding(input_dim=vocab_size, 
#                           output_dim=embedding_dim, 
#                           input_length=maxlen))
#model.add(layers.Flatten())
#model.add(Dense(200, input_dim=400, activation='relu'))
#model.add(Dense(150, activation='relu'))
#model.add(Dense(100, activation='relu'))  
##model.add(Dropout(0.2)) 
#model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_features, target_train, epochs=50, batch_size=200)
# evaluate the model
scores = model.evaluate(test_features, target_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 


############################################################################
############## Now Let's Perform Independent Test of some News ###########
##############                 DEEP NEURAL NETWORK    ###################
############## Prediction examples for a True News ####################
print("Positive (1) News Prediction")

Y = vectorizer.transform(["State revenue projections have missed the mark month after month."])
prediction = model.predict(Y)
print(prediction)


############## Prediction examples for a Negative Tweet  ####################
print("Negative (0) News Prediction")

Yn = vectorizer.transform(["Mark Sharpe has lowered property taxes by 17 percent."])
prediction = model.predict(Yn)
print(prediction)



