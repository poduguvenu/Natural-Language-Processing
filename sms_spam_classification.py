# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:24:58 2021

@author: podug

Here is the detailed explanation of implementing a Spam classifier in python using 
Natural Language Processing. 

"""

# Importing Required Libraries
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# Importing the Data
messages = pd.read_csv('C:/Users/podug/Datasets/SmsSpamCollection/SMSSpamCollection', 
                       sep='\t', names=['label', 'message'])

# Data Cleaning And Preprocessing
# Initializing PorterStemmer()
ps = PorterStemmer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model using Niave Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
plot_confusion_matrix(model, X_test, y_test, 
                      display_labels=['ham', 'spam'], values_format='d')

# Using Lemmatization
lemm = WordNetLemmatizer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemm.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating TF-IDF model
tf_idf = TfidfVectorizer(max_features=5000)
X = tf_idf.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model using Niave Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(model, X_test, y_test, 
                      display_labels=['ham', 'spam'], values_format='d')













