import pandas as pd
import numpy as np

data = pd.read_csv('spam.csv')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1, inplace = True)
data['label'] = data['class'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']

import nltk
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

import pickle
pickle.dump(cv,open('transform.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state = 1)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

accu = accuracy_score(y_pred,y_test)

pickle.dump(clf,open('final_model.pkl','wb'))