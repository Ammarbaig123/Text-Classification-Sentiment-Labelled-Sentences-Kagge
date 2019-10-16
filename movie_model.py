# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:57:37 2019

@author: Ammar Baig
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV


filepath_dict = {'yelp':   'D:/FYP/sentiment-labelled-sentences-data-set/sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': 'D:/FYP/sentiment-labelled-sentences-data-set/sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb':   'D:/FYP/sentiment-labelled-sentences-data-set/sentiment labelled sentences/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t',engine="python",error_bad_lines=False, warn_bad_lines=False )
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)
df = pd.concat(df_list)
df.iloc[0]
print(df.head())
print(df.tail())
df.head(n=3) 
df['label'].hist(bins=100)


for source in df['source'].unique():
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('Accuracy for {} data: {:.4f}'.format(source, score))
