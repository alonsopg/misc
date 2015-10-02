# -*- coding: utf-8 -*-

from Tkinter import Tk
from tkFileDialog import askopenfilename
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer


vectorizer= CountVectorizer(encoding='latin1', min_df=1)
countVect = CountVectorizer()
stemmer = SnowballStemmer('spanish')

Tk().withdraw


opinion = askopenfilename()

if opinion:
    with open(opinion) as opinion_file:

        X = vectorizer.fit_transform(opinion_file)

        print "These are the raw words ,which actually are the features: \n",
        feature_names = vectorizer.get_feature_names()
        print feature_names

        print "\n These are the stems of the text\n:",
        stemmed_features = [stemmer.stem(i) for
                        i in word_tokenize(feature_names)]

else:
   # user might select no file and hit cancel the file open dialog
   pass
