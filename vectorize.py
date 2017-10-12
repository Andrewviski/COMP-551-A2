# -*- coding: utf-8 -*-
import os, sys, csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def init(ngram_range = (1,2), max_features=5000, analyzer="char_wb"):
    f = open("punctuation_removed.txt", "r")
    data = f.readlines()
    f.close()
    data = data[1:]

    f = open('data/train_set_y.csv', 'r')
    reader = csv.reader(f)
    label = [row[1] for row in reader]
    label = label[1:]

    count_vect = CountVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
    X = count_vect.fit_transform(data)

    print X.shape
    print label.shape
    return X, label

init()
