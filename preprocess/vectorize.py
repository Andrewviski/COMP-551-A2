# -*- coding: utf-8 -*-
import os, sys, csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import parameters as p


def create_data(ngram_range = (1,1), max_features=5000, analyzer="char_wb", tfidf=True):
    f = open("punctuation_removed.txt", "r")
    data = f.readlines()
    f.close()
    data = data[1:]

    f = open('../data/train_set_y.csv', 'r')
    reader = csv.reader(f)
    label = [row[1] for row in reader]
    label = label[1:]
    label = np.array(label).reshape((-1,1))
    
    

    if tfidf:
        tfidf_vect = TfidfVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
        X = tfidf_vect.fit_transform(data)
    else:
        count_vect = CountVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
        X = count_vect.fit_transform(data)

    # Zipping the data and labels

    # zipped = zip(data, label)
    # for s in zipped:
    #     try:
    #         print(s[1].strip('\n') + " " + s[0])
    #     except:
    #         print(s[1])

    # print X.shape
    # print label.shape
    # print count_vect.get_feature_names()

    # save arrays to .npy files
    np.savez("data",*[X, label])
    return X, label

if __name__ == "__main__":

    create_data(p.ngram, p.max_features, p.analyzerm, p.tfidf)