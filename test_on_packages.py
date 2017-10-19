# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
import numpy as np
import csv
# .... import a bunch of models here...
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


def PCA(X, n_component):
    C = np.cov(X.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(C)
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_vec = [p[1] for p in eig_pairs] ## sorted eigen_vec
    V = np.stack(eig_vec,axis=1)
    w = X.dot(V[:n_component].T)
    return w


def init():
    X = np.load("preprocess/X.npy")
    y = np.load("preprocess/Y.npy").astype(int)
    # print(X.shape)
    # print(y.shape)
    X, y = shuffle(X,y.reshape((-1,)))
    return X, y

def process_test_set(ngram_range = (1,1), max_features=5000, analyzer="char_wb", tfidf=True):
    f = open('./data/test_set_x.csv', 'r')
    reader = csv.reader(f)
    data = [row[1].decode('latin-1').encode("utf-8").translate(None, " \n") for row in reader]
    f.close()

    if tfidf:
        tfidf_vect = TfidfVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
        X = tfidf_vect.fit_transform(data)
        for x in tfidf_vect.get_feature_names():
            print x
    else:
        count_vect = CountVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
        X = count_vect.fit_transform(data)
    print X.shape

    return X

if __name__ == "__main__":
    X,y = init()

    test_data = process_test_set()
    #
    # estimator = LogisticRegression(solver='sag',multi_class='multinomial')
    # result = estimator.fit(X,y).predict(test_data)

    # with open('logistic_regression.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(result)
    #
    # forest = RandomForestClassifier(n_estimators=100, random_state=1)
    # multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    # result = multi_target_forest.fit(X, y).predict(test_data)
    #
    # with open('logistic_regression.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(result)


