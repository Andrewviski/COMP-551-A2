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

    training = [u' ', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r',
     u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'\x7f', u'\xbf', u'\xd7', u'\xdf', u'\xe0', u'\xe1', u'\xe2',
     u'\xe3', u'\xe4', u'\xe5', u'\xe6', u'\xe7', u'\xe8', u'\xe9', u'\xea', u'\xeb', u'\xec', u'\xed', u'\xee',
     u'\xef', u'\xf0', u'\xf1', u'\xf2', u'\xf3', u'\xf4', u'\xf5', u'\xf6', u'\xf7', u'\xf8', u'\xf9', u'\xfa',
     u'\xfb', u'\xfc', u'\xfd', u'\xff', u'\u0105', u'\u0107', u'\u010d', u'\u010f', u'\u0115', u'\u0117', u'\u0119',
     u'\u011b', u'\u011f', u'\u0131', u'\u013a', u'\u013e', u'\u0142', u'\u0144', u'\u0148', u'\u014d', u'\u0151',
     u'\u0153', u'\u0155', u'\u0159', u'\u015b', u'\u015d', u'\u015f', u'\u0161', u'\u0165', u'\u016b', u'\u016f',
     u'\u0171', u'\u017a', u'\u017c', u'\u017e', u'\u0192', u'\u024d', u'\u0300', u'\u0301', u'\u0303', u'\u031b',
     u'\u0327', u'\u032f', u'\u0335', u'\u0336', u'\u033f', u'\u0340', u'\u035c', u'\u035d', u'\u0361', u'\u03b4',
     u'\u03c0', u'\u03c3', u'\u03c9', u'\u0431', u'\u0433', u'\u0434', u'\u044d', u'\u0489', u'\u200b', u'\u24de',
     u'\ufe0f', u'\ufeff', u'\uff41', u'\uff43', u'\uff44', u'\uff45', u'\uff47', u'\uff48', u'\uff49', u'\uff4c',
     u'\uff4e', u'\uff4f', u'\uff50', u'\uff52', u'\uff53', u'\uff55', u'\U0001d400', u'\U0001d404', u'\U0001d411',
     u'\U0001d419', u'\U0001d510', u'\U0001d51e', u'\U0001d522', u'\U0001d526', u'\U0001d52b', u'\U0001d52f',
     u'\U0001d613', u'\U0001d622', u'\U0001d624', u'\U0001d625', u'\U0001d626', u'\U0001d627', u'\U0001d629',
     u'\U0001d62a', u'\U0001d62d', u'\U0001d633', u'\U0001d634', u'\U0001d635', u'\U0001d636', u'\U0001d639']

    if tfidf:
        tfidf_vect = TfidfVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
        X = tfidf_vect.fit_transform(data)
        # for x in set(tfidf_vect.get_feature_names()) - set(training):
        #     print x
        for x in set(training) - set(tfidf_vect.get_feature_names()):
            print x
    else:
        count_vect = CountVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
        X = count_vect.fit_transform(data)

    return X

if __name__ == "__main__":
    X,y = init()

    test_data = process_test_set()

    estimator = LogisticRegression(solver='sag',multi_class='multinomial')
    result = estimator.fit(X,y).predict(test_data)

    with open('logistic_regression.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(result)

    forest = RandomForestClassifier(n_estimators=100, random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    result = multi_target_forest.fit(X, y).predict(test_data)

    with open('logistic_regression.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(result)


