# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
import numpy as np
import csv

# .... import a bunch of models here...
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import precision_score


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
    X = np.load("Train_X.npy")
    y = np.load("Train_Y.npy").astype(int)
    # print(X.shape)
    # print(y.shape)
    X, y = shuffle(X,y.reshape((-1,)))
    return X, y


if __name__ == "__main__":
    X,y = init()
    # print (float)(np.count_nonzero(X))/(X.shape[0] * X.shape[1])

    test_data = np.load("Test_X.npy")

    # print (float)(np.count_nonzero(test_data)) / (test_data.shape[0] * test_data.shape[1])

    ## Logistic Regression Sklearn
    # predicted_log_reg = model_selection.cross_val_predict(LogisticRegression(solver='sag',multi_class='multinomial'), X, y, cv=10) # this is really slow
    # print("Logistic Regression:     " + str(precision_score(y, predicted_log_reg, average='weighted')))

    # estimator = LogisticRegression(solver='sag',multi_class='multinomial')
    # estimator.fit(X,y)
    # yp = estimator.predict(test_data).astype(int)
    #
    # f = open('logistic_regression.csv', 'wt')
    # try:
    #     writer = csv.writer(f)
    #     writer.writerow(('Id', 'Category'))
    #     for i in range(len(yp)):
    #         writer.writerow((i, yp[i]))
    # finally:
    #     f.close()
    #
    # ## Random Forest Sklearn
    # forest = RandomForestClassifier(n_estimators=100, random_state=1,criterion = "entropy")
    # forest.fit(X,y)
    # yp = forest.predict(test_data).astype(int)
    # f = open('random_forest.csv', 'wt')
    # try:
    #     writer = csv.writer(f)
    #     writer.writerow(('Id', 'Category'))
    #     for i in range(len(yp)):
    #         writer.writerow((i, yp[i]))
    # finally:
    #     f.close()
    #
    estimator = MultinomialNB()
    estimator.fit(X,y)
    yp = estimator.predict(test_data).astype(int)

    f = open('Multinomial.csv', 'wt')
    try:
        writer = csv.writer(f)
        writer.writerow(('Id', 'Category'))
        for i in range(len(yp)):
            writer.writerow((i, yp[i]))
    finally:
        f.close()

