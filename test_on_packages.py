# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
import numpy as np
import csv
# .... import a bunch of models here...
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    print test_data[:10]
    # print (float)(np.count_nonzero(test_data)) / (test_data.shape[0] * test_data.shape[1])


    predicted_log_reg = model_selection.cross_val_predict(LogisticRegression(solver='sag',multi_class='multinomial'), X, y, cv=10)
    print("Logistic Regression:     " + str(precision_score(y, predicted_log_reg, average='weighted')))

    estimator = LogisticRegression(solver='sag',multi_class='multinomial')
    result = estimator.fit(X,y).predict(test_data)

    with open('logistic_regression.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(result)

    forest = RandomForestClassifier(n_estimators=100, random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    result = multi_target_forest.fit(X, y).predict(test_data)

    with open('random_forest.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(result)


