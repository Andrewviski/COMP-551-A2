# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

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
    test_data = np.load("Test_X.npy")

    # print (float)(np.count_nonzero(test_data)) / (test_data.shape[0] * test_data.shape[1])

    ## Logistic Regression Sklearn
    # predicted_log_reg = model_selection.cross_val_predict(LogisticRegression(solver='sag',multi_class='multinomial'), X, y, cv=10) # this is really slow
    # print("Logistic Regression:     " + str(precision_score(y, predicted_log_reg, average='weighted')))
    # estimator = LogisticRegression(solver='sag',multi_class='multinomial')
    # estimator.fit(X[:-10000],y[:-10000])
    # print("Logistic regression score",estimator.score(X[-10000:],y[-10000:]))
    # yp = estimator.predict(test_data).astype(int)
    # np.savetxt('logistic_regression.csv', yp, delimiter=",")

    ## Random Forest Sklearn
    # forest = RandomForestClassifier(n_estimators=100, random_state=1,criterion = "entropy")
    # forest.fit(X[:-10000],y[:-10000])
    # print("Random forest score",forest.score(X[-10000:],y[-10000:]))
    # yp = forest.predict(test_data).astype(int)

    ## Decision Tree Sklearn
    # clf = DecisionTreeClassifier()
    # clf.fit(X[:-10000],y[:-10000])
    # print("Decision Tree score",clf.score(X[-10000:],y[-10000:]))


    ## Naive Bayes Sklearn
    clf = MultinomialNB(alpha=1)
    clf.fit(X[:-10000],y[:-10000])
    print("MultinomialNB score",clf.score(X[-10000:],y[-10000:]))
    exit(0)
    # yp = clf.predict(test_data).astype(int)
    

    ## XGboost
    # num_round = 10
    # dtrain = xgb.DMatrix(X[:-10000], label=y[:-10000])
    # param = {'max_depth': 10, 'eta': 0.8, 'silent': 1, 'objective': "multi:softmax", 'eval_metric':'auc',"num_class":5}
    # bst = xgb.train(param, dtrain, num_round)

    # dtest = xgb.DMatrix(X[-10000:])
    # ypred = bst.predict(dtest)
    # print("XBG score",accuracy_score(y[-10000:],ypred))

    # dtest = xgb.DMatrix(test_data)
    # yp = bst.predict(dtest)

    f = open('MultinomialNB.csv', 'wt')
    try:
        writer = csv.writer(f)
        writer.writerow(('Id', 'Category'))
        for i in range(len(yp)):
            writer.writerow((i, yp[i]))
    finally:
        f.close()

