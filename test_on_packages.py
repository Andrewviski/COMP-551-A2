# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np
import csv

# .... import a bunch of models here...
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

def pca(X, n_component):
    p = PCA(n_components=n_component,svd_solver= 'arpack')
    return p.fit_transform(X)


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
    X_val = np.load("Val_X.npy")
    y_val = np.load("Val_Y.npy")

    print("Train X.shape ",X.shape,"valid X.shape",X_val.shape,"Test X.shape ",test_data.shape)

    # X = pca(X,50)
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
    # forest.fit(X,y)
    # print("Random forest score",forest.score(X_val,y_val))
    # yp = forest.predict(test_data).astype(int)

    ## Decision Tree Sklearn
    # clf = DecisionTreeClassifier()
    # clf.fit(X,y)
    # print("Decision Tree score",clf.score(X_val,y_val))


    ## Naive Bayes Sklearn
    clf = MultinomialNB(alpha=1)
    clf.fit(X,y)
    print("MultinomialNB score",clf.score(X_val, y_val))
    yp = clf.predict(test_data).astype(int)
    

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

