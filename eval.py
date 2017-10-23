from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support
import numpy as np

from linear.NB import *
from nonlinear.ID3 import ID3
from nonlinear.random_forest import random_forest
from linear.logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv

from nonlinear.KNN import KNN, KNNFast
from nonlinear.KNN_heap import KNN_KDTrees
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
# .... import a bunch of models here...

def init():
    X = np.load("Manual_train_X.npy")
    y = np.load("Train_Y.npy")
    X, y = shuffle(X,y)
    return X, y

# def PCA(X, n_component):
#     C = np.cov(X.T)
#     eig_val_cov, eig_vec_cov = np.linalg.eig(C)
#     eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
#     eig_pairs.sort(key=lambda x: x[0], reverse=True)
#     eig_vec = [p[1] for p in eig_pairs] ## sorted eigen_vec
#     V = np.stack(eig_vec,axis=1)
#     w = X.dot(V[:n_component].T)
#     return w


def evaluate(models):
    X,y = init()
    test_data = np.load("Manual_test_X.npy")

    # pca = PCA(n_components=15)
    # X = pca.fit_transform(X)
    # test_data = pca.fit_transform(test_data)
    # print(X.shape)

    acc_scorer = make_scorer(accuracy_score)
    yps = []
    kf = KFold(n_splits=10)
    for model in models:
        acc_avg = []
        f1_0_avg = []
        f1_1_avg = []
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            # print(X_train.shape, y_train.shape)
            model.fit(X_train, y_train.reshape((-1,)))
            yp_valid = model.predict(X_valid)
            precision, recall, f1, _ = precision_recall_fscore_support(y_valid, yp_valid)
            acc = accuracy_score(y_valid, yp_valid)
            print("acc",acc)
            acc_avg.append(acc)
            f1_0_avg.append(f1[0])
            f1_1_avg.append(f1[1])

        # print("accuracy mean",np.mean(np.array(acc_avg)))

        model.fit(X,y.reshape((-1,)))
        
        yp = model.predict(test_data)
        with open('MultinomialNB.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('Id', 'Category'))
            for i in range(len(yp)):
                writer.writerow((i, yp[i]))
        # yps.append(yp)
   

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.tree import DecisionTreeClassifier
    clf = [NaiveBayes2(smoothing=1)]
    # clf = [random_forest(600)]
    # clf = [DecisionTreeClassifier()]
    # evaluate(clf)
    # process_test_set()
    # clf = [KNN_KDTrees(k=5)]
    evaluate(clf)
