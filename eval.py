from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import numpy as np

from preprocess.vectorize import init
from linear.NB import NaiveBayes
# .... import a bunch of models here...


def PCA(X, n_component):
    C = np.cov(X.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(C)
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_vec = [p[1] for p in eig_pairs] ## sorted eigen_vec
    V = np.stack(eig_vec,axis=1)
	w = X.dot(V[:n_component].T)
	return w


def evaluate(models):
    ## evaluate a list of models
    X,y = init()
    X, y = shuffle(X,y)
    for model in models:
        scores = cross_val_score(model, X, y, cv = 10)
        print(np.mean(scores))


if __name__ == "__main__":
    clf = [NaiveBayes(smoothing = 1)]
    evaluate(clf)
