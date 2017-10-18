from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.utils import shuffle
import numpy as np
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


def init():
    X = np.load("preprocess/X.npy")
    y = np.load("preprocess/Y.npy").astype(int)
    # print(X.shape)
    # print(y.shape)
    X, y = shuffle(X,y.reshape((-1,)))
    return X, y


    

if __name__ == "__main__":
    # clf = [NaiveBayes(smoothing = 1)]
    # clf = [ID3()]
    X,y = init()

    predicted_log_reg = cross_val_predict(LogisticRegression(solver='sag',multi_class='multinomial'), X, y, cv=10)

    print("Logistic Regression:     " + str(precision_score(y, predicted_log_reg, average='weighted')))

    predicted_log_reg = cross_val_predict(LogisticRegression(solver='sag',multi_class='multinomial'), X, y, cv=10)

    