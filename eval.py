from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from linear.NB import *
import csv

# .... import a bunch of models here...


def init():
    X = np.load("Manual_train_X.npy")
    y = np.load("Train_Y.npy")
    X, y = shuffle(X,y)
    return X, y


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
    X,y = init()

    kf = KFold(n_splits=10)
    for model in models:
        acc_avg = []
        f1_0_avg = []
        f1_1_avg = []
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            model.fit(X_train, y_train.reshape((-1,)))
            yp_valid = model.predict(X_valid)
            precision, recall, f1, _ = precision_recall_fscore_support(y_valid, yp_valid)
            acc = accuracy_score(y_valid, yp_valid)
            print(confusion_matrix(y_valid, yp_valid))
            acc_avg.append(acc)
            f1_0_avg.append(f1[0])
            f1_1_avg.append(f1[1])


def predict_on_testset(model, X, y, filename):
    model.fit(X, y.reshape((-1,)))
    test_data = np.load("Manual_test_X.npy")
    yp = model.predict(test_data)
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('Id', 'Category'))
        for i in range(len(yp)):
            writer.writerow((i, yp[i]))
   

if __name__ == "__main__":
    X, y = init()
    # Import estimators from subfolders "linear" or "nonlinear",
    # construct instances and put them into the "clf" list.
    clf = [NaiveBayes2(smoothing=1)]
    # To test cross-validation scores, run evaluate(clf)
    # which will print the precision for each fold and     evaluate(clf)

