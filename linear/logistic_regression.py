import numpy as np
from math import exp
from numpy.linalg import norm
import sys

class LogisticRegression():

    def __init__(self, alpha = 0.05, l2 = 0.1, tol=0.0001):
        self.alpha = alpha
        self.l2 = l2
        self.tol = tol

    def convert_one_hot(self, Y):
        one_hot_Y = np.zeros((len(Y), self.label_num))
        one_hot_Y[np.arange(len(Y)), Y] = 1
        return one_hot_Y

    def softmax(self, x):
        e = np.exp(x - np.max(x))  # prevent overflow
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

    def fit(self, X, Y):
        # n samples, m features
        n, m = X.shape
        self.label_num = len(set(Y))
        k = self.label_num

        # converting labels on one-hot encoding
        one_hot_Y = self.convert_one_hot(Y)

        # Initializing weights and bias
        self.theta = np.zeros((m, k))
        self.bias = np.ones(k)
        improve = sys.maxint
        while norm(improve) > self.tol:

            p_y_given_x = self.softmax(np.dot(X, self.theta) + np.array([self.bias,] * n))
            distance = one_hot_Y - p_y_given_x
            improve = self.alpha * np.dot(X.T, distance) - self.alpha * self.l2 * self.theta
            print norm(improve)
            self.theta += improve

            self.bias += self.alpha * np.mean(distance, axis=0)

    def predict(self, X):
        assert X.shape[1] == self.theta.shape[0]
        result = self.softmax(np.dot(X, self.theta) + np.array([self.bias,] * X.shape[0])).argmax(axis=1)
        return result



if __name__ == "__main__":
    ## dummy test
    X = np.array([[1,0,1],[0,2,1/2],[3,1/3,3],[0,2,4],[1,2,5/2],[1,3,4]])
    Y = np.array([0,2,1,0,2,3])

    lr = LogisticRegression()
    lr.fit(X,Y)
    x = np.array([[0,3,3], [1,0,1],[0,2,1/2],[0,3,1]])
    yp = lr.predict(x)

