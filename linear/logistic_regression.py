import numpy as np
from math import exp
from numpy.linalg import norm
import sys

class LogisticRegression():

    def __init__(self, alpha = 0.01, l2 = 0.1, tol=0.0000001):
        self.alpha = alpha
        self.l2 = l2
        self.tol = tol

    def convert_one_hot(self, Y):
        one_hot_Y = np.zeros((len(Y), self.label_num)).astype(int)
        one_hot_Y[np.arange(len(Y)), Y] = 1
        return one_hot_Y

    def softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]  # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]  # dito
        return e_x / div


    def fit(self, X, Y):
        # n samples, m features
        n, m = X.shape
        self.label_num = len(set(Y))
        k = self.label_num

        # converting labels on one-hot encoding
        one_hot_Y = self.convert_one_hot(Y)

        # Initializing weights and bias
        self.theta = np.random.rand(m, k)
        self.bias = np.zeros(k)
        improve = sys.maxint
        while norm(improve) > self.tol:

            p_y_given_x = self.softmax(np.dot(X, self.theta) + np.array([self.bias,] * n))

            distance = one_hot_Y - p_y_given_x
            # print (np.arange(one_hot_Y.shape[0]), one_hot_Y)
            first = p_y_given_x[np.arange(one_hot_Y.shape[0]), Y]
            print first
            cost = -np.mean(np.log(first))
            print cost
            improve = self.alpha * np.dot(X.T, distance) - self.alpha * self.l2 * self.theta
            self.theta += improve
            self.bias += self.alpha * np.mean(distance, axis=0)

    def predict(self, X):
        assert X.shape[1] == self.theta.shape[0]
        result = self.softmax(np.dot(X, self.theta) + np.array([self.bias,] * X.shape[0])).argmax(axis=1)
        print result
        return result


if __name__ == "__main__":
    ## dummy test
    X = np.array([[1,0,1],[0,2,1/2],[3,1/3,3],[0,2,4],[1,2,5/2],[1,3,4]])
    Y = np.array([0,2,1,0,2,3])

    lr = LogisticRegression()
    lr.fit(X,Y)
    x = np.array([[0,3,3], [1,0,1],[0,2,1/2],[0,3,1]])
    yp = lr.predict(X)

