import numpy as np
from math import exp
from numpy.linalg import inv
import sys

class LogisticRegression():

    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def convert_one_hot(self, Y):
        one_hot_Y = np.zeros((len(Y), self.label_num))
        one_hot_Y[np.arange(len(Y)), Y] = 1
        return one_hot_Y

    def fit(self, X, Y):
        # n samples, m features
        n, m = X.shape
        self.label_num = len(set(Y))
        # converting labels on one-hot encoding
        one_hot_Y = self.convert_one_hot(Y)
        self.theta = np.zeros((self.label_num, m))

        for k in range(self.label_num):
            # for example i
            for i in range(n):
                # for feature j
                for j in range(m):

                        # print X[i]
                        # print self.theta
                        z = X[i].dot(self.theta[k])
                        predicted = 1 + exp(-z)
                        derivative = (one_hot_Y[i][k] - predicted) * X[i][j] 
                        self.theta[k][j] += self.alpha * derivative

    def predict(self, X):
        
        assert X.shape[1] == len(self.theta[0])
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            maximum = 0
            max_index = 0
            for j in range(self.label_num):
                z = self.theta[j].dot(X[i])
                predicted = 1 + exp(-z)
                if predicted > maximum:
                    maximum = predicted
                    max_index = j
            result[i] = max_index
        print result
        return result
        


if __name__ == "__main__":
    ## dummy test
    X = np.array([[1,0,1],[0,2,1/2],[3,1/3,3],[0,2,4],[1,2,5/2],[1,3,4]])
    Y = np.array([0,2,1,0,2,2])

    lr = LogisticRegression()
    lr.fit(X,Y)
    x = np.array([[0,3,3], [1,0,1],[0,2,1/2]])
    yp = lr.predict(x)

