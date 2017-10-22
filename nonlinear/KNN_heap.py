import numpy as np
import heapq
from pprint import pprint
import sys
sys.setrecursionlimit(1000000)
import scipy as sp

class KNN_KDTrees():
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        data=[]
        for i in range(len(X)):
            # temprow=np.append(X[i])
            data.append(tuple(X[i]))
        self.KDtree=sp.spatial.KDTree(data,leafsize=20)
        pprint(len(self.KDtree.data))
        self.X = X
        self.y = Y

    def predict(self, X):
        y = []
        for x in X:
            # print "Querying KD Tree"
            result=self.KDtree.query(x,k=self.k)
            dists=result[0]
            # print "Getting points"
            points=[]
            for i in result[1]:
                points.append(self.y[i])
            # print "Getting best class"
            res = max(set(points), key=points.count)
            # print "Predicted class %d" %res
            y.append(res)
        return y