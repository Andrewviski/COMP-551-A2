import sys
import scipy as sp

# allow deep recursion needed by kdtrees
sys.setrecursionlimit(1000000)

# KNN implementation using KDTrees
class KNN_KDTrees():
    def __init__(self, k):
        self.k = k

    # predication method for the classifier
    def fit(self, X, Y):
        data = []

        # convert the data to the right format for KDTree
        for i in range(len(X)):
            data.append(tuple(X[i]))

        # preprocess the data
        self.KDtree = sp.spatial.KDTree(data, leafsize=20)

        # assign data to initialize the algorithm
        self.X = X
        self.y = Y

    # predication method for the classifier
    def predict(self, X):
        y = []

        # get k nearest neighbours
        results = self.KDtree.query(X, k=self.k)
        for result in results:
            points = []
            for i in result[1]:
                points.append(self.y[i])

            # classify based on majority on nearest points
            res = max(set(points), key=points.count)
            y.append(res)
        return y
