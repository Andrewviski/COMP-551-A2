from nonlinear.RandomizedTree import randomized_tree
from nonlinear.ID3 import pick_majority
import random
import numpy as np

class random_forest():

    def __init__(self,size):
        self.size=size
        self.trees=[randomized_tree() for _ in range(size)]

    def fit(self,X,Y):
        for i in range(self.size):
            self.trees[i].fit(X,Y)

    def predict(self,X):
        Yp = []
        for test in X:
            trees_results=[]
            for tree in self.trees:
                trees_results.append(tree.predict_row(test))
            Yp.append(pick_majority(trees_results))
        return Yp
