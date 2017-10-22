from nonlinear.RandomizedTree import randomized_tree
from nonlinear.ID3 import pick_majority
import random
import math
import numpy as np

class random_forest():

    def __init__(self,size):
        self.size=size
        self.trees=[randomized_tree() for _ in range(size)]

    def fit(self,X,Y):
        m=int(math.sqrt(len(X)))
        for i in range(self.size):
            taken=set([])
            subX=[]
            subY=[]
            for _ in range(m):
                idx=random.randint(0,len(X))
                while idx in taken:
                    idx = random.randint(0, len(X))

                subX.append(X[idx])
                subY.append(Y[idx])
            self.trees[i].fit(subX,subY)

    def predict(self,X):
        Yp = []
        for x in X:
            trees_results=[]
            for tree in self.trees:
                trees_results.append(tree.predict_row(x))
            Yp.append(pick_majority(trees_results))
        print len(Yp)
        return Yp
