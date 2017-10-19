from nonlinear.ID3 import ID3
from nonlinear.ID3 import pick_majority
import random
import numpy as np

class random_forest():

    def __init__(self,size):
        self.size=size
        self.trees=[ID3() for _ in range(size)]

    def fit(self,X,Y):
        for i in range(self.size):
            Xs=[]
            Ys=[]
            taken=set([])
            for _ in range(len(X[0])/self.size):
                idx=random.randint(0,len(X))
                while idx in taken:
                    idx = random.randint(0, len(X))
                taken.add(idx)
                Xs.append(X[idx])
                Ys.append(Y[idx])
            self.trees[i].fit(Xs,Ys)

    def predict(self,X):
        Yp = []
        for test in X:
            temp=[]
            for tree in self.trees:
                temp.append(tree.predict_row(test))
            print(temp)
            Yp.append(pick_majority(temp))
        return np.array(Yp)
