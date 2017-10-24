#!/usr/bin/env python3
import random
import math
import numpy as np
import pickle
from nonlinear.ID3 import ID3


min_gain = 0.02

gain_output = ""
gain_file = open("gainResults.csv", "w")
gain_idx = 0

# a modified ID3 tree that use randomization to pick split features
class randomized_tree(ID3):

    #get best features after
    def get_best_feature(self, X, Y, trials=50):
        best_gain = -np.inf
        best_feature = -1
        for _ in range(trials):
            feature = random.randint(0, len(X[0]) - 1)
            gain = self.information_gain(X, Y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return (best_feature, best_gain)

    # exactly same as fit in ID3, we need it declared here for OOP reseaons
    def fit(self, X, Y):
        self.__fit(np.array(X), np.array(Y), 0)

    # exactly same as __fit in ID3 except get_best_feature is randomized
    def __fit(self, X, Y, depth):
        classes = [_ for _ in Y]
        if len(set(classes)) < 2 or depth > self.max_depth or len(X) <= self.min_size:
            self.isLeaf = True
            self.predication = self.pick_majority(classes)
        else:
            best, gain = self.get_best_feature(X, Y, int(math.sqrt(len(X[0])))*3)
            gain_file.write(str(gain_idx) + "," + str(gain) + '\n')
            if gain < min_gain:
                self.isLeaf = True
                self.predication = self.pick_majority(classes)
            else:
                (Xs, Ys) = self.split(X, Y, best)
                for i in range(len(Xs)):
                    t = randomized_tree(self.max_depth, self.min_size,self.class_weights)
                    t.__fit(Xs[i], Ys[i], depth + 1)
                    t.partitionFeature = best
                    t.partitionValue = X[i][best]
                    self.children.append(t)

