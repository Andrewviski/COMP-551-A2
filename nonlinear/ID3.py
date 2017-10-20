#!/usr/bin/env python3
import sys
import os
import random
import math
from pprint import pprint
import numpy as np
import pickle
import copy
max_depth = 100
min_size = 10
fileName = "decisionTree.txt"


def split(X, Y, feature):
    bucketX = {}
    bucketY = {}
    #print "Trying feature %d" % feature
    for i in range(len(X)):
        idx = X[i][feature]

        if not idx in bucketX:
            bucketX[idx] =[]

        if not idx in bucketY:
            bucketY[idx] =[]

        bucketX[idx].append(X[i])
        bucketY[idx].append(Y[i])

    return (np.array(bucketX.values()),np.array(bucketY.values()))



def entropy(Y):
    total_size = len(Y)
    if total_size == 0:
        print("shiyaat")
        exit(-1)
    class_size = {}
    for e in Y:
        if e in class_size:
            class_size[e] += 1
        else:
            class_size[e] = 1

    res = 0
    for subset_size in class_size.values():
        res += ((-subset_size / float(total_size)) * np.log2(subset_size / float(total_size)))
    return res


def information_gain(feature, X, Y):
    (Xs,Ys) = split(X, Y, feature)
    sz = len(X)
    score = 0
    for i in range(len(Xs)):
        score += (len(Xs[i]) / float(sz)) * entropy(Ys[i])
    return entropy(Y) - score


def get_best_feature(X, Y, done):
    best_gain = -np.inf
    best_feature = -1
    l = len(X[0])
    for feature in range(0,l):
        if not feature in done:
            gain, f = (information_gain(feature, X, Y), feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = f
    return best_feature


def adjust(X, precision):
    return (np.multiply([10 ^ precision], X)).astype(int)

def pick_majority(classes):
    return max(set(classes), key=classes.count)

class ID3():
    def __init__(self):
        self.children = []

        self.isLeaf = False
        self.partitionFeature = -1
        self.partitionValue = -1
        self.Predication = -1

    def fit(self, X, Y):
        print "training on %d features with %d samples"% (len(X[0]),len(X))
        self.__fit(np.array(adjust(X, 5)), np.array(Y), 0, [])
        pickle.dump(self.children, open(fileName, "w"))

    def __fit(self, X, Y, depth, done):
        classes=[_ for _ in Y]
        if len(set(classes)) < 2 or depth > max_depth or len(X) <= min_size or len(done) >= len(X[0]):
            self.isLeaf = True
            self.predication = pick_majority(classes)
        else:
            best = get_best_feature(X, Y, done)
            (Xs,Ys) = split(X, Y, best)
            print("splitting on the %dth feature" % best)
            for i in range(len(Xs)):
                t = ID3()
                temp=copy.copy(done)
                temp.append(best)
                t.__fit(Xs[i], Ys[i], depth + 1, temp)
                t.partitionFeature = best
                t.partitionValue = X[i][best]
                self.children.append(t)

    def predict(self, X):
        yp = []
        X = adjust(X, 5)
        for test in X:
            yp.append(self.predict_row(np.array(test)))
        print yp
        return yp

    def predict_row(self, X):
        if self.isLeaf == True:
            return self.predication
        else:
            for child in self.children:
                if X[child.partitionFeature] == child.partitionValue:
                    return child.predict_row(X)
            mn = np.inf
            bestChild = None
            for child in self.children:
                if np.abs(X[child.partitionFeature] - child.partitionValue) < mn:
                    mn = np.abs(X[child.partitionFeature] - child.partitionValue)
                    bestChild = child
            return bestChild.predict_row(X)

    def __print_tree(self, depth=0):
        if not self.isLeaf:
            print('%s[X%d < %s]' % (depth * ' ', (self.partitionFeature + 1), self.partitionValue))
            for child in self.children:
                child.__print_tree(depth + 1)
        else:
            print('%s[%s]' % (depth * ' ', str(self.predication)))

    def print_tree(self):
        self.__print_tree(0)
