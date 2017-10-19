#!/usr/bin/env python3
import sys
import os
import random
import math
from pprint import pprint
import numpy as np
import pickle

max_depth = 100
min_size = 10
fileName = "decisionTree.txt"


def split(X, Y, feature):
    buckets = {}

    for i in range(0, len(X)):
        idx = int(X[i][feature])
        if not idx in buckets:
            buckets[idx] = {'X': [], 'Y': []}
        buckets[idx]['X'].append(X[i])
        buckets[idx]['Y'].append(Y[i])
    return (buckets)



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
    splits = split(X, Y, feature).values()
    sz = len(X)
    score = 0
    for subset in splits:
        score += (len(subset['X']) / float(sz)) * entropy(subset['Y'])
    return entropy(Y) - score


def get_best_feature(X, Y, done):
    best_gain = -np.inf
    best_feature = -1;
    l = len(X[0]);
    for feature in range(0, l):
        if not feature in done:
            gain, f = (information_gain(feature, X, Y), feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = f
    return (best_gain, best_feature)


def adjust(X, precision):
    return (np.multiply([10 ^ precision], X)).astype(int)


def pick_majority(Y):
    classes = [v for v in Y]
    return max(set(classes), key=classes.count)


class ID3():
    def __init__(self):
        self.children = []

        self.isLeaf = False
        self.partitionFeature = -1
        self.partitionValue = -1
        self.Predication = -1

    def fit(self, X, Y):
        self.__fit(np.array(adjust(X, 5)), np.array(Y), 0, set([]))
        pickle.dump(self.children, open(fileName, "w"))

    def __fit(self, X, Y, depth, done):
        nclasses = len(set(_ for _ in Y))
        if nclasses < 2 or depth > max_depth or len(X) <= min_size or len(done) == X.shape[1]:
            self.isLeaf = True
            self.predication = pick_majority(Y)
        else:
            best = get_best_feature(X, Y, done)
            subsets = split(X, Y, best[1])
            done.add(best[1])
            print("splitting on the %dth feature" % best[1])
            for splitValue in subsets:
                subset = subsets[splitValue]
                t = ID3()
                t.__fit(np.array(subset['X']), np.array(subset['Y']), depth + 1, done)
                t.partitionFeature = int(best[1])
                t.partitionValue = splitValue
                ##print("With value %s"%splitValue)
                self.children.append(t)

    def predict(self, X):
        yp = []
        X = adjust(X, 5)
        for test in X:
            yp.append(self.predict_row(np.array(test)))
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("[*] Reading Training data...")
        data = []
        for line in open(sys.argv[1]):
            tempRow = []
            for value in line.split(","):
                value = value.strip()
                if value != "":
                    if not value.isdigit():
                        tempRow.append(0)
                    else:
                        tempRow.append(value)
            data.append(tempRow)

        print("[*] Shuffling data...")
        random.shuffle(data)
        # pprint(data)

        Y = np.array([row[-1] for row in data if len(row) > 0])
        X = np.array([row[:-2] for row in data if len(row) > 0])
        # pprint(split(X,Y,get_best_attribute(X,Y)[1]))
        tree = ID3()
        print("[*] Training tree...")
        tree.fit(X, Y)
        # tree.print_tree()

        print("[*] Testing...")
        correct = 0
        total = 0
        for line in open(sys.argv[2]):
            row = [_ for _ in line.strip().split(",")]
            pred = tree.predict(np.array(row[:-2]))
            print("ID3 predict %s, actual class is %s" % (pred, row[-1]))
            if pred == row[-1]:
                correct += 1
            total += 1
        print('Accuracy= %f%%' % (correct * 100 / float(total)))

    else:
        print("Usage:\n./ID3 [TrainData].csv [TestData].csv {Max_depth} {Min_size}")
