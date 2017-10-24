#!/usr/bin/env python3
import numpy as np
import copy

# minimum gain for a node to be considered for splitting
min_gain = 0.2


# Implementation of ID3 classifier
class ID3():

    # a function to compute the weighted majority class out of a list of classes
    def pick_majority(self, classes):
        score = [0, 0, 0, 0, 0]
        for c in classes:
            score[c] += self.class_weights[c]
        return np.argmax(score)

    # constructor for ID3 node
    def __init__(self,max_depth=125,min_size=4,weights=(1,1,1.1,1.1,1.2)):
        self.class_weights=weights
        self.children = []

        self.max_depth = max_depth
        self.min_size = min_size

        self.isLeaf = False
        self.partitionFeature = -1
        self.partitionValue = -1
        self.Predication = -1


    # public training function
    def fit(self, X, Y):
        self.__fit(np.array(X), np.array(Y), 0, [])

    # Inner training function
    def __fit(self, X, Y, depth, done):
        classes = [_ for _ in Y]

        # check base cases
        if len(set(classes)) < 2 or depth > self.max_depth or len(X) <= self.min_size or len(done) >= len(X[0]):
            self.isLeaf = True
            self.predication = self.pick_majority(classes)
        else:
            # compute highest information gain and it's corresponding feature
            best, gain = self.get_best_feature(X, Y, done)

            # if we don't have enough gain stop and do majority classification
            if gain < min_gain:
                self.isLeaf = True
                self.predication = self.pick_majority(classes)
            else:

                # otherwise, split the data and create a child for each split
                (Xs, Ys) = self.split(X, Y, best)
                print("splitting on the %dth feature" % best)


                for i in range(len(Xs)):
                    t = ID3(self.max_depth,self.min_size,self.class_weights)

                    # declare this feature to be used
                    temp = copy.copy(done)
                    temp.append(best)

                    # train the subtree
                    t.__fit(Xs[i], Ys[i], depth + 1, [])

                    # store splitting feature and it's value
                    t.partitionFeature = best
                    t.partitionValue = X[i][best]

                    #add the child
                    self.children.append(t)

    # public predication function
    def predict(self, X):
        yp = []
        # predict every input and return a predictions
        for test in X:
            yp.append(self.predict_row(np.array(test)))
        return yp

    # predict one point
    def predict_row(self, X):
        # check base case
        if self.isLeaf:
            return self.predication
        else:
            # pick the child with the nearest partition value on the split feature
            mn = np.inf
            best_child = None
            for child in self.children:
                if np.abs(X[child.partitionFeature] - child.partitionValue) < mn:
                    mn = np.abs(X[child.partitionFeature] - child.partitionValue)
                    best_child = child

            # return child predication
            return best_child.predict_row(X)


    # DEBUG methods used to print the tree
    def __print_tree(self, depth=0):
        if not self.isLeaf:
            print('%s[X%d < %s]' % (depth * ' ', (self.partitionFeature + 1), self.partitionValue))
            for child in self.children:
                child.__print_tree(depth + 1)
        else:
            print('%s[%s]' % (depth * ' ', str(self.predication)))

    def print_tree(self):
        self.__print_tree(0)

    # get the best feature to split on using information gain
    def get_best_feature(self,X, Y, done):
        best_gain = -np.inf
        best_feature = -1
        l = len(X[0])

        # try all unused features
        for feature in range(0, l):
            if not feature in done:
                gain = self.information_gain(X, Y, feature)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
        return (best_feature, best_gain)

    # split the data based on a feature
    def split(self,X, Y, feature):
        bucketX = {}
        bucketY = {}

        # assign each row to a bucket based on it's value on feature
        for i in range(len(X)):
            idx = X[i][feature]

            if not idx in bucketX:
                bucketX[idx] = []

            if not idx in bucketY:
                bucketY[idx] = []

            bucketX[idx].append(X[i])
            bucketY[idx].append(Y[i])

        # return the X splits and their corresponding Y splits
        return np.array(bucketX.values()), np.array(bucketY.values())

    # compute entropy for data on a specific feature
    def entropy(self,data, feature):
        class_size = {}
        entropy = 0.0

        for row in data:
            idx = row[feature]
            if class_size.has_key(idx):
                class_size[idx] += 1.0
            else:
                class_size[idx] = 1.0

        total_size = len(data)
        for subset_size in class_size.values():
            entropy += ((-subset_size / float(total_size)) * np.log2(subset_size / float(total_size)))
        return entropy

    # compute entropy for 1D vector
    def vector_entropy(self,v):
        class_size = {}
        entropy = 0.0

        for e in v:
            if class_size.has_key(e):
                class_size[e] += 1.0
            else:
                class_size[e] = 1.0

        total_size = len(v)
        for subset_size in class_size.values():
            entropy += ((-subset_size / float(total_size)) * np.log2(subset_size / float(total_size)))
        return entropy

    # compute information gain for data in X,Y when get splitted on feature
    def information_gain(self,X, Y, feature):
        values = {}
        subset_entropy = 0.0

        for i in range(len(X)):
            idx = X[i][feature]
            if values.has_key(idx):
                values[idx][0] += 1.0
                values[idx][1].append(Y[i])
            else:
                values[idx] = [1.0, [Y[i]]]

        total_size = len(X)

        for value in values.keys():
            prob = values[value][0] / total_size
            Y_subset = values[value][1]
            subset_entropy += prob * self.vector_entropy(Y_subset)

        return self.vector_entropy(Y) - subset_entropy