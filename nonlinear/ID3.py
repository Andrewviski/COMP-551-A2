#!/usr/bin/env python3
import sys
import BasicSelector
import random
import math
from pprint import pprint
import numpy as np

max_depth = 20
min_size = 3
Nbuckets=90

def split(X,Y,feature):
    unique_values=set(X[:,feature])
    buckets={}
    for value in unique_values:
        buckets[int(value)%Nbuckets]={'X':[],'Y':[]}

    for i in range(0,len(X)):
        buckets[int(X[i][feature])%Nbuckets]['X'].append([x for i,x in enumerate(X[i]) if i!=feature])
        buckets[int(X[i][feature])%Nbuckets]['Y'].append(Y[i])
    return (buckets)

def entropy(Y):
    total_size=len(Y)
    class_size={}
    for e in Y:
        if e  in class_size:
            class_size[e]+=1
        else:
            class_size[e]=1
    return sum([(-(subset_size/total_size)*math.log2(subset_size/total_size)) for subset_size in class_size.values()])

def information_gain(feature,X,Y):
    splits=split(X,Y,feature).values()
    sz=len(X)
    score=0
    for subset in splits:
        score+=(len(subset['X'])/sz)*entropy(subset['Y'])
    return entropy(Y)-score

def get_best_attribute(X,Y):
    return max((information_gain(feature,X,Y),feature) for feature in range(0,len(X[0])))


class ID3:
    def __init__(self):
        self.children = []

        self.isLeaf = False
        self.partitionFeature = -1
        self.partitionValue=-1
        self.Predication = -1

    def pick_majority(self, Y):
        classes = [v for v in Y]
        return max(set(classes), key=classes.count)

    def fit(self, X, Y, depth):
        nclasses = len(set(_ for _ in Y))
        if nclasses < 2 or depth > max_depth or len(X) <= min_size:
            self.isLeaf = True
            self.predication = self.pick_majority(Y)
        else:
            best=get_best_attribute(X,Y)
            subsets = split(X,Y,best[1])
            print("splitting on %d"%best[1])
            for splitValue in subsets:
                subset=subsets[splitValue]
                t=ID3()
                t.fit(np.array(subset['X']),np.array(subset['Y']),depth+1)
                t.partitionFeature=int(best[1])
                t.partitionValue=splitValue
                print("With value %s"%splitValue)
                self.children.append(t)

    def predict(self, X):
        if self.isLeaf==True:
           return self.predication
        else:
            for child in self.children:
                if X[child.partitionFeature]==child.partitionValue:
                    return child.predict(X)
            mn=np.inf
            bestChild=None
            for child in self.children:
                #print(X[child.partitionFeature] +"   vs   "+int(child.partitionValue))
                if abs(int(X[child.partitionFeature])-int(child.partitionValue))<mn:
                    #print("new min= "+str(mn))
                    mn=abs(int(X[child.partitionFeature])-int(child.partitionValue))
                    bestChild=child
            return bestChild.predict(X)

    def __print_tree(self, depth=0):
        if not self.isLeaf:
            print('%s[X%d < %s]' % (depth * ' ', (self.partitionFeature + 1), self.partitionValue))
            for child in self.children:
                child.__print_tree(depth + 1)
        else:
            print('%s[%s]' % (depth * ' ', str(self.predication)))

    def print(self):
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
        #pprint(data)

        Y = np.array([row[-1] for row in data if len(row)>0])
        X = np.array([row[:-2] for row in data if len(row)>0])
        #pprint(split(X,Y,get_best_attribute(X,Y)[1]))
        tree = ID3()
        print("[*] Training tree...")
        tree.fit(X, Y, 0)
        #tree.print()

        print("[*] Testing...")
        correct = 0
        total=0
        for line in open(sys.argv[2]):
            row=[_ for _ in line.strip().split(",")]
            pred = tree.predict(np.array(row[:-2]))
            print("ID3 predict %s, actual class is %s" % (pred, row[-1]))
            if pred == row[-1]:
                correct += 1
            total+=1
        print('Accuracy= %f%%' % (correct * 100 / float(total)))

    else:
        print("Usage:\n./ID3 [TrainData].csv [TestData].csv {Max_depth} {Min_size}")
