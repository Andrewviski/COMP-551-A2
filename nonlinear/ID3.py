#!/usr/bin/env python3
import sys
import BasicSelector
import random

max_depth = 100
min_size = 3

class ID3:
    def __init__(self):
        self.l = None
        self.r = None
        self.partitionFeature = -1
        self.partitionValue = -1
        self.isLeaf = False
        self.Predication = -1

    def pick_majority(self, Y):
        outcomes = [v for v in Y]
        return max(set(outcomes), key=outcomes.count)



    def fit(self, X, Y, depth):
        nclasses = len(set(_ for _ in Y))
        if nclasses < 2 or depth > max_depth or len(X) <= min_size:
            self.isLeaf = True
            self.predication = self.pick_majority(Y)
        else:
            best = BasicSelector.get_best_split(X, Y,random=False,trials=20)
            (lx, ly, rx, ry) = best['partitions']
            self.partitionFeature = best['feature']
            self.partitionValue = best['value']

            if len(lx)!=0:
                self.l = ID3()
                self.l.fit(lx, ly, depth + 1)

            if len(rx) != 0:
                self.r = ID3()
                self.r.fit(rx, ry, depth + 1)

    def predict(self, X):
        if self.isLeaf:
            return self.predication
        elif X[self.partitionFeature] <= self.partitionValue or self.r==None:
            return self.l.predict(X)
        elif X[self.partitionFeature] > self.partitionValue or self.l==None:
            return self.r.predict(X)

    def __print_tree(self, depth=0):
        if not self.isLeaf:
            print('%s[X%d < %s]' % (depth * ' ', (self.partitionFeature + 1), self.partitionValue))
            if self.l!=None:
                self.l.__print_tree(depth + 1)
            if self.r != None:
                self.r.__print_tree(depth + 1)
        else:
            print('%s[%s]' % (depth * ' ', (self.predication)))

    def print(self):
        self.__print_tree(0)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("[*] Reading Training data...")
        data = []
        for line in open(sys.argv[1]):
            tempRow = []
            for value in line.split(","):
                value = value.strip()
                if value != "":
                    tempRow.append(value)
            data.append(tempRow)

        print("[*] Shuffling data...")
        random.shuffle(data)
        #pprint(data)

        Y = [row[0] for row in data]
        X = [row[1:] for row in data]
        tree = ID3()
        print("[*] Training tree...")
        tree.fit(X, Y, 0)
        tree.print()

        print("[*] Testing...")
        correct = 0
        total=0
        for line in open(sys.argv[2]):
            row=[_ for _ in line.strip().split(",")]
            pred = tree.predict(row[1:])
            #print("ID3 predict %s, actual class is %s" % (pred, row[-1]))
            if pred == row[0]:
                correct += 1
            total+=1
        print('Accuracy= %f%%' % (correct * 100 / float(total)))

    else:
        print("Usage:\n./ID3 [TrainData].csv [TestData].csv {Max_depth} {Min_size}")
