import sys
from pprint import pprint
import random

max_depth = 100
min_size = 10


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

    def partition(self, feature, val, X, Y):
        lx, ly, rx, ry = [], [], [], []
        for i in range(len(X)):
            if X[i][feature] < val:
                lx.append(X[i])
                ly.append(Y[i])
            else:
                rx.append(X[i])
                ry.append(Y[i])
        return (lx, ly, rx, ry)

    # evaluate a partitioning of the data
    def evaluate(self, partitionsX, partitionsY, classes):
        n_rows = sum([len(part) for part in partitionsX])
        # best -> 0, worst -> 1
        test_rating = 0.0
        for i in range(len(partitionsX)):
            sz = len(partitionsX[i])
            if sz != 0:
                score = 0.0
                for c in classes:
                    p = [_ for _ in partitionsY[i]].count(c) / sz
                    score += p * p
                test_rating += (1.0 - score) * (sz / n_rows)
        return test_rating

    def get_best_partition_brute_force(self, X, Y):
        classes = list(set(_ for _ in Y))
        best = {'score': 9999, 'feature': -1, 'value': -1, 'partitions': None}
        for row in X:
            for feature in range(len(row)):
                (lx, ly, rx, ry) = self.partition(feature, row[feature], X, Y)
                cur_score = self.evaluate([lx, rx], [ly, ry], classes)
                if cur_score < best['score']:
                    best['score'] = cur_score
                    best['feature'] = feature
                    best['value'] = float(row[feature])
                    best['partitions'] = (lx, ly, rx, ry)
        return best

    def get_best_partition_random(self, X, Y,trials):
        classes = list(set(_ for _ in Y))
        best = {'score': 9999, 'feature': -1, 'value': -1, 'partitions': None}
        for _ in range(trials):
            row=random.randint(0,len(X)-1)
            (lx, ly, rx, ry) = self.partition(feature, row[feature], X, Y)
            cur_score = self.evaluate([lx, rx], [ly, ry], classes)
            if cur_score < best['score']:
                best['score'] = cur_score
                best['feature'] = feature
                best['value'] = float(row[feature])
                best['partitions'] = (lx, ly, rx, ry)
        return best

    def fit(self, X, Y, depth):
        nclasses = len(set(_ for _ in Y))
        if nclasses < 2 or depth > max_depth or len(X) <= min_size:
            self.isLeaf = True
            self.predication = self.pick_majority(Y)
        else:
            best = self.get_best_partition_random(X, Y,100)
            (lx, ly, rx, ry) = best['partitions']
            self.partitionFeature = best['feature']
            self.partitionValue = best['value']
            self.l = ID3()
            self.l.fit(lx, ly, depth + 1)
            self.r = ID3()
            self.r.fit(rx, ry, depth + 1)

    def predict(self, X):
        if self.isLeaf:
            return self.predication
        elif X[self.partitionFeature] <= self.partitionValue:
            return self.l.predict(X)
        elif X[self.partitionFeature] > self.partitionValue:
            return self.r.predict(X)

    def __print_tree(self, depth=0):
        if not self.isLeaf:
            print('%s[X%d < %.3f]' % (depth * ' ', (self.partitionFeature + 1), self.partitionValue))
            self.l.__print_tree(depth + 1)
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
                    tempRow.append(float(value))
            data.append(tempRow)

        print("[*] Shuffling data...")
        random.shuffle(data)
        #pprint(data)

        Y = [int(row[0]) for row in data]
        X = [row[1:] for row in data]
        tree = ID3()
        print("[*] Training tree...")
        tree.fit(X, Y, 0)
        tree.print()

        print("[*] Testing...")
        correct = 0
        total=0
        for line in open(sys.argv[2]):
            row=[float(_) for _ in line.strip().split(",")]
            pred = tree.predict(row[1:])
            #print("ID3 predict %s, actual class is %s" % (pred, Y[i]))
            if pred == row[0]:
                correct += 1
        print('Accuracy= %f %%' % (correct * 100 / float(total)))

    else:
        print("Usage:\n./ID3 [TrainData].csv [TestData].csv {Max_depth} {Min_size}")
