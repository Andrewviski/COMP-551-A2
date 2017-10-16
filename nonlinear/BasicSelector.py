#!/usr/bin/env python3
import random


def partition(feature, val, X, Y):
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
def evaluate(partitionsX, partitionsY, classes):
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


def get_best_partition_brute_force(X, Y):
    classes = list(set(_ for _ in Y))
    best = {'score': 9999, 'feature': -1, 'value': -1, 'partitions': None}
    for feature in range(len(X[0])):
        done = {}
        for row in X:
            if row[feature] in done:
                continue
            done[row[feature]]=1
            (lx, ly, rx, ry) = partition(feature, row[feature], X, Y)
            cur_score = evaluate([lx, rx], [ly, ry], classes)
            if cur_score < best['score']:
                best['score'] = cur_score
                best['feature'] = feature
                best['value'] = row[feature]
                best['partitions'] = (lx, ly, rx, ry)
    print(">>>>Best Spliting on X%d=%s, left size=%d, right size=%d"
          %(best["feature"],best["value"],len(best["partitions"][0]),len(best["partitions"][2])))
    return best


def get_best_partition_random(X, Y, trials):
    classes = list(set(_ for _ in Y))
    best = {'score': 9999, 'feature': -1, 'value': -1, 'partitions': None}
    done = {}
    for _ in range(trials):
        row = random.randint(0, len(X) - 1)
        feature = random.randint(0, len(X[row]) - 1)
        (lx, ly, rx, ry) = partition(feature, X[row][feature], X, Y)
        cur_score = evaluate([lx, rx], [ly, ry], classes)
        if cur_score < best['score']:
            best['score'] = cur_score
            best['feature'] = feature
            best['value'] = X[row][feature]
            best['partitions'] = (lx, ly, rx, ry)
            best['partitions'] = (lx, ly, rx, ry)

        print(">>>>Best Spliting on X%d=%s, left size=%d, right size=%d"
              % (best["feature"], best["value"], len(best["partitions"][0]), len(best["partitions"][2])))
    return best

def get_best_split(X,Y,random=True,trials=10):
    if not random:
        return get_best_partition_brute_force(X,Y)
    else:
        return get_best_partition_random(X,Y,trials)