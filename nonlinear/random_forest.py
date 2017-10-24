from nonlinear.RandomizedTree import randomized_tree
import random
import math
import numpy as np


# hard coded weights based on the unbalance in the training data
class_weights =[1.2,
         1,
         1.1,
         1,
         1.2]

# a function to compute the weighted majority class out of a list of classes
def pick_majority (classes):
    score = [0, 0, 0, 0, 0]
    for c in classes:
        score[c] += class_weights[c]
    return np.argmax(score)


# random forest classifier implementation
class random_forest():

    def __init__(self, size):
        self.size = size
        # hardcoded randomized trees with depth=150 and
        self.trees = [randomized_tree(150, 4,class_weights) for _ in range(size)]

    # training routine for classifier
    def fit(self, X, Y):

        #pick m inputs for each tree
        m = int(math.sqrt(len(X)))

        #split the data based on class
        class_samples = [[], [], [], [], []]
        for i in range(len(X)):
            class_samples[Y[i]].append(X[i])

        # train each tree
        for i in range(self.size):
            subX = []
            subY = []

            # the proportion of each class in the sample
            proportion = [5, 51, 27, 13, 5]

            #sample input points according to into subX,subY
            for c in range(len(proportion)):
                l = (m * proportion[c]) / 100
                for _ in range(l):
                    idx = random.randint(0, len(class_samples[c]) - 1)
                    subX.append(class_samples[c][idx])
                    subY.append(c)

            # train the ith tree
            self.trees[i].fit(subX, subY)

    # predication routine for random forests
    def predict(self, X):
        Yp = []
        for x in X:
            trees_results = []

            # predict each input against all trees
            for tree in self.trees:
                trees_results.append(tree.predict_row(x))

            #pick majority
            Yp.append(pick_majority(trees_results))
        return Yp
