import numpy as np
import heapq


class KNN_With_heap():
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = []
        for x in X:
            heap = []
            for j, xt in enumerate(self.X):
                diff = (x - xt)
                d = np.dot(diff, x - xt)
                if len(heap) < self.k:
                    heapq.heappush(heap, (-d, self.y[j]))
                elif -d > heap[0][0]:
                    heapq.heappushpop(heap, (-d, self.y[j]))
            res = max(set(np.ndarray.tolist(np.array(heap)[:, 1])), key=heap.count)
            print res
            y.append(res)
        return y