import numpy as np
# from sortedcontainers import SortedList
from scipy.sparse import csc_matrix

class KNN():
    """ Implementation of K nearest neighbours """
    def __init__(self, k):
        self.k = k
    
    def fit(self, X,y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        y = np.zeros(len(X))
        for i,x in enumerate(X): # test points
            print("%sth point"%(i))
            sl = SortedList(load=self.k) # stores (distance, class) tuples
            for j,xt in enumerate(self.X): # training points
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add( (d, self.y[j]) )
            # vote
            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v,0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            for v,count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

class KNNFast():
    def __init__(self, k):
        self.k = k
    
    def fit(self, X,y):
        self.X = X
        self.y = y

    def predict(self, X):
        ## inverted list
        I = []
        for i in range(self.X.shape[1]):
            I.append([])
        for i in range(self.X.shape[0]):
            s = self.X[i]
            for j,w in enumerate(s):
                if w!=0:
                    I[j].append((i,w))
       
        # I_ = np.zeros((self.X.shape[1],self.X.shape[0]))
        # arg = np.argwhere(self.X!=0).reshape((-1,2))
        # values = self.X[np.where(self.X!=0)]
        # I_[arg[...,1],arg[...,0]] = values
        # print(I_)

        ## find match
        A = np.zeros((X.shape[0],self.X.shape[0]))
        A_ = np.zeros((X.shape[0],self.X.shape[0]))
        for i,r in enumerate(X):
            for d,rd in enumerate(r):
                I_d = I[d]
                for s,s_d in I_d:
                    A[i,s] += rd*s_d
                
                # I_d_ = I_[d]
                # for s,s_d in zip(np.argwhere(I_d_!=0),I_d_[np.where(I_d_!=0)]):
                #     A_[i,s] += rd*s_d
        # print(A)
        # print(A_)

        neighb_idx = np.argpartition(A, -self.k)[...,-self.k:]
        
        ## predict
        yp = np.zeros((X.shape[0],len(set(self.y))))
        for i,n in enumerate(neighb_idx):
            votes = self.y[n]
            # print("votes",votes)
            counts = np.bincount(votes)
            yp[i,np.argwhere(counts!=0).ravel()] += counts[np.where(counts != 0)]
        # print(yp)
        return np.argmax(yp,axis=1)
        


if __name__=="__main__":
    X = np.array([[1,2,0,0],[0,-1,0,0.3],[0.5,10,8,7]])
    y = np.array([0,0,1])
    r = np.array([[0.3,-1,0,5],[-1,0,3,-8]])
   
    knn = KNNFast(1)
    # knn = KNN(3)
    knn.fit(X,y)
    y_hat = knn.predict(r)
    print(y_hat)
