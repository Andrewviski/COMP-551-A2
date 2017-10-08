import numpy as np
# from scipy.misc import logsumexp



class NaiveBayes():
    """Multinomial Naive Bayes"""
    def __init__(self):
        self.theta = [] # [i]: theta for the ith feature
        self.X2idx = [] # mapping of x in ith feature to index in theta
        self.UNK = 0 # for unkown feature value
    
    def fit(self, X, Y, smoothing = 1e-2):
        _, m = X.shape # m = # features
        self.labels = set(Y)
        self.priors = np.zeros(len(self.labels)) # [i]: Pr(Y=i)
        
        for c in range(m):
            x = X[:,c]
            # print("\n%sth feature"%(c))
            theta_i = np.zeros((len(set(x))+1,len(self.labels))) # [i][j] = Counts(X_c=idx(i)-1|Y=j). UNK = [0]
            x2idx = dict(zip(set(x), range(1,len(set(x))+1)))
            # print("x2idx",x2idx)
            self.X2idx.append(x2idx)
            for y in self.labels:      
                values = [x2idx[a] for a in x[Y==y]]
                theta_i[values,y] += np.count_nonzero(values==values)
            # print("theta_i",theta_i)
            theta_i = np.log(theta_i+smoothing)-np.log(np.sum(theta_i,axis=0)+smoothing*theta_i.shape[0]) # normalize with smoothing
            self.theta.append(theta_i)

        for c in self.labels:
            self.priors[c] = float(len(Y[Y == c])) / len(Y)


    def predict(self, X):
        assert X.shape[1] == len(self.theta)
        P = np.zeros((X.shape[0], len(self.labels))) # [i,j] = P(Y=j|x_i)
        for i in range(X.shape[1]):
            theta_i = self.theta[i]
            x2idx = self.X2idx[i]
            x = X[:,i]
            values = [x2idx.get(a,self.UNK) for a in x]
            P += np.sum(theta_i[values])
        
        for c in self.labels:
            P[:,c] += np.log(self.priors[c])
        return np.argmax(P,axis=1)


if __name__ == "__main__":
    X = np.array([[1,0,1],[0,2,1/2],[3,1/3,3],[0,2,4],[1,2,5/2]])
    Y = np.array([0,2,1,0,2])

    nb = NaiveBayes()
    nb.fit(X,Y, smoothing = 1)
    x = np.array([[0,3,3]])
    yp = nb.predict(x)
