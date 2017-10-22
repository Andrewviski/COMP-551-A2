import numpy as np
# from scipy.misc import logsumexp

class NaiveBayes():
    """Multinomial Naive Bayes"""
    def __init__(self,smoothing = 1e-2):
        self.theta = [] # [i]: theta for the ith feature
        self.X2idx = [] # mapping of x in ith feature to index in theta
        self.UNK = 0 # for unkown feature value
        self.smoothing = smoothing # for smoothing MLE
    
    def fit(self, X, Y):
        self.theta = []
        self.X2idx = []
        _, m = X.shape # m = # features
        self.labels = {0,1,2,3,4}
        self.priors = np.zeros(len(self.labels)) # [i]: Pr(Y=i)

        for c in range(m):
            x = X[:,c]
            # print("\n%sth feature"%(c))
            theta_i = np.zeros((len(set(x))+1,len(self.labels))) # [i][j] = Counts(X_c=idx(i)-1|Y=j). UNK = [0]
            x2idx = dict(zip(set(x), range(1,len(set(x))+1)))
            
            for y in self.labels:
                values = np.array([x2idx[a] for a in x[Y==y]]).astype(int)
                counts = np.bincount(values)
                # print(np.argwhere(counts!=0) ,counts[np.where(counts != 0)])
                theta_i[np.argwhere(counts!=0).ravel(),y] += counts[np.where(counts != 0)]
            # print("theta_i",theta_i)
            # self.theta.append(theta_i)
            theta_i = theta_i+self.smoothing
            theta_i = np.log(theta_i)-np.log(np.sum(theta_i,axis=0)) # normalize with smoothing
            self.X2idx.append(x2idx)
            self.theta.append(theta_i)
            
        for c in self.labels:
            self.priors[c] = np.log(float(len(Y[Y == c])+self.smoothing)) -np.log(len(Y)+len(self.labels)*self.smoothing)
       

    def predict(self, X):
        assert X.shape[1] == len(self.theta), (X.shape[1],len(self.theta))
        P = np.zeros((X.shape[0], len(self.labels))) # [i,j] = P(Y=j|x_i)
        for i in range(X.shape[1]):
            theta_i = self.theta[i]
            x2idx = self.X2idx[i]
            x = X[:,i]
            values = np.array([x2idx.get(a,self.UNK) for a in x]).astype(int)
            
            P += theta_i[values]
        
        P += self.priors
        return np.argmax(P,axis=1)

   
class NaiveBayes2():
    """Multinomial Naive Bayes. sum up numbers of data"""
    def __init__(self,smoothing = 1e-2):
        self.smoothing = smoothing # for smoothing MLE


    def fit(self,X,Y):
        n,m = X.shape
        self.labels = set(Y)
        self.counts = np.zeros((m,len(self.labels)))
        for y in self.labels:
            x = X[Y==y].T
            self.counts[:,y] += np.sum(x,axis=1)
        self.counts += self.smoothing
        self.theta = np.log(self.counts) - np.log(np.sum(self.counts, axis=0))
        self.priors = np.zeros(len(self.labels))
        for c in self.labels:
            self.priors[c] = np.log(float(len(Y[Y == c])+self.smoothing)) -np.log(len(Y)+len(self.labels)*self.smoothing)
    
    def predict(self, X):
        assert X.shape[1] == len(self.theta), (X.shape[1],len(self.theta))
        P = np.zeros((X.shape[0], len(self.labels))) # [i,j] = P(Y=j|x_i)
        P = X.dot(self.theta)
        P += self.priors
        return np.argmax(P,axis=1)

if __name__ == "__main__":
    ## dummy test
    X = np.array([[1,0,1],[0,2,float(1)/2],[3,float(1)/3,3],[0,2,4],[1,2,float(5)/2],[1,0,3]])
    Y = np.array([0,2,1,0,1,0])

    nb = NaiveBayes2(smoothing = 1)
    nb.fit(X,Y)
    x = np.array([[2,1,5],[0,2,3]])
    yp = nb.predict(x)
