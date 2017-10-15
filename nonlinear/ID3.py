import numpy as np
import sys
from pprint import pprint

class ID3:

    def __init__(self,x,y):
        self.l=None
        self.r=None
        self.isLeaf=False
        self.Predication=-1
        self.fit(x,y)

    def fit(self,x,y):
        nclasses=len(np.unique(y))
        if nclasses<2:
            self.isLeaf=True
            self.predication=y[0]
        else:
            ##naive spliting for now ...
            ##need to research entropy based spliting...
            l=len(x)
            l//=2
            print("We are at dept:"+ str(l))
            x1,x2=x[l:],x[:l]
            y1,y2=y[l:],y[:l]
            self.l=ID3(x1,y1)
            self.r=ID3(x2,y2)

if __name__=="__main__":
    if len(sys.argv)==4:
        queries = open(sys.argv[3])
        data =[]
        X = []
        Y = []
        for input,output in zip(open(sys.argv[1]),open(sys.argv[2])):
            tempX=[]
            for featureValue in input.split(","):
                featureValue=featureValue.strip()
                if featureValue!="":
                    tempX.append(featureValue)
            X.append(tempX)
            Y.append(output.strip())
        pprint(str(X)+"   =   "+str(Y))
        tree=ID3(np.array(X),np.array(Y))

    else:
        print("Usage:\n./ID3 [Input].csv [Output].csv [Queries].csv")
