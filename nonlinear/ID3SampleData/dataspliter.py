#!/usr/bin/env python3
import sys
if __name__ == "__main__":
    if len(sys.argv)>=2:
        f=open(sys.argv[1],"r")
        trainPortion=int((int(sys.argv[2])*(sum(1 for line in open(sys.argv[1],"r"))))/100.0)

        train = open(sys.argv[1].split(".")[0]+"Train.csv","w")
        test = open(sys.argv[1].split(".")[0] + "Test.csv", "w")

        idx=0;
        for line in f:
            if idx<=trainPortion:
                train.write(line)
            else:
                test.write(line)
            idx+=1
