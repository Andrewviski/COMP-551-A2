# -*- coding: utf-8 -*-
import csv, sys
from collections import Counter

def init():
    try:
        f = open(sys.argv[1], 'r')
    except:
        print "Please give a valid filename."
        sys.exit()
    lines = f.readlines()
    f.close()
    l = []
    for i in range(0, len(lines) - 1):
        line = lines[i]
        if '�' in line:
            l.append(i)

    f = open('data/train_set_y.csv', 'r')
    reader = csv.reader(f)
    lines = [row[1] for row in reader]
    f.close()
    print Counter([lines[i] for i in l])
    


init()