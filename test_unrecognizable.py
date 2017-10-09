# -*- coding: utf-8 -*-
import csv
from collections import Counter

def init():
    f = open('temp.txt', 'r')
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