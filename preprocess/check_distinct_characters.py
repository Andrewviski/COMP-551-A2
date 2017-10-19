# -*- coding: utf-8 -*-
import csv
def init():
    f = open('digits_removed.txt', 'r')
    content = f.read()
    f.close()

    # f = open('../data/train_set_x.csv', 'r')
    # reader = csv.reader(f)
    # content = [row[1] for row in reader]
    # f.close()

    l = []
    for line in content:
    	for s in line.translate(None, " \n"):
	        if s not in l:
	            l.append(s)
    for c in l:
        print c

init()