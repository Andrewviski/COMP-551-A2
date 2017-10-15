# -*- coding: utf-8 -*-
import string
import parameters as p

def init():
    f = open('remove.txt', 'r')
    lines = f.readlines()
    f.close()
    remove = []
    for line in lines:
        if line:
            remove.append(line.strip('\n'))
    # print remove

    f = open('digits_removed.txt', 'r')
    lines = f.readlines()
    f.close()
    exclude = set(string.punctuation)
    if p.remove_punctuation == True
        exclude = exclude.union(set(remove))

    for line in lines:
        for symbol in exclude:
            if symbol in line:
                line = line.replace(symbol, '')

        try:
            print line.strip('\n')
        except:
            print ''

init()