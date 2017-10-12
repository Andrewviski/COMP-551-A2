# -*- coding: utf-8 -*-
import string

def init():
    f = open('digits_removed.txt', 'r')
    lines = f.readlines()
    f.close()
    exclude = set(string.punctuation)
    for line in lines:
        
        line = ''.join(ch for ch in line if ch not in exclude)
        try:
            print line.strip('\n')
        except:
            print ''

init()