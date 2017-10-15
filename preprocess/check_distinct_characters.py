# -*- coding: utf-8 -*-
def init():
    f = open('punctuation_removed.txt', 'r')
    content = f.read()
    f.close()
    l = []
    for s in content:
        if s not in l:
            l.append(s)
            print(s)

init()