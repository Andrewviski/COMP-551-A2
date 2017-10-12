# -*- coding: utf-8 -*-

def init():
    f = open('url_removed.txt', 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        index = line.find('\xf0\x9f')
        line = iter_remove(line, index)
        try:
            print line.strip('\n')
        except:
            print ''

def iter_remove(line, index):
    if not index == -1:
        line = line.replace(line[index:index + 4], '')
        index = line.find('\xf0\x9f')
        iter_remove(line, index)
    else:
        return line

init()