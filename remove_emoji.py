# -*- coding: utf-8 -*-

def init():
    f = open('url_removed.txt', 'r')
    lines = f.readlines()
    f.close()
    for i in range(0, len(lines) - 1):
        line = lines[i]
        index = line.find('\xf0\x9f')
        lines[i] = iter_remove(line, index)
        try:
            print lines[i].strip('\n')
        except:
            print '\n'

def iter_remove(line, index):
    if not index == -1:
        line = line.replace(line[index:index + 4], '')
        index = line.find('\xf0\x9f')
        iter_remove(line, index)
    else:
        return line

init()