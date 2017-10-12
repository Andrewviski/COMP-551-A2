# -*- coding: utf-8 -*-

def init():
    f = open('emoji_removed.txt', 'r')
    lines = f.readlines()
    f.close()
    for i in range(0, len(lines) - 1):
        line = lines[i]
        line = ''.join([j for j in line if not j.isdigit()])
        try:
            print line.strip('\n')
        except:
            print '\n'

init()