# -*- coding: utf-8 -*-

def init():
    f = open('emoji_removed.txt', 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        line = ''.join([j for j in line if not j.isdigit()])
        try:
            print line.strip('\n')
        except:
            print ''

init()