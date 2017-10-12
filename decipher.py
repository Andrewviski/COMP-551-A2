# -*- coding: utf-8 -*-
import nltk.tag.hmm
import sys, os

def init():
    path = sys.argv[1]
    f = open(os.join(path, 'train_cipher.txt'), 'r')
    lines = f.readlines()
    f.close()

init()