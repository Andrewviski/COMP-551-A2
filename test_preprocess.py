# -*- coding: utf-8 -*-
import csv
import re
import string

def build_alphabet():
    f = open('alphabet.txt', 'r')
    lines = f.readlines()
    f.close()
    words = lines[0]
    alphabet = words.split(",")
    return alphabet

def init():
    f = open('temp.txt', 'r')
    lines = f.readlines()
    f.close()
    alphabet = build_alphabet() + [c for c in string.ascii_letters] + [c for c in string.digits] + ['\n', ' ']
    print alphabet
    for i in range(0,3):
        line = lines[i]
        has_bad_word = False
        for char in line:
            if char not in alphabet:
                has_bad_word = True
                print [char]
                break
        if has_bad_word:
            print(line)


init()