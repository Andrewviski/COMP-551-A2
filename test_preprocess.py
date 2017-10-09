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
    for i in range(0, len(lines) - 1):
        line = lines[i]
        has_bad_word = False
        j = 0
        while True:
            # print(j)
            if j < len(line) - 1:
                char = line[j]
                if char not in alphabet:
                    if char + line[j+1] not in alphabet:
                        has_bad_word = True
                        print "Number" + str(i)
                        print char + line[j+1]
                        break
                    j += 1
                j += 1
            else:
                break
        if has_bad_word:
            print(line)

    for char in lines[173]:
        print [char]
def check():
    f = open('data/train_set_x.csv', 'r')
    reader = csv.reader(f)
    lines = [row[1] for row in reader]
    f.close()
    for char in lines[77]:
        print [char]

# init()
check()