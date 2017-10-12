# -*- coding: utf-8 -*-
import csv
import re

def init():
    f = open('data/train_set_x.csv', 'r')
    reader = csv.reader(f)
    lines = [row[1] for row in reader]
    f.close()

    regex1 = r"[h|H][t|T][t|T][p|P][^\s]*"
    for line in lines:

        matches = re.finditer(regex1, line)
        last_end = 0
        extracted = ""
        for m in matches:
            extracted += line[last_end:m.start()]
            last_end = m.end()
        extracted += line[last_end:]
        line = extracted
        print line

init()