# -*- coding: utf-8 -*-
import csv
import re

def init():
    f = open('data/train_set_x.csv', 'r')
    reader = csv.reader(f)
    lines = [row[1] for row in reader]
    f.close()

    alphabet = [Ą,ą,Ł,€,Š,§,š,Ș,Ź,ź,Ż,Č,ł,Ž,ž,č,ș,Œ,œ,Ÿ,ż,À,Á,Â,Ă,Ä,Ć,Æ,Ç,È,É,Ê,Ë,Ì,Í,Î,Ï,Đ,Ń,Ò,Ó,Ô,Ő,Ö,Ś,Ű,Ù,Ú,Û,Ü,Ę,Ț,ß,à,á,â,ă,ä,ć,æ,ç,è,é,ê,ë,ì,í,î,ï,đ,ń,ò,ó,ô,ő,ö,ś,ű,ù,ú,û,ü,ę,ț,ÿ]
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