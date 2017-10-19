# -*- coding: utf-8 -*-
import csv
import re
import preprocess.parameters as p
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def init():
    f = open('../data/train_set_x.csv', 'r')
    reader = csv.reader(f)
    lines = [row[1] for row in reader]
    lines = lines[1:]
    f.close()
    return lines


def remove_url(lines):
    regex1 = r"[h|H][t|T][t|T][p|P][^\s]*"
    result = []
    for line in lines:

        matches = re.finditer(regex1, line)
        last_end = 0
        extracted = ""
        for m in matches:
            extracted += line[last_end:m.start()]
            last_end = m.end()
        extracted += line[last_end:]
        result.append(extracted)
    return result


def iter_remove(line, index):
    if not index == -1:
        line = line.replace(line[index:index + 4], '')
        index = line.find('\xf0\x9f')
        iter_remove(line, index)
    else:
        return line


def remove_emoji(lines):
    result = []
    for line in lines:
        index = line.find('\xf0\x9f')
        result.append(iter_remove(line, index))
    return result


def remove_digits(lines):
    return [''.join([j for j in line if not j.isdigit()]) for line in lines]


def remove_spaces(lines):
    return [line.translate(None, " \n") for line in lines]


def create_data(data, ngram_range=(1, 1), max_features=5000, analyzer="char_wb", tfidf=True):
    f = open('./data/train_set_y.csv', 'r')
    reader = csv.reader(f)
    label = [row[1] for row in reader]
    label = label[1:]
    label = np.array(label).reshape((-1, 1))

    if tfidf:
        tfidf_vect = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, analyzer=analyzer)
        X = tfidf_vect.fit_transform(data)
        for x in tfidf_vect.get_feature_names():
            print x
    else:
        count_vect = CountVectorizer(ngram_range=ngram_range, max_features=max_features, analyzer=analyzer)
        X = count_vect.fit_transform(data)


def pipeline(l, lines):
    if l[0]: lines = remove_url(lines)
    if l[1]: lines = remove_emoji(lines)
    if l[2]: lines = remove_digits(lines)
    if l[3]: lines = remove_spaces(lines)
    return lines

def process_test_set(ngram_range = (1,1), max_features=5000, analyzer="char_wb", tfidf=True):
    f = open('./data/test_set_x.csv', 'r')
    reader = csv.reader(f)
    data = [row[1].translate(None, " \n") for row in reader]
    f.close()

    data = data[1:]

    if tfidf:
        tfidf_vect = TfidfVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
        X = tfidf_vect.fit_transform(data)
    else:
        count_vect = CountVectorizer(ngram_range = ngram_range, max_features = max_features, analyzer=analyzer)
        X = count_vect.fit_transform(data)

    X = X.todense()
    return X


if __name__ == "__main__":
    lines = init()
    pipeline([p.remove_url, p.remove_emoji, p.remove_digits, p.remove_spaces], lines)
    create_data()