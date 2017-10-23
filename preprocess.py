# -*- coding: utf-8 -*-
import csv
import re
import string
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import parameters as p


def init():
    f = open('./data/train_set_x.csv', 'r')
    reader = csv.reader(f)
    lines = [row[1] for row in reader]
    lines = lines[1:]
    f.close()
    return lines


def lower_case(lines):
    return [line.lower() for line in lines]


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
        extracted += line[last_end:] + '\n'
        result.append(extracted)
    return result


def iter_remove(line, index):
    if not index == -1:
        line = line.replace(line[index:index + 4], '')
        index = line.find('\xf0\x9f')
        return iter_remove(line, index)
    else:
        return line


def remove_emoji(lines):
    result = []
    for i in range(len(lines)):
        line = lines[i]
        index = line.find('\xf0\x9f')
        line = iter_remove(line, index)
        result.append(line)
    return result


def remove_digits(lines):
    return [''.join([j for j in line if not j.isdigit()]) for line in lines]

def remove_spaces(lines):
    return [line.translate(None, " \n") for line in lines]

def remove_punctuation(lines):
    f = open('./preprocess/remove.txt', 'r')
    to_remove = f.readlines()
    f.close()
    remove = []
    for line in to_remove:
        if line:
            remove.append(line)

    exclude = set(string.punctuation).union(set(remove))

    result = []
    for line in lines:
        for symbol in exclude:
            if symbol in line:
                line = line.replace(symbol, '')
        result.append(line)
    return result


def pipeline(l, lines):
    lines = lower_case(lines)
    if l[0]: lines = remove_url(lines)
    if l[1]: lines = remove_emoji(lines)
    if l[2]: lines = remove_digits(lines)
    if l[3]: lines = remove_spaces(lines)
    if l[4]: lines = remove_punctuation(lines)
    return lines


def create_data(data, vocab, ngram_range=(1, 1), max_features=5000, analyzer="char_wb", tfidf=True, save=False):
    f = open('./data/train_set_y.csv', 'r')
    reader = csv.reader(f)
    label = [row[1] for row in reader]
    label = label[1:]
    label = np.array(label).reshape((-1, 1))

    if tfidf:
        vect = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, vocabulary = vocab, analyzer=analyzer)
        X = vect.fit_transform(data)
        # print X.shape

    else:
        vect = CountVectorizer(ngram_range=ngram_range, max_features=max_features, vocabulary = vocab, analyzer=analyzer)
        X = vect.fit_transform(data)
    
    if save:
        np.save("Train_X",X.todense())
        np.save("Train_Y",label)

    return vect.get_feature_names()


def process_test_set():
    f = open('./data/test_set_x.csv', 'r')
    reader = csv.reader(f)
    data = [row[1].translate(None, " \n") for row in reader]
    f.close()

    return data[1:]


def check_characters(l):
    for s in l:
        print s


def generate_vocab(lines):
    l1 = create_data(lines, None, p.ngram, p.max_features, p.analyzer, p.tfidf)
    l2 = process_test_set(None, p.ngram, p.max_features, p.analyzer, p.tfidf)
    uni = list(set(l1).union(set(l2)))
    inter = list(set(l1).intersection(set(l2)))
    diff1 = list(set(l1) - set(l2))
    diff2 = list(set(l2) - set(l1))
    # print "Intersection: "
    # print inter
    # print "Diff1:        "
    # check_characters(diff1)
    # print "Diff2:        "
    # print(diff2)
    # check_characters(diff2)
    return uni, inter, diff1, diff2


def generate_data_given_vocab(lines, vocab):
    l1 = create_data(lines, vocab, p.ngram, p.max_features, p.analyzer, p.tfidf, save=True)
    l2 = process_test_set(vocab, p.ngram, p.max_features, p.analyzer, p.tfidf,save=True)
    return l1, l2


def do_nothing():
    ## do no preprocessing. Use *original* .csv file downloaded from Kaggle
    with open('./data/train_set_x.csv', 'r') as f:
        reader = csv.reader(f)
        train_lines = [row[1] for row in reader]
        train_lines = train_lines[1:]

    with open('./data/test_set_x.csv', 'r') as f:
        reader = csv.reader(f)
        data = [row[1] for row in reader]
        test_lines = data[1:]

    with open('./data/train_set_y.csv', 'r') as f:
        reader = csv.reader(f)
        label = [row[1] for row in reader]
        label = label[1:]
        y_train = np.array(label).reshape((-1, 1)).astype(int)

    with open('../valid_set_x_y.csv', 'r') as f:
        reader = csv.reader(f)
        data = [row[1] for row in reader]
        valid_lines = data[1:]
    with open('../valid_set_x_y.csv', 'r') as f:
        reader = csv.reader(f)
        label = [row for row in reader]
        label = label[1:]
        label = [row[2] for row in label]
        valid_y = np.array(label).reshape((-1, 1)).astype(int)

    train_lines = lower_case(train_lines)
    test_lines = lower_case(test_lines)
    valid_lines = lower_case(valid_lines)

    vect = CountVectorizer(ngram_range = (1,1), analyzer="char",encoding='latin-1')
    X_train = vect.fit_transform(train_lines)
    vect2 = CountVectorizer(ngram_range = (1,1), analyzer="char",encoding='latin-1', vocabulary = vect.vocabulary_)
    X_test = vect2.fit_transform(test_lines)
    X_valid = vect2.fit_transform(valid_lines)

    np.save("Train_X", X_train.todense())
    np.save("Test_X", X_test.todense())
    np.save("Train_Y",y_train)
    np.save("Val_X",X_valid.todense())
    np.save("Val_Y",valid_y)

def construct_vector(lines, test_lines, existing = False):
    
    all_chars = list(set("".join(lines)))
    print(all_chars)
    feature_num = len(all_chars) + 1  # UNK

    result = np.zeros((len(lines),feature_num)).astype(int)
    for i in range(len(lines)):
        line = lines[i]
        line = line
        counter = Counter(line)
        for (u,v) in counter.items():
            index = all_chars.index(u)
            result[i][index] = v
    print("Manual train X shape",result.shape)
    np.save("Manual_train_X.npy", result)


    result = np.zeros((len(test_lines),feature_num)).astype(int)
    for i in range(len(test_lines)):

        line = test_lines[i]
        line = line
        counter = Counter(line)
        for (u,v) in counter.items():

            try:
                index = all_chars.index(u)
            except:
                index = feature_num - 1
            result[i][index] = v
            # print u, v, index
    print("Manual test X shape",result.shape)
    np.save("Manual_test_X.npy", result)



if __name__ == "__main__":
    # do_nothing()

    lines = init()
    lines = lower_case(lines)
    lines = remove_spaces(lines)
    # vocab,_,_,_ = generate_vocab(lines)
    # lines = pipeline([p.remove_url, p.remove_emoji, p.remove_digits, p.remove_spaces, p.remove_punctuation], lines)
    # generate_data_given_vocab(lines, vocab)
    test_data = process_test_set()
    construct_vector(lines, test_data)