## make our own validation set
import csv
import numpy as np


with open('../data/train_set_x.csv', 'r') as f:
    reader = csv.reader(f)
    train_lines = [row[1].decode('latin-1').encode("utf-8").translate(None, " \n") for row in reader]
    train_lines = train_lines[1:]
with open('../data/train_set_y.csv', 'r') as f:
    reader = csv.reader(f)
    label = [row[1] for row in reader]
    label = label[1:]
    label = np.array(label).reshape((-1, 1)).astype(int)

assert len(train_lines) == len(label)

## collect per language character frequency from training set
freq = []
lens = []
for i in range(5):
    char2freq = {}
    lens_ = []
    lines = [train_lines[idx] for idx in np.argwhere(label==i).ravel()]
    for line in lines:
        n_char = 0
        for c in line.lower().strip():
            if c==" ":
                continue
            if not char2freq.get(c):
                char2freq[c] = 1
            else:
                char2freq[c] = char2freq.get(c)+1
            n_char += 1
        lens_.append(n_char)
    lens_ = np.array(lens_)
    lens.append([np.mean(lens_),np.std(lens_)])
    freq.append(char2freq)


n_valid = 1000 # #validation data per language
validation = []
for i in range(5):
    chars = [k for k,v in freq[i].items()]
    p = [float(v) for k,v in freq[i].items()]
    p = np.array(p)/np.sum(p)
    data_per_lang = []

    length = np.abs(np.random.normal(lens[i][0]/2,lens[i][1],size=n_valid)).astype(int) # length sampled from gaussian
    for l in length:
        ## sample a line of character     
        idx = np.random.choice(len(p), l,replace = True, p=p).astype(int)
        line = [chars[a] for a in idx]
        data_per_lang.append(((" ").join(line),i))
    validation.extend(data_per_lang)

## save into data/
with open('../valid_set_x_y.csv', 'wt') as f:
    writer = csv.writer(f)
    writer.writerow(('Id', 'Category'))
    for i in range(len(validation)):
        writer.writerow((i, validation[i][0],validation[i][1]))

