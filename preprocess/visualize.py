## visualize distribution for the training data
import numpy as np
import matplotlib.pyplot as plt

## label distribution
def label_dist():
    Y = np.load("Train_Y.npy")
    lens = []
    for i in range(5):
        lens.append(np.sum(Y==i))
    
    lens = np.array(lens).astype(float)
    lens = lens/np.sum(lens)
    print(lens)
    n, bins, patches = plt.hist(range(5),weights=lens,normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Languages')
    plt.ylabel('Proportion')
    plt.title(r'Distribution of labels')
    # plt.axis(["Slovak","French","Spanish","German","Polish"])
    plt.axis([-1,5,0,1])
    plt.show()


## character variety per language distribution


if __name__ == "__main__":
    label_dist()