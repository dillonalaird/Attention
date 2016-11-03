from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

plt.style.use("ggplot")


def get_stats(fname, max_s):
    lines = open(fname).readlines()
    words = [len(line.split(" ")) for line in lines]
    print("Percentage under {}: {}".format(max_s, len([w for w in words if w < max_s])/len(words)))
    plt.hist(words, bins=[i for i in xrange(max(words))])
    plt.show()


if __name__ == "__main__":
    get_stats("data/train.small.en", 30)
    get_stats("data/train.small.vi", 30)
    get_stats("data/train.medium.en", 30)
    get_stats("data/train.medium.de", 30)
