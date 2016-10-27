from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class AttentionNN(object):
    def __init__(self, config, sess):
        self.sess = sess
        self.batch_size = config.batch_size
        self.max_size   = config.max_size
        self.epochs     = config.epochs

        self.build_model()

    def build_model(self):
        pass

    def train(self):
        pass
