from __future__ import division
from __future__ import print_function

from attention import AttentionNN

import tensorflow as tf
import numpy as np
import random


class config:
    max_size = 30
    batch_size = 10
    random_seed = 123
    epochs = 1
    hidden_size = 16
    num_layers = 1
    checkpoint_dir = "checkpoints"
    dataset = "small"
    is_test = False
    show = True
    nwords = 10

    source_data_path  = "data/train.small.en"
    target_data_path  = "data/train.small.vi"
    source_vocab_path = "data/vocab.small.en"
    target_vocab_path = "data/vocab.small.vi"


tf.set_random_seed(config.random_seed)
random.seed(config.random_seed)

with tf.Session() as sess:
    source = np.random.randint(config.nwords, size=(config.batch_size, config.max_size))
    target = np.random.randint(config.nwords, size=(config.batch_size, config.max_size))

    attn = AttentionNN(config, sess)
