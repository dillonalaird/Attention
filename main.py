from __future__ import division
from __future__ import print_function

from attention import AttentionNN
from data import read_vocabulary

import random
import tensorflow as tf


flags = tf.app.flags

flags.DEFINE_integer("max_size", 30, "Maximum sentence length [30]")
flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch [128]")
flags.DEFINE_integer("random_seed", 123, "Value of random seed [123]")
flags.DEFINE_integer("epochs", 10, "Number of epochs to run [10]")
flags.DEFINE_integer("hidden_size", 1024, "Size of hidden units [1024]")
flags.DEFINE_integer("num_layers", 4, "Depth of RNNs [4]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory [checkpoints]")
flags.DEFINE_string("dataset", "small", "Dataset to use [small]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for training [False]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


class small:
    source_data_path  = "data/train.small.en"
    target_data_path  = "data/train.small.vi"
    source_vocab_path = "data/vocab.small.en"
    target_vocab_path = "data/vocab.small.vi"


class medium:
    source_data_path  = "data/train.medium.en"
    target_data_path  = "data/train.medium.de"
    source_vocab_path = "data/vocab.medium.en"
    target_vocab_path = "data/vocab.medium.vi"


def main(_):
    config = FLAGS
    if config.dataset == "small":
        data_config = small
    elif config.dataset == "medium":
        data_config = medium
    else:
        raise Exception("[!] Unknown dataset {}".format(config.dataset))

    config.source_data_path  = data_config.source_data_path
    config.target_data_path  = data_config.target_data_path
    config.source_vocab_path = data_config.source_vocab_path
    config.source_vocab_path = data_config.source_vocab_path

    nvocab = len(read_vocabulary(config.source_vocab_path))

    config.nvocab = nvocab
    with tf.Session() as sess:
        attn = AttentionNN(config, sess)
        attn.train()



if __name__ == "__main__":
    tf.app.run()
