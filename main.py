from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf
from attention import AttentionNN


flags = tf.app.flags

flags.DEFINE_integer("max_size", 30, "Maximum sentence length")
flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch")
flags.DEFINE_integer("random_seed", 123, "Value of random seed")
flags.DEFINE_integer("epochs", 10, "Number of epochs to run")
flags.DEFINE_string("input_data_path", "data/train.small.en", "Input data path")
flags.DEFINE_string("output_data_path", "data/train.small.vi", "Output data path")
flags.DEFINE_string("input_vocab", "data/vocab.small.en", "Input vocab path")
flags.DEFINE_string("output_vocab", "data/vocab.small.vi", "Output vocab path")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


def main(_):
    config = FLAGS
    with tf.Session() as sess:
        attn = AttentionNN(config, sess)
        attn.train()



if __name__ == "__main__":
    tf.app.run()
