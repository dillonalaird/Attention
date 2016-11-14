from __future__ import division
from __future__ import print_function

from attention import AttentionNN
from data import read_vocabulary
from bleu.length_analysis import process_files

import os
import time
import pprint
import random
import numpy as np
import tensorflow as tf


pp = pprint.PrettyPrinter().pprint

flags = tf.app.flags

flags.DEFINE_integer("max_size", 30, "Maximum sentence length [30]")
flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch [128]")
flags.DEFINE_integer("random_seed", 123, "Value of random seed [123]")
flags.DEFINE_integer("epochs", 12, "Number of epochs to run [10]")
flags.DEFINE_integer("hidden_size", 1024, "Size of hidden units [1024]")
flags.DEFINE_integer("emb_size", 256, "Size of embedding dimension [256]")
flags.DEFINE_integer("num_layers", 4, "Depth of RNNs [4]")
flags.DEFINE_float("dropout", 0.2, "Dropout probability [0.2]")
flags.DEFINE_float("minval", -0.1, "Minimum value for initialization [-0.1]")
flags.DEFINE_float("maxval", 0.1, "Maximum value for initialization [0.1]")
flags.DEFINE_float("lr_init", 1.0, "Initial learning rate [1.0]")
flags.DEFINE_float("max_grad_norm", 5.0, "Maximum gradient cutoff [5.0]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory [checkpoints]")
flags.DEFINE_string("dataset", "small", "Dataset to use [small]")
flags.DEFINE_string("name", "default", "Model name [default]")
flags.DEFINE_string("sample", None, "Sample from dataset [None]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for training [False]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


class debug:
    source_data_path       = "data/train.debug.en"
    target_data_path       = "data/train.debug.vi"
    source_vocab_path      = "data/vocab.small.en"
    target_vocab_path      = "data/vocab.small.vi"
    valid_source_data_path = "data/test.debug.en"
    valid_target_data_path = "data/test.debug.vi"
    test_source_data_path  = "data/test.debug.en"
    test_target_data_path  = "data/test.debug.vi"


class small:
    source_data_path       = "data/train.small.en"
    target_data_path       = "data/train.small.vi"
    valid_source_data_path = "data/valid.small.en"
    valid_target_data_path = "data/valid.small.en"
    source_vocab_path      = "data/vocab.small.en"
    target_vocab_path      = "data/vocab.small.vi"


class medium:
    source_data_path  = "data/train.medium.en"
    target_data_path  = "data/train.medium.de"
    source_vocab_path = "data/vocab.medium.en"
    target_vocab_path = "data/vocab.medium.de"


def get_bleu_score(samples, target_file):
    hyp_file = "hyp" + str(int(time.time()))
    with open(hyp_file, "w") as f:
        for sample in samples:
            for s in sample:
                if s == "</s>": break
                f.write(" " + s)
            f.write("\n")

    process_files(hyp_file, target_file)
    os.remove(hyp_file)


def main(_):
    config = FLAGS
    if config.dataset == "small":
        data_config = small
    elif config.dataset == "medium":
        data_config = medium
    elif config.dataset == "debug":
        data_config = debug
    else:
        raise Exception("[!] Unknown dataset {}".format(config.dataset))

    config.source_data_path  = data_config.source_data_path
    config.target_data_path  = data_config.target_data_path
    config.source_vocab_path = data_config.source_vocab_path
    config.target_vocab_path = data_config.target_vocab_path

    s_nwords = len(read_vocabulary(config.source_vocab_path))
    t_nwords = len(read_vocabulary(config.target_vocab_path))

    config.s_nwords = s_nwords
    config.t_nwords = t_nwords
    pp(config.__dict__["__flags"])
    with tf.Session() as sess:
        attn = AttentionNN(config, sess)
        if config.sample:
            attn.load()
            samples = attn.sample(config.sample)
            for s in samples: print(" ".join(s))
        else:
            if not config.is_test:
                attn.run(data_config.valid_source_data_path,
                         data_config.valid_target_data_path)
            else:
                attn.load()
                loss = attn.test(data_config.test_source_data_path,
                                 data_config.test_target_data_path)
                print("[Test] [Loss: {}] [Perplexity: {}]".format(loss, np.exp(loss)))
                samples = attn.sample(data_config.test_source_data_path)
                get_bleu_score(samples, data_config.test_target_data_path)


if __name__ == "__main__":
    tf.app.run()
