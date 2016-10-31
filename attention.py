from __future__ import division
from __future__ import print_function

from datetime import datetime
from data import data_iterator
from data import read_vocabulary

import tensorflow as tf
import numpy as np
import math
import os


class AttentionNN(object):
    def __init__(self, config, sess):
        self.sess          = sess
        self.hidden_size   = config.hidden_size
        self.num_layers    = config.num_layers
        self.batch_size    = config.batch_size
        self.max_size      = config.max_size
        self.dropout       = config.dropout
        self.epochs        = config.epochs
        self.s_nwords      = config.s_nwords
        self.t_nwords      = config.t_nwords
        self.show          = config.show
        self.minval        = config.minval
        self.maxval        = config.maxval
        self.lr_init       = config.lr_init
        self.max_grad_norm = config.max_grad_norm

        self.source_data_path  = config.source_data_path
        self.target_data_path  = config.target_data_path
        self.source_vocab_path = config.source_vocab_path
        self.target_vocab_path = config.target_vocab_path
        self.checkpoint_dir    = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Directory {} not found".format(self.checkpoint_dir))

        self.source = tf.placeholder(tf.int32, [self.batch_size, self.max_size], name="source")
        self.target = tf.placeholder(tf.int32, [self.batch_size, self.max_size], name="target")

        self.build_model()

    def build_model(self):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.lr = tf.Variable(self.lr_init, trainable=False, name="lr")

        with tf.variable_scope("encoder"):
            self.s_emb = tf.get_variable("embedding", shape=[self.s_nwords, self.hidden_size],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            if self.dropout > 0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        with tf.variable_scope("decoder"):
            self.t_emb = tf.get_variable("embedding", shape=[self.t_nwords, self.hidden_size],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            if self.dropout > 0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        with tf.variable_scope("proj"):
            self.proj_W = tf.get_variable("W", shape=[self.hidden_size, self.t_nwords],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))
            self.proj_b = tf.get_variable("b", shape=[self.t_nwords],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))

        with tf.variable_scope("attention"):
            self.v_a = tf.get_variable("v_a", shape=[self.hidden_size, 1],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))
            self.W_a = tf.get_variable("W_a", shape=[2*self.hidden_size, self.hidden_size],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))
            self.b_a = tf.get_variable("b_a", shape=[self.hidden_size],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))
            self.W_c = tf.get_variable("W_c", shape=[2*self.hidden_size, self.hidden_size],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))
            self.b_c = tf.get_variable("b_c", shape=[self.hidden_size],
                    initializer=tf.random_uniform_initializer(self.minval, self.maxval))

        # TODO: put this cpu?
        with tf.variable_scope("encoder"):
            source_xs = tf.nn.embedding_lookup(self.s_emb, self.source)
        with tf.variable_scope("decoder"):
            target_xs = tf.nn.embedding_lookup(self.t_emb, self.target)

        initial_state = self.encoder.zero_state(self.batch_size, tf.float32)
        s = initial_state
        encoder_hs = []
        with tf.variable_scope("encoder"):
            for t, x in enumerate(tf.split(1, self.max_size, source_xs)):
                x = tf.squeeze(x)
                if t > 0: tf.get_variable_scope().reuse_variables()
                hs = self.encoder(x, s)
                s = hs[1]
                h = hs[0]
                encoder_hs.append(h)

        decoder_hs = []
        # s is now final encoding hidden state
        with tf.variable_scope("decoder"):
            for t, x in enumerate(tf.split(1, self.max_size, target_xs)):
                x = tf.squeeze(x)
                if t > 0: tf.get_variable_scope().reuse_variables()
                hs = self.decoder(x, s)
                s = hs[1]
                h = hs[0]
                decoder_hs.append(h)

        attn_hs    = []
        encoder_hs = tf.pack(encoder_hs)
        with tf.variable_scope("attention"):
            for h_t in decoder_hs:
                scores = [tf.matmul(tf.tanh(tf.batch_matmul(tf.concat(1, [h_t, tf.squeeze(h_s)]),
                                                            self.W_a) + self.b_a),
                                                            self.v_a)
                          for h_s in tf.split(0, self.max_size, encoder_hs)]
                a_t    = tf.nn.softmax(tf.transpose(tf.squeeze(tf.pack(scores))))
                a_t    = tf.expand_dims(a_t, 2)
                c_t    = tf.squeeze(tf.batch_matmul(tf.transpose(encoder_hs, perm=[1,2,0]), a_t))
                h_t    = tf.tanh(tf.batch_matmul(tf.concat(1, [h_t, c_t]), self.W_c) + self.b_c)
                attn_hs.append(h_t)

        logits     = []
        self.probs = []
        with tf.variable_scope("proj"):
            for h_t in attn_hs:
                logit  = tf.batch_matmul(h_t, self.proj_W) + self.proj_b
                prob   = tf.nn.softmax(logit)
                logits.append(logit)
                self.probs.append(prob)

        logits     = logits[:-1]
        targets    = tf.split(1, self.max_size, self.target)[1:]
        weights    = [tf.ones([self.batch_size]) for _ in xrange(self.max_size - 1)]
        self.loss  = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
        self.optim = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        inc = self.global_step.assign_add(1)

        # TODO: renormalize gradients instead of clip
        opt = tf.train.GradientDescentOptimizer(self.lr)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gvs = opt.compute_gradients(self.loss, [v for v in trainable_vars],
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        clipped_gvs = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g,v in gvs]
        with tf.control_dependencies([inc]):
            self.optim = opt.apply_gradients(clipped_gvs)

        self.sess.run(tf.initialize_all_variables())
        tf.train.write_graph(self.sess.graph_def, "./logs", "attn_graph.pb", False)
        tf.scalar_summary("loss", self.loss)
        self.saver = tf.train.Saver()

    def get_model_name(self):
        date = datetime.now()
        return "attention-{}-{}-{}".format(date.month, date.day, date.hour)

    def train(self):
        data_size = len(open(self.source_data_path).readlines())
        N = int(math.ceil(data_size/self.batch_size))
        merged_sum = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/{}".format(self.get_model_name()),
                                        self.sess.graph)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar("Train", max=self.epochs*N)

        for epoch in xrange(self.epochs):
            iterator = data_iterator(self.source_data_path,
                                     self.target_data_path,
                                     read_vocabulary(self.source_vocab_path),
                                     read_vocabulary(self.target_vocab_path),
                                     self.max_size, self.batch_size)
            i = 0
            total_loss = 0.
            for dsource, dtarget in iterator:
                if self.show: bar.next()
                outputs = self.sess.run([self.loss, self.global_step, self.optim, merged_sum],
                                        feed_dict={self.source: dsource,
                                                   self.target: dtarget})
                loss = outputs[0]
                total_loss += loss
                if i % 2 == 0:
                    writer.add_summary(outputs[-1], N*epoch + i)
                if i % 100 == 0:
                    print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, i, loss))
                i += 1

            step = outputs[1]
            self.saver.save(self.sess,
                            os.path.join(self.checkpoint_dir, self.get_model_name()),
                            global_step=step.astype(int))
            # without dropout after with, with dropout after 8
            if epoch > 5:
                self.lr_init = self.lr_init/2
                self.lr.assign(self.lr_init).eval()
        if self.show:
            bar.finish()
            print("")
        print("Loss: {}".format(total_loss/N))

    def test(self, source_data_path, target_data_path):
        data_size = len(open(source_data_path).readlines())
        N = int(math.ceil(data_size/self.batch_size))

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar("Test", max=N)

        iterator = data_iterator(source_data_path,
                                 target_data_path,
                                 read_vocabulary(self.source_vocab_path),
                                 read_vocabulary(self.target_vocab_path),
                                 self.max_size, self.batch_size)

        total_loss = 0
        for dsource, dtarget in iterator:
            if self.show: bar.next()
            loss = self.sess.run([self.loss],
                                 feed_dict={self.source: dsource,
                                            self.target: dtarget})
            total_loss += loss[0]

        if self.show:
            bar.finish()
            print("")
        total_loss /= N
        perplexity = np.exp(total_loss)
        return perplexity

    def load(self):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("[!] No checkpoint found")
