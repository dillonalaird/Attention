from __future__ import division
from __future__ import print_function

from datetime import datetime
from data import data_iterator_len
from data import data_iterator
from data import read_vocabulary

import tensorflow as tf
import numpy as np
import sys
import os


class AttentionNN(object):
    def __init__(self, config, sess):
        self.sess          = sess
        self.hidden_size   = config.hidden_size
        self.num_layers    = config.num_layers
        self.batch_size    = config.batch_size
        self.max_size      = config.max_size
        self.init_dropout  = config.dropout
        self.epochs        = config.epochs
        self.s_nwords      = config.s_nwords
        self.t_nwords      = config.t_nwords
        self.minval        = config.minval
        self.maxval        = config.maxval
        self.lr_init       = config.lr_init
        self.max_grad_norm = config.max_grad_norm
        self.dataset       = config.dataset
        self.emb_size      = config.emb_size
        self.is_test       = config.is_test
        self.name          = config.name

        self.source_data_path  = config.source_data_path
        self.target_data_path  = config.target_data_path
        self.source_vocab_path = config.source_vocab_path
        self.target_vocab_path = config.target_vocab_path
        self.checkpoint_dir    = config.checkpoint_dir

        self.train_iters = 0

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Directory {} not found".format(self.checkpoint_dir))

        self.source    = tf.placeholder(tf.int32, [self.batch_size, self.max_size], name="source")
        self.target    = tf.placeholder(tf.int32, [self.batch_size, self.max_size], name="target")
        self.target_ws = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        self.dropout   = tf.placeholder(tf.float32, name="dropout")

        self.build_variables()
        self.build_model()

    def build_variables(self):
        self.lr = tf.Variable(self.lr_init, trainable=False, name="lr")
        initializer = tf.random_uniform_initializer(self.minval, self.maxval)

        with tf.variable_scope("encoder"):
            self.s_emb = tf.get_variable("embedding", shape=[self.s_nwords, self.emb_size],
                                         initializer=initializer)
            self.s_proj_W = tf.get_variable("s_proj_W", shape=[self.emb_size, self.hidden_size],
                                            initializer=initializer)
            self.s_proj_b = tf.get_variable("s_proj_b", shape=[self.hidden_size],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        with tf.variable_scope("decoder"):
            self.t_emb = tf.get_variable("embedding", shape=[self.t_nwords, self.emb_size],
                                         initializer=initializer)
            self.t_proj_W = tf.get_variable("t_proj_W", shape=[self.emb_size, self.hidden_size],
                                            initializer=initializer)
            self.t_proj_b = tf.get_variable("t_proj_b", shape=[self.hidden_size],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

            # projection
            self.proj_W = tf.get_variable("W", shape=[self.hidden_size, self.emb_size],
                                          initializer=initializer)
            self.proj_b = tf.get_variable("b", shape=[self.emb_size],
                                          initializer=initializer)
            self.proj_Wo = tf.get_variable("Wo", shape=[self.emb_size, self.t_nwords],
                                           initializer=initializer)
            self.proj_bo = tf.get_variable("bo", shape=[self.t_nwords],
                                           initializer=initializer)

            # attention
            self.v_a = tf.get_variable("v_a", shape=[self.hidden_size, 1],
                                       initializer=initializer)
            self.W_a = tf.get_variable("W_a", shape=[2*self.hidden_size, self.hidden_size],
                                       initializer=initializer)
            self.b_a = tf.get_variable("b_a", shape=[self.hidden_size],
                                       initializer=initializer)
            self.W_c = tf.get_variable("W_c", shape=[2*self.hidden_size, self.hidden_size],
                                       initializer=initializer)
            self.b_c = tf.get_variable("b_c", shape=[self.hidden_size],
                                       initializer=initializer)

    def build_model(self):
        with tf.variable_scope("encoder"):
            source_xs = tf.nn.embedding_lookup(self.s_emb, self.source)
            source_xs = tf.split(1, self.max_size, source_xs)
        with tf.variable_scope("decoder"):
            target_xs = tf.nn.embedding_lookup(self.t_emb, self.target)
            target_xs = tf.split(1, self.max_size, target_xs)

        initial_state = self.encoder.zero_state(self.batch_size, tf.float32)
        s = initial_state
        encoder_hs = []
        with tf.variable_scope("encoder"):
            for t in xrange(self.max_size):
                x = tf.squeeze(source_xs[t], [1])
                x = tf.matmul(x, self.s_proj_W) + self.s_proj_b
                if t > 0: tf.get_variable_scope().reuse_variables()
                hs = self.encoder(x, s)
                s = hs[1]
                h = hs[0]
                encoder_hs.append(h)
        encoder_hs = tf.pack(encoder_hs)

        logits = []
        probs = []
        with tf.variable_scope("decoder"):
            x = tf.squeeze(target_xs[0], [1])
            for t in xrange(self.max_size):
                if not self.is_test or t == 0:
                    x = tf.squeeze(target_xs[t], [1])
                x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
                if t > 0: tf.get_variable_scope().reuse_variables()
                s, logit, prob = self.decode_attention(t, x, s, encoder_hs)
                logits.append(logit)
                probs.append(prob)
                if self.is_test:
                    x = tf.cast(tf.argmax(prob, 1), tf.int32)
                    x = tf.nn.embedding_lookup(self.t_emb, x)

        logits     = logits[:-1]
        targets    = tf.split(1, self.max_size, self.target)[1:]
        #weights    = [tf.ones([self.batch_size]) for _ in xrange(self.max_size - 1)]
        self.loss  = tf.nn.seq2seq.sequence_loss(logits, targets, self.target_ws)
        self.probs = tf.transpose(tf.pack(probs), [1, 0, 2])

        self.optim = tf.contrib.layers.optimize_loss(self.loss, None,
                self.lr_init, "SGD", clip_gradients=5.,
                summaries=["learning_late", "loss", "gradient_norm"])

        tf.initialize_all_variables.run()
        self.saver = tf.train.Saver()

    def decode_attention(self, t, x, s, encoder_hs):
        hs  = self.decoder(x, s)
        s   = hs[1]
        h_t = hs[0]

        #scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_size, encoder_hs)]
        #scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.mul(encoder_hs, h_t), 2)
        a_t    = tf.nn.softmax(tf.transpose(scores))
        a_t    = tf.expand_dims(a_t, 2)
        c_t    = tf.batch_matmul(tf.transpose(encoder_hs, perm=[1,2,0]), a_t)
        c_t    = tf.squeeze(c_t, [2])
        h_tld  = tf.tanh(tf.matmul(tf.concat(1, [h_t, c_t]), self.W_c) + self.b_c)

        oemb  = tf.matmul(h_tld, self.proj_W) + self.proj_b
        logit = tf.matmul(oemb, self.proj_Wo) + self.proj_bo
        prob  = tf.nn.softmax(logit)
        return s, logit, prob

    def get_model_name(self):
        date = datetime.now()
        return "{}-{}-{}-{}-{}".format(self.name, self.dataset, date.month, date.day, date.hour)

    def train(self, epoch, merged_sum, writer):
        #if epoch > 5:
        #    self.lr_init = self.lr_init/2
        #    self.lr.assign(self.lr_init).eval()

        total_loss = 0.
        i = 0
        iterator = data_iterator_len(self.source_data_path,
                                     self.target_data_path,
                                     read_vocabulary(self.source_vocab_path),
                                     read_vocabulary(self.target_vocab_path),
                                     self.max_size, self.batch_size)
        for dsource, slen, dtarget, tlen in iterator:
            target_ws = [1.]*tlen + [0.]*(self.max_size - tlen)
            outputs = self.sess.run([self.loss, self.lr, self.optim, merged_sum],
                                    feed_dict={self.source: dsource,
                                               self.target: dtarget,
                                               self.target_ws: target_ws,
                                               self.dropout: self.init_dropout})
            loss = outputs[0]
            lr   = outputs[1]
            itr  = self.train_iters*epoch + i
            total_loss += loss
            if itr % 2 == 0:
                writer.add_summary(outputs[-1], itr)
            if itr % 10 == 0:
                print("[Train] [Time: {}] [Epoch: {}] [Iteration: {}] [lr: {}] [Loss: {}] [Perplexity: {}]"
                      .format(datetime.now(), epoch, itr, lr, loss, np.exp(loss)))
                sys.stdout.flush()
            i += 1
        self.train_iters = i
        return total_loss/i


    def test(self, source_data_path, target_data_path):
        iterator = data_iterator(source_data_path,
                                 target_data_path,
                                 read_vocabulary(self.source_vocab_path),
                                 read_vocabulary(self.target_vocab_path),
                                 self.max_size, self.batch_size)

        total_loss = 0
        i = 0
        for dsource, dtarget in iterator:
            loss, = self.sess.run([self.loss],
                                 feed_dict={self.source: dsource,
                                            self.target: dtarget,
                                            self.dropout: 0.0})
            total_loss += loss
            self.losses.append(loss)
            i += 1

        total_loss /= i
        return total_loss

    def sample(self, source_data_path):
        source_vocab = read_vocabulary(self.source_vocab_path)
        target_vocab = read_vocabulary(self.target_vocab_path)
        inv_target_vocab = {v:k for k,v in target_vocab.iteritems()}
        iterator = data_iterator(source_data_path,
                                 source_data_path,
                                 source_vocab,
                                 target_vocab,
                                 self.max_size, self.batch_size)
        samples = []
        for dsource, _ in iterator:
            dtarget = [[target_vocab["<s>"]] + [target_vocab["<pad>"]]*(self.max_size-1)]
            dtarget = dtarget*self.batch_size
            probs, = self.sess.run([self.probs],
                                   feed_dict={self.source: dsource,
                                              self.target: dtarget,
                                              self.dropout: 0.0})
            for b in xrange(self.batch_size):
                samples.append([inv_target_vocab[np.argmax(p)] for p in probs[b]])

        return samples

    def run(self, valid_source_data_path, valid_target_data_path):
        merged_sum = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/{}".format(self.get_model_name()),
                                        self.sess.graph)

        best_valid_loss = float("inf")
        for epoch in xrange(self.epochs):
            train_loss = self.train(epoch, merged_sum, writer)
            valid_loss = self.test(valid_source_data_path, valid_target_data_path)
            print("[Train] [Avg. Loss: {}] [Avg. Perplexity: {}]".format(train_loss, np.exp(train_loss)))
            print("[Valid] [Loss: {}] [Perplexity: {}]".format(valid_loss, np.exp(valid_loss)))
            if epoch == 0 or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.name))


    def load(self):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("[!] No checkpoint found")
