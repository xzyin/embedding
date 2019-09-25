#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
class Word2vecModel(object):
    def __init__(self, vocab_size, embed_size, num_sampled, learn_rating, log_dir):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self.lr = learn_rating
        self.log_dir = log_dir

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.context_words = tf.placeholder(tf.int32, shape=[None], name="context_words")
            self.target_words = tf.placeholder(tf.int32, shape=[None, 1], name="target_words")

    def _create_embeddding(self):
        with tf.name_scope("embed"):
            self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size,
                                                              self.embed_size], -1.0, 1.0),
                                                              name='embed_matrix')

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.embed_context = tf.nn.embedding_lookup(self.embed_matrix, self.context_words, name='embed')
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                              stddev=1.0 / (self.embed_size ** 0.5)),
                                          name='nce_weight')
            self.nce_bias = tf.Variable(tf.zeros(self.vocab_size), name='nce_bias')
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight,
                                                      biases=self.nce_bias,
                                                      labels=self.target_words,
                                                      inputs=self.embed_context,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size), name='loss')

    def _carete_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                             global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.merge = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(logdir=self.log_dir)

    def set_learn_rate(self, learn_rate):
        self.lr = learn_rate

    def build_graph(self):
        self._create_placeholders()
        self._create_embeddding()
        self._create_loss()
        self._carete_optimizer()
        self._create_summaries()

