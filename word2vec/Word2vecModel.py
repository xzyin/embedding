#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
class Word2vecModel(object):

    def __init__(self, vocab_size, embed_size, num_sampled, learn_rate, log_dir):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self.lr = learn_rate
        self.log_dir = log_dir

    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.context_words = tf.placeholder(tf.int32, shape=[None], name="context_words")
            self.target_words = tf.placeholder(tf.int32, shape=[None, 1], name="target_words")
            self.batches_loss = tf.placeholder(tf.float32, shape=[None], name="batches_loss")

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

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                                 global_step=self.global_step)

    def _create_average_loss(self):
        with tf.name_scope("average_loss"):
            self.average_loss = tf.reduce_mean(self.batches_loss, name="loss_mean")

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.average_loss)
            tf.summary.histogram("histogram loss", self.average_loss)
            self.merge = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(logdir=self.log_dir)

    def build_graph(self):
        self._create_placeholders()
        self._create_embeddding()
        self._create_loss()
        self._create_optimizer()
        self._create_average_loss()
        self._create_summaries()

class Word2vecModelPipeline(object):

    def __init__(self, view_seqs, window_size, batch_size, vocab_size, embed_size, num_sampled, learn_rate, log_dir):
        self.view_seqs = view_seqs
        self.window_size = window_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self.lr = learn_rate
        self.log_dir = log_dir

    def _pair(self, items_bytes):
        items = str(items_bytes, 'utf-8').strip().split()
        pairs = list()
        for (i, center) in enumerate(items):
            size = np.random.randint(1, self.window_size)
            target_string = [] + items[max(i - size, 0): i] + items[i + 1: min(i + size, len(items))]
            for target in target_string:
                pairs.append([np.int32(center), np.int32(target)])
        return np.reshape(pairs, newshape=(-1, 2))

    def _create_prepare_data(self):
        self.data_view = tf.data.TextLineDataset.from_tensor_slices(self.view_seqs)
        self.pairs_tensor = self.data_view.map(lambda x: tf.py_func(self._pair, [x], tf.int32))
        self.pairs_batches = self.pairs_tensor.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))\
            .batch(self.batch_size)
        self.batches_iterator = self.pairs_batches.make_initializable_iterator()
        self.batch_pair = self.batches_iterator.get_next()
        self.context = self.batch_pair[:,0]
        self.target = tf.reshape(self.batch_pair[:,1], [-1, 1])

    def _create_embeddding(self):
        with tf.name_scope("embed"):
            self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size,
                                                              self.embed_size], -1.0, 1.0),
                                                              name='embed_matrix')

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.embed_context = tf.nn.embedding_lookup(self.embed_matrix, self.context, name='embed')
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                              stddev=1.0 / (self.embed_size ** 0.5)),
                                          name='nce_weight')
            self.nce_bias = tf.Variable(tf.zeros(self.vocab_size), name='nce_bias')
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight,
                                                      biases=self.nce_bias,
                                                      labels=self.target,
                                                      inputs=self.embed_context,
                                                      num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size), name='loss')

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                                 global_step=self.global_step)

    def _create_loss_placeholder(self):
        with tf.name_scope("average_batch_loss"):
            self.average_batch_loss = tf.placeholder(tf.float64)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.average_batch_loss)
            self.merge = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(logdir=self.log_dir)

    def build_graph(self):
        self._create_prepare_data()
        self._create_embeddding()
        self._create_loss()
        self._create_embeddding()
        self._create_loss()
        self._create_optimizer()
        self._create_loss_placeholder()
        self._create_summaries()

    def train(self, epoch, lsize):
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))) as sess:
            sess.run(tf.global_variables_initializer())
            total_loss = 0.0
            batch_cnt = 0
            for i in range(epoch):
                sess.run(self.batches_iterator.initializer)
                while True:
                    try:
                        loss, _ = sess.run([self.loss, self.optimizer])
                        batch_cnt += 1
                        total_loss += loss
                        if batch_cnt % lsize == 0:
                            logging.info("Epoch {}, Batches Loss {:5.5f}".format(i, loss))
                    except tf.errors.OutOfRangeError:
                        batch_average = total_loss / batch_cnt
                        logging.info("Epoch {}, Average Batch Loss {:5.5f}".format(i, batch_average))
                        merge = sess.run(self.merge, feed_dict={self.average_batch_loss: batch_average})
                        self.train_writer.add_summary(merge, i)
                        total_loss = 0.0
                        break
            embedding_matrix = sess.run(self.embed_matrix)
        return embedding_matrix
