#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
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
        self.pairs_tensor = self.data_view.map(lambda x: tf.py_func(self._pair, [x], tf.int32), num_parallel_calls=5).prefetch(10000)
        self.pairs_batches = self.pairs_tensor.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))\
            .batch(self.batch_size).prefetch(10000)
        self.batches_iterator = self.pairs_tensor.make_initializable_iterator()
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

    def train(self, epoch, lsize, timeline_path=None):
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))) as sess:
            sess.run(tf.global_variables_initializer())
            total_loss = 0.0
            batch_cnt = 0
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for i in range(epoch):
                sess.run(self.batches_iterator.initializer)
                while True:
                    try:
                        if i == 0 and batch_cnt <= 3010 and batch_cnt >= 3000 and timeline is not None:
                            loss, _ = sess.run([self.loss, self.optimizer], options=options, run_metadata=run_metadata)
                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open(timeline_path + "_{}.json".format(batch_cnt), "w") as f:
                                f.write(chrome_trace)
                                f.flush()
                                f.close()
                        else:
                            loss, _ = sess.run([self.loss, self.optimizer])
                        batch_cnt += 1
                        total_loss += loss
                        if batch_cnt % lsize == 0:
                            logging.info("Epoch {}, batch count:{} loss {:5.5f}".format(i, batch_cnt, loss))
                    except tf.errors.OutOfRangeError:
                        batch_cnt = 0
                        batch_average = total_loss / batch_cnt
                        logging.info("Epoch {}, Average batch loss {:5.5f}".format(i, batch_average))
                        merge = sess.run(self.merge, feed_dict={self.average_batch_loss: batch_average})
                        self.train_writer.add_summary(merge, i)
                        total_loss = 0.0
                        break
            embedding_matrix = sess.run(self.embed_matrix)
            self.train_writer.add_graph(sess.graph)
        return embedding_matrix


class Word2vecModelRecordPipeline(object):

    def __init__(self, files_queue, batch_size, vocab_size, embed_size, num_sampled, learn_rate, log_dir):
        self.files_queue = files_queue
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_sampled = num_sampled
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self.lr = learn_rate
        self.log_dir = log_dir

    def _parse_function(self, serialize_string):
        feature_description = {
            'input': tf.VarLenFeature(dtype=tf.int64),
            'output': tf.VarLenFeature(dtype=tf.int64),
        }

        features = tf.io.parse_single_example(serialized=serialize_string, features=feature_description)
        input_sparse_tensor = features["input"]
        output_sparse_tensor = features["output"]
        input_dense_tensor = tf.sparse_tensor_to_dense(input_sparse_tensor)
        output_dense_tensor = tf.sparse_tensor_to_dense(output_sparse_tensor)
        input_tensor = tf.reshape(input_dense_tensor, shape=(self.batch_size, ))
        output_tensor = tf.reshape(output_dense_tensor, shape=(self.batch_size, 1))
        return (input_tensor, output_tensor)

    def _create_prepare_data(self):
        self.dataset_record = tf.data.TFRecordDataset(self.files_queue).prefetch(100000)
        self.parse_dataset = self.dataset_record.map(self._parse_function)
        self.dataset = self.parse_dataset.prefetch(100000)
        self.dataset_iterator = self.dataset.make_initializable_iterator()
        self.context, self.target = self.dataset_iterator.get_next()

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
        self._create_optimizer()
        self._create_loss_placeholder()
        self._create_summaries()

    def train(self, epoch, lsize, timeline_path=None):
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))) as sess:
            sess.run(tf.global_variables_initializer())
            total_loss = 0.0
            batch_cnt = 0
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for i in range(epoch):
                sess.run(self.dataset_iterator.initializer)
                while True:
                    try:
                        if i == 0 and batch_cnt <= 3010 and batch_cnt >= 3000 and timeline_path is not None:
                            loss, _ = sess.run([self.loss, self.optimizer], options=options, run_metadata=run_metadata)
                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            timeline_file_path = os.path.join(timeline_path, "word2vec_{}.json".format(batch_cnt))
                            with open(timeline_file_path, "w") as f:
                                f.write(chrome_trace)
                                f.flush()
                                f.close()
                        else:
                            loss, _ = sess.run([self.loss, self.optimizer])
                        batch_cnt += 1
                        total_loss += loss
                        if batch_cnt % lsize == 0:
                            logging.info("Epoch {}, batch count:{} loss {:5.5f}".format(i, batch_cnt, loss))
                    except tf.errors.OutOfRangeError:
                        batch_average = total_loss / batch_cnt
                        logging.info("Epoch {}, Average batch loss {:5.5f}".format(i, batch_average))
                        merge = sess.run(self.merge, feed_dict={self.average_batch_loss: batch_average})
                        self.train_writer.add_summary(merge, i)
                        total_loss = 0.0
                        batch_cnt = 0
                        break
            embedding_matrix = sess.run(self.embed_matrix)
            self.train_writer.add_graph(sess.graph)
        return embedding_matrix