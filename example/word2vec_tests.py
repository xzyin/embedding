#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
from collections import Counter
import codecs
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

'''
训练word2vec模型
'''

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

def train(vocab_dict, view_seqs):
    epoch = args["iter"]
    window_size = args["window_size"]
    batch_size = args["batch_size"]
    learn_rate = args["lr"]
    log_dir = args["log_dir"]
    embed_size = args["size"]
    num_sampled = args["num_sampled"]
    word2vec_model = Word2vecModel(vocab_size=len(vocab_dict)+1, embed_size=embed_size,
                                   num_sampled=num_sampled, learn_rate=learn_rate, log_dir=log_dir)
    word2vec_model.build_graph()
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            batch_count = 0
            k = 0
            batches_loss = []
            for context, target in generate_batch(window_size, batch_size, view_seqs):
                feed_dict = {word2vec_model.context_words: context,
                             word2vec_model.target_words: target}
                loss_batch, _ = sess.run([word2vec_model.loss,
                                          word2vec_model.optimizer],
                                         feed_dict=feed_dict)
                batches_loss.append(loss_batch)
                batch_count += context.shape[0]

                k += 1
                if k % 60 == 0:
                    logging.info("epoch: {}, batch count: {}, pair count: {}, loss: {:5.5f}"
                                 .format(i, k, batch_count, loss_batch))
            feed_dict = {word2vec_model.batches_loss: np.array(batches_loss)}
            average_loss, merge = sess.run([word2vec_model.average_loss,word2vec_model.merge], feed_dict=feed_dict)
            word2vec_model.train_writer.add_summary(merge, i)
            logging.info("Average loss at epoch {}, pair count {}, current loss {:5.5f}, learn rate {:3.3f}"
                         .format(i, batch_count, average_loss, learn_rate))
        dump(sess, word2vec_model, vocab_dict=vocab_dict)
        word2vec_model.train_writer.close()
    return sess, word2vec_model

def build_vocab(path, min_count, with_rating=False):
    counter = Counter()
    vocab_dict = dict()
    view_seqs = []
    index = 0
    # 生成词汇表字典
    with codecs.open(path, 'r', 'utf-8') as f:
        for line in f:
            counter.update(line.strip().split(" "))
            index += 1
            if index % 100000 == 0:
                logging.info("build vocabulary, load index:{}".format(index))

    for (key, value) in counter.items():
        if value >= min_count:
            vocab_dict[key] = len(vocab_dict) + 1
    f.close()
    # 处理用户观影序列成index 序列
    index = 0
    with codecs.open(path, 'r', 'utf-8') as f1:
        for line in f1:
            items = [str(vocab_dict.get(i)) for i in line.strip().split(" ") if i in vocab_dict.keys()]
            view_seqs.append(items)
            index += 1
            if index % 1000000 == 0:
                logging.info("filter minimum vocabulary, load index:{}".format(index))
    return vocab_dict, view_seqs

def generate_batch(window_size, batch_size, view_seqs):
    center_batch = np.zeros(batch_size, dtype=np.int32)
    target_batch = np.zeros([batch_size, 1])
    index = 0
    for items in view_seqs:
        for (i, center) in enumerate(items):
            size = np.random.randint(1, window_size)
            targets = [] + items[max(i - size, 0): i] + items[i + 1: min(i + size, len(items))]
            for target in targets:
                if index < batch_size:
                    center_batch[index], target_batch[index] = center, target
                    index += 1
                else:
                    yield (center_batch, target_batch)
                    center_batch = np.zeros(batch_size, dtype=np.long)
                    target_batch = np.zeros([batch_size, 1])
                    index = 0
    yield (center_batch[0:index], target_batch[0:index,])

def dump(sess, model, vocab_dict):
    output = open(args["output"], "w")
    sess.run(tf.global_variables_initializer())
    index_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    word_vector = sess.run(model.embed_matrix)
    for i, vec in enumerate(word_vector):
        if i in index_dict.keys():
            vid = index_dict.get(i)
            vector = " ".join([str(dim) for dim in vec])
            line = "{} {}\n".format(vid, vector)
            output.write(line)
    output.close()


def main():
    vocab_dict, view_seqs = build_vocab(args["input"], args["min_count"], False)
    logging.info("build vocabulary successful, vocabulary size: {}".format(len(vocab_dict)))
    train(vocab_dict, view_seqs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="Word2vec")
    ap.add_argument("--input", help="input path of the word sequences ")
    ap.add_argument("--iter", type=int, default=20, help="max iteration of the word2vec")
    ap.add_argument("--window_size", type=int, default=5, help="window size of the word2vec model")
    ap.add_argument("--batch_size", type=int, default=300, help="batch size of train model data")
    ap.add_argument("--lr", type=float, default=1.0, help="learn rate of the word2vec model")
    ap.add_argument("--size", type=int, default=128, help="dimensions size of the word embedding space")
    ap.add_argument("--min_count", type=int, default=10, help="minimum word frequency")
    ap.add_argument("--log_dir", help="directory of tensor board log")
    ap.add_argument("--num_sampled", type=int, default=64, help="num sampled of the word2vec model")
    ap.add_argument("--output", help="output path of the vector")
    args = vars(ap.parse_args())
    main()
