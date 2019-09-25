#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from queue import Queue
from threading import Thread
import threading
import numpy as np
import logging
import argparse
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")

class SkipGramModel(object):

    def __init__(self):
        self._input_path = None
        self._model_path = None
        self._batch_size = None
        self._embedding_size = None
        self._num_sampled = None
        self._num_steps = None
        # 设置batch队列,训练过程和数据预处理异步
        self._batch_queue = Queue(maxsize=50)
        self._vocab_size = None
        self._lr = None
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self._vocab_dict = {}

    #设置迭代次数
    def set_num_steps(self, num_steps):
        self._num_steps = num_steps

    #设置输入路径
    def set_input_path(self, path):
        self._input_path = path

    #设置模型路径
    def set_model_path(self, path):
        self._model_path = path

    #设置batch大小
    def set_batch_size(self, size):
        self._batch_size = size

    #设置embedding size大小
    def set_embedding_size(self, size):
        self._embedding_size = size

    #设置采样数目
    def set_num_sample(self, size):
        self._num_sampled = size

    #设置最大迭代次数
    def set_max_iteration(self, size):
        self._num_steps = size

    #设置学习速率
    def set_learn_rate(self, lr):
        self._lr = lr

    def _build_pair_batch(self):
        raise NotImplementedError("not implement error of _build_pair_batch() in {}".format(self.__class__.__name__))

    def _train(self):
        raise NotImplementedError("not implement error of _train() in {}".format(self.__class__.__name__))

    def _build_vocab(self):
        raise NotImplementedError("not implement error of _build_vocab() in {}".format(self.__class__.__name__))

    def fit(self):
        self._build_graph()
        self._build_vocab()
        for i in range(self._num_steps):
            self._build_pair_batch()
            self._train()

    #context words存放了每个batch中上下文作为输入, target_words中存放了每个batch的label作为输出
    def _create_place_holders(self):
        with tf.name_scope("input_label"):
            self.context_words = tf.placeholder(tf.int32, shape=[None], name="context_words")
            self.target_words = tf.placeholder(tf.int32, shape=[None, 1], name="target_words")

    def _create_input_vector(self):
        with tf.name_scope("input_vector"):
            self.intput_vector = tf.Variable(tf.random_uniform([self._vocab_size, self._embedding_size],
                                                              -0.25, 0.25),
                                            name="intput_vector")
    def _create_loss(self):
        with tf.name_scope("loss"):
            self.context_vector = tf.nn.embedding_lookup(self.intput_vector, self.context_words, name="input_hidden")
            self.nce_weight = tf.Variable(tf.truncated_normal([self._vocab_size, self._embedding_size],
                                                              stddev=1.0 / (self._embedding_size ** 0.5)),
                                          name="nce_weight")
            self.nce_bias = tf.Variable(tf.zeros([self._vocab_size]), name="nce_bias")
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight,
                                                      biases=self.nce_bias,
                                                      labels=self.target_words,
                                                      inputs=self.context_vector,
                                                      num_sampled=self._num_sampled,
                                                      num_classes=self._vocab_size), name="loss")
    def _create_optimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self._lr).minimize(self.loss, global_step=self._global_step)

    def _build_graph(self):
        self._create_place_holders()
        self._create_input_vector()
        self._create_loss()
        self._create_optimizer()

class Word2vecSkipGram(object):
    def __init__(self):
        pass


class CfPair2vecSkipGramSingle(SkipGramModel):
    def __init__(self):
        self._slice_pair = []
        self._pair_count = None
        self._slice_size = None
        self._output_path = None
        SkipGramModel.__init__(self)

    '''
    (1) 建立词汇表并记录itemcf pair中所有pair的对数
    (2) 根据数据pair对的个数对整个数据集进行有效的切分
    '''
    def _build_vocab(self):
        self._pair_count = 0
        fin = open(self._input_path, "r")
        line = fin.readline()
        while line:
            words = line.strip().split(" ")
            for word in words:
                if word in self._vocab_dict.keys():
                    continue
                else:
                    self._vocab_dict[word] = len(self._vocab_dict)
            line = fin.readline()
            self._pair_count += 1
            if self._pair_count % 100000 == 0:
                logging.info("build vocab, index: {}".format(self._pair_count))
        slice_size = int(self._pair_count / self._slice_size)
        assert slice_size != 0
        for i in range(self._slice_size - 1):
            self._slice_pair.append((i * slice_size, (i + 1) * slice_size - 1))
        self._slice_pair.append((slice_size * (self._slice_size - 1), self._pair_count - 1))
        self._vocab_size = len(self._vocab_dict)
        self._index_dict = dict(zip(self._vocab_dict.values(), self._vocab_dict.keys()))
        logging.info("vocab size of the item2vec: {}".format(len(self._vocab_dict)))

    def _build_batches_once(self):
        fin = open(self._input_path)
        batch_size = 0
        index = 0
        line = fin.readline()
        input_batches = []
        output_batches = []
        input_batch = np.zeros(self._batch_size, dtype=np.int32)
        output_batch = np.zeros([self._batch_size, 1])
        while line:
            res = line.strip().split(" ")
            for r in res:
                for b in res:
                    if r in self._vocab_dict.keys() \
                            and b in self._vocab_dict.keys() \
                            and r != b:
                        input_batch[batch_size] = self._vocab_dict.get(r)
                        output_batch[batch_size] = self._vocab_dict.get(b)
                        batch_size += 1
                        if batch_size >= self._batch_size:
                            input_batches.append(input_batch)
                            output_batches.append(output_batch)
                            input_batch = np.zeros(batch_size, dtype=np.int32)
                            output_batch = np.zeros([batch_size, 1])
                            batch_size = 0
            line = fin.readline()
            index += 1
            if (index % 100000 == 0):
                logging.info("build slice batches index:{}".format(index))
        if batch_size > 0:
            input_batches.append(input_batch[0:batch_size])
            output_batches.append(output_batch[0:batch_size, ])
        logging.info("read slice batches success!")
        return zip(input_batches, output_batches)

    def _build_batches(self, start, end, startOps):
        fin = open(self._input_path)
        count = end - start + 1
        batch_size = 0
        index = 0
        fin.seek(startOps)
        line = fin.readline()
        input_batches = []
        output_batches = []
        input_batch = np.zeros(self._batch_size, dtype=np.int32)
        output_batch = np.zeros([self._batch_size, 1])
        while line and index < count:
            res = line.strip().split(" ")
            for r in res:
                for b in res:
                    if r in self._vocab_dict.keys() \
                            and b in self._vocab_dict.keys() \
                            and r != b:
                        input_batch[batch_size] = self._vocab_dict.get(r)
                        output_batch[batch_size] = self._vocab_dict.get(b)
                        batch_size += 1
                        if batch_size >= self._batch_size:
                            input_batches.append(input_batch)
                            output_batches.append(output_batch)
                            input_batch = np.zeros(batch_size, dtype=np.int32)
                            output_batch = np.zeros([batch_size, 1])
                            batch_size = 0
            line = fin.readline()
            index += 1
            if(index % 100000 == 0):
                logging.info("build slice batches index:{}".format(index))
        if batch_size > 0:
            input_batches.append(input_batch[0:batch_size])
            output_batches.append(output_batch[0:batch_size,])
        ops = fin.tell()
        fin.close()
        logging.info("read slice batches success!")
        return (zip(input_batches, output_batches), ops)

    def set_slice_size(self, size):
        self._slice_size = size

    def set_output_path(self, path):
        self._output_path = path

    def fit(self):
        self._build_vocab()
        self._build_graph()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))) as sess:
            sess.run(tf.global_variables_initializer())
            batches = self._build_batches_once()
            for iter in range(self._num_steps):
                total_loss = 0.0
                count = 0
                #for (i, (start, end)) in enumerate(self._slice_pair):
                    #(batches, tmpOps) = self._build_batches(start, end, ops)
                for (input_batch, output_batch) in batches:
                    feed_dict = {self.context_words: input_batch,
                                 self.target_words: output_batch}
                    loss_batch, _ = sess.run([self.loss, self.optimizer],
                                             feed_dict=feed_dict)
                    total_loss += loss_batch
                    count += 1
                logging.info("iteration {}, total loss:{}".format(iter, total_loss))
    def dump_input_vector(self):
        fout = open(self._output_path, "w")
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True,
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))) as sess:
            sess.run(tf.global_variables_initializer())
            input_vector = sess.run(self.intput_vector)
            for (index, vector) in enumerate(input_vector):
                res = []
                res.append(self._index_dict[index])
                for dim in vector:
                    res.append(str(dim))
                fout.write(" ".join(res) + "\n")



def main():
    model = CfPair2vecSkipGramSingle()
    model.set_num_steps(args["max_iter"])
    model.set_input_path(args["input"])
    model.set_embedding_size(args["embed_size"])
    model.set_slice_size(args["slice_size"])
    model.set_batch_size(args["batch_size"])
    model.set_num_sample(args["num_sample"])
    model.set_learn_rate(args["lr"])
    model.set_output_path(args["output"])
    model.fit()
    model.dump_input_vector()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_iter", default=20, help="max iteration of train")
    ap.add_argument("--input", default="/data/app/xuezhengyin/item2vec.action", help="input of the item2vec pair")
    ap.add_argument("--output", default="/data/app/xuezhengyin/item2vec.vector")
    ap.add_argument("--embed_size", default=300, help="embedding size of the video vector")
    ap.add_argument("--slice_size", default=20, help="slice size of the input data")
    ap.add_argument("--batch_size", default=1000, help="batch size of the model input")
    ap.add_argument("--num_sample", default=64, help="number sample of the word2vec model")
    ap.add_argument("--lr", default=1.0, help="learn rate of the word2vec model")
    args = vars(ap.parse_args())
    main()

