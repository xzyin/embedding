#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs
import numpy as np
import tensorflow as tf
import threading
from multiprocessing import Pool
from collections import Counter
import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

"""
统计序列中词汇的词频个数
如果词频个数小于最小词频对该词汇进行过滤
过滤后得到所有的词汇序列
"""
class Word2vecTokenizer(object):

    @staticmethod
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

    @staticmethod
    def transfer(block, vocab_dict, is_int=True):
        sequences = []
        for index, line in enumerate(block):
            items = [str(vocab_dict.get(i)) for i in line.strip().split(" ") if i in vocab_dict.keys()]
            if len(items) >= 2:
                if(is_int):
                    sequences.append(items)
                else:
                    sequences.append(" ".join(items))
            if index % 100000 == 0:
                logging.info("transfer sequeneces:{}, in pid:{}".format(index, os.getpid()))
        return sequences

    @staticmethod
    def block_vocab_counter(block):
        counter = Counter()
        for line in block:
            counter.update(line.strip().split(" "))
        return counter

    @staticmethod
    def build_vocab_threading(path, thread, min_count, is_int, with_rating=True):
        sequences = []
        vocab_dict = dict()
        counter = Counter()
        lines = codecs.open(path, 'r', 'utf-8').readlines()
        all_len = len(lines)
        block_len = int(all_len / thread)
        blocks = []
        for i in range(thread):
            start_offset = block_len * i
            end_offset = block_len * (i + 1)
            if i + 1 == thread:
                end_offset = all_len
            blocks.append(lines[start_offset:end_offset])
        with Pool(thread) as pool:
            for block_counter in pool.imap_unordered(Word2vecTokenizer.block_vocab_counter, blocks):
                counter.update(block_counter)
            for (key, value) in counter.items():
                if value >= min_count:
                    vocab_dict[key] = len(vocab_dict) + 1
        logging.info("build vocabulary success, vocabulary size:{}".format(len(vocab_dict)))
        pool.join()
        pool.close()
        with Pool(thread) as pool:
            results = [pool.apply_async(Word2vecTokenizer.transfer, (block, vocab_dict, is_int)) for block in blocks]
            pool.close()
            pool.join()
        for result in results:
            sequences.extend(result.get())
        logging.info("build sequence success, sequence lenth: {}".format(len(sequences)))
        return vocab_dict, sequences

    @staticmethod
    def generate_sequence_pair(items, window_size):
        centers_result = []
        targets_result = []
        for (i, center) in enumerate(items):
            size = np.random.randint(1, window_size)
            targets = [] + items[max(i - size, 0): i] + items[i + 1: min(i + size, len(items))]
            for target in targets:
                centers_result.append(np.int64(center))
                targets_result.append(np.int64(target))
        return (centers_result, targets_result)

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def serialize_pair_batches(input_words, output_words):
        feature = {
            'input': Word2vecTokenizer._int64_feature(input_words),
            'output': Word2vecTokenizer._int64_feature(output_words),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def generate_block_pair(block, window_size, batch_size, store_size, tf_record_path, start, thread_size):
        # 记录文件中存放了多少个batch
        batch_count = 0
        # 记录batch里面存放了多少个pair
        pair_count = 0
        # 记录文件名后缀
        suffix=start
        # 记录每个batch的输入
        centers = np.zeros(batch_size, dtype=np.int64)
        # 记录每个batch的输出
        targets = np.zeros((batch_size, 1), dtype=np.int64)
        # 文件名称
        path = "{}_{}.tfrecord".format(tf_record_path, suffix)
        writer = tf.python_io.TFRecordWriter(path)
        for items in block:
            #生成每个序列的pair
            (center_result, target_result) = Word2vecTokenizer.generate_sequence_pair(items, window_size)
            for (center, target) in zip(center_result, target_result):
                if pair_count < batch_size:
                    centers[pair_count] = center
                    targets[pair_count] = target
                    pair_count += 1
                else:
                    #写入pair
                    serial_pair = Word2vecTokenizer.serialize_pair_batches(centers.flatten(), centers.flatten())
                    writer.write(serial_pair)
                    #清空pair
                    centers = np.zeros(batch_size, dtype=np.int64)
                    targets = np.zeros((batch_size, 1), dtype=np.int64)
                    pair_count = 0
                    batch_count += 1
                    if batch_count > store_size:
                        writer.close()
                        suffix += thread_size
                        path = "{}_{}.tfrecord".format(tf_record_path, suffix)
                        writer = tf.python_io.TFRecordWriter(path)
                        batch_count = 0
        writer.close()


    '''
    用于构建TensorFlow 的TFRecord格式
    '''
    @staticmethod
    def build_batches_pair_tf_record(view_seqs, window_size, thread, batch_size, store_size, store_path):
        all_len = len(view_seqs)
        block_len = int(all_len / thread)
        blocks = []
        for i in range(thread):
            start_offset = block_len * i
            end_offset = block_len * (i + 1)
            if i + 1 == thread:
                end_offset = all_len
            blocks.append(view_seqs[start_offset:end_offset])
        with Pool(thread) as pool:
            [pool.apply_async(Word2vecTokenizer.generate_block_pair, \
                                       (block,
                                        window_size,
                                        batch_size,
                                        store_size,
                                        store_path,
                                        index,
                                        thread)) for (index, block) in enumerate(blocks)]
            pool.close()
            pool.join()

    """
    根据window size大小和batch_size大小生成训练数据集
    """
    @staticmethod
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

    @staticmethod
    def generate_batch_queue(window_size, batch_size, view_sequence, queue):
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        index = 0
        for items in view_sequence:
            for (i, center) in enumerate(items):
                size = np.random.randint(1, window_size)
                targets = [] + items[max(i - size, 0): i] + items[i + 1: min(i + size, len(items))]
                for target in targets:
                    if index < batch_size:
                        center_batch[index], target_batch[index] = center, target
                        index += 1
                    else:
                        queue.put((center_batch, target_batch))
                        center_batch = np.zeros(batch_size, dtype=np.long)
                        target_batch = np.zeros([batch_size, 1])
                        index = 0
        queue.put((center_batch[0:index], target_batch[0:index, ]))

if __name__=="__main__":
    (vocab_dict, view_seqs) = Word2vecTokenizer.build_vocab("C:\\Users\\xuezhengyin210834\\Desktop\\text_seqs", 10, False)
    for (center, target) in Word2vecTokenizer.generate_batch(10, 300, view_seqs):
        print(center)
        print(target)