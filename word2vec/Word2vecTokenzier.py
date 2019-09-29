#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs
import numpy as np
from multiprocessing import Pool
from collections import Counter
import multiprocessing
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
        #生成词汇表字典
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
        #处理用户观影序列成index 序列
        index = 0
        with codecs.open(path, 'r', 'utf-8') as f1:
            for line in f1:
                items = [vocab_dict.get(i) for i in line.strip().split(" ") if i in vocab_dict.keys()]
                view_seqs.append(items)
                index += 1
                if index % 1000000 == 0:
                    logging.info("filter minimum vocabulary, load index:{}".format(index))
        return vocab_dict, view_seqs

    @staticmethod
    def transfer(block, vocab_dict):
        print("this is a good idea")
        sequences = []
        for index, line in enumerate(block):
            items = [vocab_dict.get(i) for i in line.strip().split(" ") if i in vocab_dict.keys()]
            if len(items) >= 2:
                sequences.append(items)
            if index % 100000 == 0:
                logging.info("transfer sequeneces:{}, in pid:{}".format(index, os.getpgid()))
        return sequences

    @staticmethod
    def block_vocab_counter(block):
        counter = Counter()
        for line in block:
            counter.update(line.strip().split(" "))
        return counter

    @staticmethod
    def build_vocab_threading(path, thread, min_count, with_rating=True):
        sequences = []
        vocab_dict = dict()
        counter = Counter()
        with codecs.open(path, 'r', 'utf-8') as f:
            lines = f.readlines()
            f.close()
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
        with Pool(thread) as pool:
            results = [pool.apply_async(Word2vecTokenizer.transfer, (block, vocab_dict)) for block in blocks]
            pool.close()
            pool.join()
        for result in results:
            sequences.extend(result.get())
        logging.info("build sequence success, sequence lenth: {}".format(len(sequences)))
        return vocab_dict, sequences

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