#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import numpy as np
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
        #生成词汇表字典
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                counter.update(line.strip().split(" "))
        for (key, value) in counter.items():
            if value >= min_count:
                vocab_dict[key] = len(vocab_dict) + 1
        f.close()
        #处理用户观影序列成index 序列
        with codecs.open(path, 'r', 'utf-8') as f1:
            for line in f1:
                items = [vocab_dict.get(i) for i in line.strip().split(" ") if i in vocab_dict.keys()]
                view_seqs.append(items)
        return vocab_dict, view_seqs

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

if __name__=="__main__":
    (vocab_dict, view_seqs) = Word2vecTokenizer.build_vocab("C:\\Users\\xuezhengyin210834\\Desktop\\text_seqs", 10, False)
    for (center, target) in Word2vecTokenizer.generate_batch(10, 300, view_seqs):
        print(center)
        print(target)