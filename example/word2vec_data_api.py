#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import abspath, dirname
sys.path.insert(0, abspath(dirname(dirname(__file__))))
from word2vec.Word2vecTokenzier import Word2vecTokenizer as wt
from word2vec.Word2vecModel import Word2vecModelPipeline

if __name__=='__main__':
    vocab_dict, view_seqs = wt.build_vocab_threading("C:\\Users\\xuezhengyin210834\\Desktop\\text_seqs", 2, 10, False)
    model = Word2vecModelPipeline(view_seqs, 10, 300, vocab_size=len(vocab_dict)+1, embed_size=10, num_sampled=64, learn_rate=0.1, log_dir=None)
    model.build_graph()
    model.train(20, 60)

