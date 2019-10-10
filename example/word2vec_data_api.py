#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
from os.path import abspath, dirname
sys.path.insert(0, abspath(dirname(dirname(__file__))))
from word2vec.Word2vecModel import Word2vecModelPipeline
from word2vec.Word2vecTokenzier import Word2vecTokenizer as wt

def main():
    vocab_dict, view_seqs =wt.build_vocab_threading(args["input"], args["thread"], args["min_count"], False)
    model = Word2vecModelPipeline(view_seqs, args["window_size"], args["batch_size"], vocab_size=len(vocab_dict) + 1,
                                  embed_size=args["embed_size"], num_sampled=args["num_sampled"],
                                  learn_rate=args["lr"], log_dir=args["log_dir"])
    model.build_graph()
    model.train(args["iter"], args["lsize"])

if __name__=='__main__':
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
    ap.add_argument("--thread", type=int, default=20, help="thread number of the preprocessing")
    ap.add_argument("--lsize", type=int, default=1000, help="logging of the batch size")
    args = vars(ap.parse_args())
    main()

