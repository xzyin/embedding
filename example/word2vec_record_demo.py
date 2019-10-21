#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
from os.path import abspath, dirname
sys.path.insert(0, abspath(dirname(dirname(__file__))))
from word2vec.Word2vecTokenzier import Word2vecTokenizer as wt

def main():
    vocab_dict, view_seqs =wt.build_vocab_threading(args["input"], args["thread"], args["min_count"], True, False)
    wt.build_batches_pair_tf_record(view_seqs, window_size=args["size"],
                                    thread=args["thread"], batch_size=args["batch_size"], store_size=args["store_size"],
                                    store_path="C:\\Users\\xuezhengyin210834\\Desktop\\word2vec")

if __name__=='__main__':
    ap = argparse.ArgumentParser(prog="Word2vec")
    # 输入数据
    ap.add_argument("--input", help="input path of the word sequences ")
    # 迭代轮数
    ap.add_argument("--iter", type=int, default=20, help="max iteration of the word2vec")
    # 窗口大小
    ap.add_argument("--window_size", type=int, default=5, help="window size of the word2vec model")
    # batch大小
    ap.add_argument("--batch_size", type=int, default=300, help="batch size of train model data")
    # 最小出现次数
    ap.add_argument("--min_count", type=int, default=10, help="minimum word frequency")
    # 采样次数
    ap.add_argument("--num_sampled", type=int, default=64, help="num sampled of the word2vec model")
    # 输出路径
    ap.add_argument("--output", help="output path of the vector")
    # 线程数
    ap.add_argument("--thread", type=int, default=2, help="thread number of the preprocessing")
    # 多少个batch size保留
    ap.add_argument("--store_size", default=100, help="store tensorflow record path")
    args = vars(ap.parse_args())
    main()
