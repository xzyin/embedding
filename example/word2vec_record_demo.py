#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
from os import listdir
import tensorflow as tf
from os.path import abspath, dirname
sys.path.insert(0, abspath(dirname(dirname(__file__))))
from word2vec.Word2vecTokenzier import Word2vecTokenizer as wt

def build_tf_record():
    vocab_dict, view_seqs =wt.build_vocab_threading(args["input"], args["thread"], args["min_count"], True, False)
    wt.build_batches_pair_tf_record(view_seqs, window_size=args["window_size"],
                                    thread=args["thread"], batch_size=args["batch_size"], store_size=args["store_size"],
                                    store_path=args["output"])

def build_file_queue():
    input_path = listdir(args["input"])
    print(input_path)
    data = tf.data.TFRecordDataset(input_path)
    data_iterator = data.make_one_shot_iterator()
    batch = data_iterator.get_next()
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))) as sess:
        print(sess.run(batch))

def read_data():
    pass

def train():
    build_file_queue()

def main():
    method = args["method"]
    if method == "data":
        build_tf_record()
    elif method == "train":
        build_file_queue()

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
    # 选择下一步需要执行的方法
    ap.add_argument("--method", default="train", help="choose the running method")
    args = vars(ap.parse_args())
    main()
