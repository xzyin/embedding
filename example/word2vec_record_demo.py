#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import listdir, mkdir
from os.path import abspath, dirname, join, exists
import sys
import argparse
import pickle
import tensorflow as tf
from word2vec.Word2vecModel import Word2vecModelRecordPipeline
from word2vec.Word2vecTokenzier import Word2vecTokenizer as wt
sys.path.insert(0, abspath(dirname(dirname(__file__))))


def cam_dir(dir_path):
    if not exists(dir_path):
        mkdir(dir_path)

def build_tf_record():
    vocab_dict, view_seqs = wt.build_vocab_threading(args["input"], args["thread"], args["min_count"], True, False)
    output_dir = args["output"]
    output_dir_record = join(output_dir, "tf_record")
    if not exists(output_dir_record):
        mkdir(output_dir_record)
    wt.build_batches_pair_tf_record(view_seqs, window_size=args["window_size"],
                                    thread=args["thread"], batch_size=args["batch_size"], store_size=args["store_size"],
                                    store_path=output_dir_record)
    dict_file = open(join(args["output"], "vocab_dict"), "wb")
    pickle.dump(vocab_dict, dict_file)
    dict_file.close()

def parse_function(serialize_string):
    feature_description = {
        'input': tf.VarLenFeature(dtype=tf.int64),
        'output': tf.VarLenFeature(dtype=tf.int64),
    }

    features = tf.io.parse_single_example(serialized=serialize_string, features=feature_description)
    input_sparse_tensor = features["input"]
    output_sparse_tensor = features["output"]
    input_dense_tensor = tf.sparse_tensor_to_dense(input_sparse_tensor)
    output_dense_tensor = tf.sparse_tensor_to_dense(output_sparse_tensor)
    input_tensor = tf.reshape(input_dense_tensor, shape=(args["batch_size"], ))
    output_tensor = tf.reshape(output_dense_tensor, shape=(args["batch_size"], 1))
    return (input_tensor, output_tensor)

def build_file_queue():
    input_dir = args["input"]
    input_dir_record = join(input_dir, "tf_record")
    input_paths = listdir(input_dir_record)
    pathes = [join(input_dir_record, path) for path in input_paths]
    dict_file = open(join(args["input"], "vocab_dict"), "rb")
    vocab_dict = pickle.load(dict_file)
    dict_file.close()
    return pathes, vocab_dict

def train():
    pathes, vocab_dict = build_file_queue()
    model = Word2vecModelRecordPipeline(files_queue=pathes,
                                        batch_size=args["batch_size"],
                                        vocab_size=len(vocab_dict) + 1,
                                        embed_size=args["embed_size"],
                                        num_sampled=args["num_sampled"],
                                        learn_rate=args["learn_rate"],
                                        log_dir=args["log_dir"])
    model.build_graph()
    timeline_dir = args["timeline"]
    cam_dir(timeline_dir)
    embeding_matrix = model.train(args["iter"], args["lsize"], timeline_dir)
    index_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    output = open(args["output"], "w")
    for (i, vector) in enumerate(embeding_matrix):
        if i in index_dict.keys():
            dims = [str(dim) for dim in vector]
            res = "{}\t{}\n".format(index_dict.get(i), " ".join(dims))
            output.write(res)
    output.close()

def main():
    method = args["method"]
    if method == "data":
        build_tf_record()
    elif method == "train":
        train()

if __name__ == '__main__':
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
    # embedding 维度大小
    ap.add_argument("--embed_size", type=int, default=128, help="embedding size of the word2vec model")
    # 采样次数
    ap.add_argument("--num_sampled", type=int, default=64, help="num sampled of the word2vec model")
    # 输出路径
    ap.add_argument("--output", help="output path of the vector")
    # 线程数
    ap.add_argument("--thread", type=int, default=1, help="thread number of the preprocessing")
    # 多少个batch size保留
    ap.add_argument("--store_size", type=int, default=100, help="store tensorflow record path")
    # 选择下一步需要执行的方法
    ap.add_argument("--method", default="train", help="choose the running method")
    # 模型的学习速率
    ap.add_argument("--learn_rate", type=int, default=1.0, help="learn rate of the word2vec model")
    # TensorBoard和TimeLine文件夹
    ap.add_argument("--log_dir", help="path of the tensorboard")
    # 打印日志的窗口大小
    ap.add_argument("--lsize", type=int, default=1000, help="logging of the batch size")
    # timeline json保存的位置
    ap.add_argument("--timeline", default=None, help="Tensor flow time line json path")
    args = vars(ap.parse_args())
    main()
