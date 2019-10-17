#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import tensorflow as tf
from os.path import abspath, dirname
sys.path.insert(0, abspath(dirname(dirname(__file__))))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_pair_batches(input_words, output_words):
    feature = {
        'input': _int64_feature(input_words),
        'output': _int64_feature(output_words),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_function(serialize_string):
    feature_description = {
        'input': tf.VarLenFeature(dtype=tf.int64),
        'output': tf.VarLenFeature(dtype=tf.int64),
    }
    return tf.io.parse_single_example(serialize_string, feature_description)

def generate_tf_record():
    #vocab_dict, view_seqs = wt.build_vocab_threading(args["input"], args["thread"], args["min_count"], False)
    a = np.zeros(30, dtype=np.int64).flatten()
    b = np.zeros((30, 1), dtype=np.int64).flatten()
    c = serialize_pair_batches(a,b)
    train_data_path = "C:\\Users\\xuezhengyin210834\\Desktop\\TFRecord.tfrecord"
    writer = tf.python_io.TFRecordWriter(train_data_path)
    writer.write(c)

def load_tf_record():
    dataset = tf.data.TFRecordDataset(["C:\\Users\\xuezhengyin210834\\Desktop\\TFRecord.tfrecord"])
    result = dataset.map(parse_function)
    iterator = result.make_one_shot_iterator()
    batch = iterator.get_next()
    input_sparse = batch["input"]
    output_sparse = batch["output"]
    input = tf.reshape(tf.sparse_tensor_to_dense(input_sparse), shape=(30,))
    output = tf.reshape(tf.sparse_tensor_to_dense(output_sparse), shape=(30, 1))
    sess = tf.Session()
    with sess.as_default():
        input, output = sess.run([input, output])
        print(input)
        print(output)



if __name__=='__main__':
    ap = argparse.ArgumentParser(prog="Word2vec")
    ap.add_argument("--input", help="input path of the word sequences ")
    ap.add_argument("--iter", type=int, default=20, help="max iteration of the word2vec")
    ap.add_argument("--window_size", type=int, default=5, help="window size of the word2vec model")
    ap.add_argument("--batch_size", type=int, default=300, help="batch size of train model data")
    ap.add_argument("--lr", type=float, default=1.0, help="learn rate of the word2vec model")
    ap.add_argument("--size", type=int, default=128, help="dimensions size of the word embedding space")
    ap.add_argument("--min_count", type=int, default=10, help="minimum word frequency")
    ap.add_argument("--log_dir", default="C:\\Users\\xuezhengyin210834\\Desktop\\word2vec_log", help="directory of tensor board log")
    ap.add_argument("--num_sampled", type=int, default=64, help="num sampled of the word2vec model")
    ap.add_argument("--output", help="output path of the vector")
    ap.add_argument("--thread", type=int, default=20, help="thread number of the preprocessing")
    ap.add_argument("--lsize", type=int, default=1000, help="logging of the batch size")
    ap.add_argument("--timeline", default=None, help="Tensor flow time line json path")
    args = vars(ap.parse_args())
    load_tf_record()