#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

def build_file_queue():
    input_dir = "C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding\\tf_record"
    input_paths = os.listdir(input_dir)
    pathes = [os.path.join(input_dir, path) for path in input_paths]
    return pathes

def parse_function(serialize_string):
    feature_description = {
        'center': tf.VarLenFeature(dtype=tf.string),
        "ouid": tf.VarLenFeature(dtype=tf.int64),
        "tc": tf.VarLenFeature(dtype=tf.int64),
        "tag": tf.VarLenFeature(dtype=tf.int64),
        'target': tf.VarLenFeature(dtype=tf.int64),
        'sample': tf.VarLenFeature(dtype=tf.int64)
    }
    features = tf.io.parse_single_example(serialized=serialize_string, features=feature_description)

    ouid = features["ouid"]
    tc = features["tc"]
    tag = features["tag"]
    target = features["target"]
    sample = features["sample"]

    return target
t1 = [[1, 2, 3], [2, 3, 4]]
t2 = [[3, 4, 5], [5, 6, 7]]
tag = tf.random_uniform([1000], minval=0, maxval=10, dtype=tf.int32)
pathes = build_file_queue()
sample_softmax_biases = tf.get_variable('soft_biases', shape=[30], initializer=tf.random_normal_initializer(mean=0, stddev=1), trainable=False)
q = tf.nn.embedding_lookup(sample_softmax_biases, [1, 2, 3])
dataset_record = tf.data.TFRecordDataset(pathes).prefetch(100000)
map_dataset = dataset_record.map(parse_function).batch(300)
data_iterator = map_dataset.make_initializable_iterator()
result = data_iterator.get_next()
kkk = tf.sparse_tensor_to_dense(result)
with tf.Session() as sess:
    sess.run(data_iterator.initializer)
    sess.run(tf.initialize_all_variables())

    a = sess.run(kkk)
    print(a)
