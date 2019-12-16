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
        'ouid': tf.VarLenFeature(dtype=tf.int64),
        "tc": tf.VarLenFeature(dtype=tf.int64),
        "tag_index": tf.VarLenFeature(dtype=tf.int64),
        "tag": tf.VarLenFeature(dtype=tf.int64),
        'tag_weight': tf.VarLenFeature(dtype=tf.float32),
        'target': tf.VarLenFeature(dtype=tf.int64)
    }
    features = tf.io.parse_single_example(serialized=serialize_string, features=feature_description)

    ouid = features["ouid"]
    tc = features["tc"]
    tag = features["tag"]
    target = features["target"]
    return target
t1 = [[1, 2, 3], [2, 3, 4]]
t2 = [[3, 4, 5], [5, 6, 7]]
tag = tf.random_uniform([1000], minval=0, maxval=10, dtype=tf.int32)
pathes = build_file_queue()
dataset_record = tf.data.TFRecordDataset(pathes).prefetch(100000)
map_dataset = dataset_record.map(parse_function).batch(300)
data_iterator = map_dataset.make_initializable_iterator()
result = data_iterator.get_next()
with tf.Session() as sess:
    sess.run(data_iterator.initializer)
    a = sess.run(tag)
    print(a)
