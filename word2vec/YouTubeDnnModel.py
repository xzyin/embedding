#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import tensorflow as tf

class YouTubeDnnModel(object):

    '''
    构建YouTubeDNN模型的输入参数
    file_queue: 文件队列
    items_size: 整个文件中包含的items的大小,用于构建items矩阵
    item_embed_size: 物品的embedding size大小
    producer_size: 表示出品人的大小
    producer_embed_size: 表示出品人embedding size的大小
    category_size: 表示类别个数的多少
    category_embed_size: 表示类别embedding维度的大小
    log_dir: 表示日志文件夹的路径
    '''
    def __init__(self, file_queue, items_size,
                 item_embed_size,
                 tag_size,
                 tag_embed_size,
                 producer_size,
                 producer_embed_size,
                 category_size,
                 category_embed_size,
                 log_dir,
                 dropout=0.1):

        self._file_queue = file_queue

        self._items_size = items_size
        self._items_embed_size = item_embed_size

        self._tag_size = tag_size
        self._tag_embed_size = tag_embed_size

        self._producer_size = producer_size
        self._producer_embed_size = producer_embed_size

        self._category_size = category_size
        self._category_embed_size = category_embed_size
        self._log_dir = log_dir

        self._dropout = dropout

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    '''
    解析每一个训练数据对的数据
    serialize_string: 序列化之后的string类型数据
    ouid: 解析后的作者索引
    tc: 解析后的类别索引
    tag: 解析后的标签索引
    target: 目标结果索引
    '''
    def _parse_function(self, serialize_string):
        feature_description = {
            "ouid": tf.VarLenFeature(dtype=tf.int64),
            "tc": tf.VarLenFeature(dtype=tf.int64),
            "tag": tf.VarLenFeature(dtype=tf.int64),
            "target": tf.VarLenFeature(dtype=tf.int64)
        }
        features = tf.io.parse_single_example(serialized=serialize_string, features=feature_description)

        ouid = features["ouid"]
        tc = features["tc"]
        tag = features["tag"]
        target = features["target"]
        return (ouid, tc, tag, target)

    '''
    预加载文件中的数据
    '''
    def _create_prepare_data(self):
        self.dataset_record = tf.data.TFRecordDataset(self._file_queue).prefetch(1000000)
        self.parse_dataset = self.dataset_record.map(self._parse_function).batch(300)
        self.data_iterator = self.parse_dataset.make_initializable_iterator()
        self._ouid, self._tc, self._tag, self._target = self.data_iterator.get_next()
        self._dense_target = tf.sparse_tensor_to_dense(self._target)

    '''
    创建embedding向量矩阵    
    '''
    def _create_embedding(self):
        with tf.name_scope("embed"):
            self._items_matrix = tf.Variable(tf.random_uniform([self._items_size,
                                                                self._items_embed_size], -1.0, 1.0),
                                             name="items_matrix")

            self._tags_matrix = tf.Variable(tf.random_uniform([self._tag_size,
                                                              self._tag_embed_size], -1.0, 1.0),
                                            name="tags_matrix")

            self._producers_matrix = tf.Variable(tf.random_uniform([self._producer_size,
                                                                   self._producer_embed_size], -1.0, 1.0),
                                                name="producer_matrix")

            self._categorys_matrix = tf.Variable(tf.random_uniform([self._category_size,
                                                                   self._category_embed_size], -1.0, 1.0),
                                                name="category_matrix")

    '''
    构建多层的dnn模型
    '''
    def _create_input_context(self):
        self._producer_context = tf.nn.embedding_lookup_sparse(self._producers_matrix,
                                                               self._ouid,
                                                               combiner="mean",
                                                               sp_weights=None,
                                                               name="producer_embed")

        self._categorys_context = tf.nn.embedding_lookup_sparse(self._categorys_matrix,
                                                                self._tc,
                                                                combiner="mean",
                                                                sp_weights=None,
                                                                name="category_embed")

        self._tag_context = tf.nn.embedding_lookup_sparse(self._tags_matrix,
                                                          self._tag,
                                                          sp_weights=None,
                                                          combiner="mean",
                                                          name="tag_embed")


    #创建浅层神经网络
    def _create_shallow_prcoess(self):
        pass

    def _create_dnn_process_nce(self):
        # 完成向量的拼接
        self._context = tf.concat([tf.concat([self._producer_context, self._categorys_context], 1), self._tag_context], 1)

    def _create_dnn_process(self):
        # 构建全连接向量
        self._context = tf.concat([tf.concat([self._producer_context, self._categorys_context], 1),
                                   self._tag_context],
                                  1)
        # 全连接层1
        layers1 = tf.layers.dense(self._context, 1024, activation=tf.nn.relu, name="layer1")
        layers1_dropout = tf.nn.dropout(layers1, keep_prob=self._dropout, name="dropout1")

        # 全连接层2
        layers2 = tf.layers.dense(layers1_dropout, 512, activation=tf.nn.relu, name="layer2")
        layers2_dropout = tf.nn.dropout(layers2, keep_prob=self._dropout, name="dropout2")

        # 全连接层3
        self._user_vector = tf.layers.dense(layers2_dropout, 30, activation=tf.nn.relu, name="layer3")
        self._sample_softmax_biases = tf.get_variable('soft_biases', initializer=tf.zeros([self._items_size]), trainable=False)
        self._loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self._items_matrix,
                                                               biases=self._sample_softmax_biases,
                                                               labels=self._dense_target,
                                                               inputs=self._user_vector,
                                                               num_sampled=20,
                                                               num_true=1,
                                                               num_classes=self._items_size,
                                                               partition_strategy="mod"))



    def build_graph(self):
        self._create_prepare_data()
        self._create_embedding()
        self._create_input_context()
        self._create_dnn_process()


    def train(self):
        with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.data_iterator.initializer)
            # tag, producer, category = sess.run([self._tag_context, self._producer_context, self._categorys_context])
            # print(tag.shape)
            # print(producer.shape)
            # print(category.shape)
            context = sess.run(self._loss)
            print(context)


def build_file_queue():
    input_dir = "C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding\\tf_record"
    input_paths = os.listdir(input_dir)
    pathes = [os.path.join(input_dir, path) for path in input_paths]
    return pathes

def load_dict():
    category_dict_path = "C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding\\category.index"
    category_read = open(category_dict_path, "rb")
    category_dict = pickle.load(category_read)

    ouid_dict_path = "C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding\\ouid.index"
    ouid_read = open(ouid_dict_path, "rb")
    ouid_dict = pickle.load(ouid_read)

    tag_dict_path = "C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding\\tag.index"
    tag_read = open(tag_dict_path, "rb")
    tag_dict = pickle.load(tag_read)

    items_dict_path = "C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding\\items.index"
    items_read = open(items_dict_path, "rb")
    items_dict = pickle.load(items_read)

    return category_dict, ouid_dict, tag_dict, items_dict


if __name__=="__main__":
    pathes = build_file_queue()
    category_dict, ouid_dict, tag_dict, items_dict = load_dict()
    model = YouTubeDnnModel(pathes, len(items_dict), 30, len(tag_dict), 30, len(ouid_dict), 30, len(category_dict), 30, None)
    model.build_graph()
    model.train()

