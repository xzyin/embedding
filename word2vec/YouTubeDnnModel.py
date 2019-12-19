#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import pickle
import argparse
import tensorflow as tf
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
    def __init__(self, home_path,
                 file_queue,
                 batch_size,
                 prefetch_size,
                 check_point,
                 log_size,
                 items_size,
                 item_embed_size,
                 tag_size,
                 tag_embed_size,
                 producer_size,
                 producer_embed_size,
                 category_size,
                 category_embed_size,
                 log_dir,
                 dropout=0.1,
                 lr=1.0):

        self._home_path = home_path
        self._file_queue = file_queue

        self._batch_size = batch_size
        self._prefetch_size = prefetch_size
        self._check_point = check_point
        self._log_size = log_size
        self._items_size = items_size
        self._items_embed_size = item_embed_size

        self._tag_size = tag_size
        self._tag_embed_size = tag_embed_size

        self._producer_size = producer_size
        self._producer_embed_size = producer_embed_size

        self._category_size = category_size
        self._category_embed_size = category_embed_size
        self._log_dir = log_dir

        self._model_path = os.path.join(self._home_path, "model")
        if os.path.exists(self._model_path) is False:
            os.mkdir(self._model_path)

        self._dropout = dropout
        self._lr = lr
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
            "center": tf.VarLenFeature(dtype=tf.string),
            "ouid": tf.VarLenFeature(dtype=tf.int64),
            "tc": tf.VarLenFeature(dtype=tf.int64),
            "tag": tf.VarLenFeature(dtype=tf.int64),
            "target": tf.VarLenFeature(dtype=tf.int64)
        }
        features = tf.io.parse_single_example(serialized=serialize_string, features=feature_description)

        center = features["center"]
        ouid = features["ouid"]
        tc = features["tc"]
        tag = features["tag"]
        target = features["target"]
        return (center, ouid, tc, tag, target)

    '''
    预加载文件中的数据
    '''
    def _create_prepare_data(self):
        with tf.name_scope("data"):
            self.dataset_record = tf.data.TFRecordDataset(self._file_queue).prefetch(self._prefetch_size)
            self.parse_dataset = self.dataset_record.map(self._parse_function).batch(self._batch_size)
            self.data_iterator = self.parse_dataset.make_initializable_iterator()
            self._center, self._ouid, self._tc, self._tag, self._target = self.data_iterator.get_next()
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
        with tf.name_scope("create_input_context"):
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

    def _create_dnn_process(self):
        with tf.name_scope("dnn_process"):
            # 构建全连接向量
            self._context = tf.concat([tf.concat([self._producer_context, self._categorys_context], 1),
                                       self._tag_context], 1)
            # 全连接层1
            layers1 = tf.layers.dense(self._context, 1024, activation=tf.nn.relu, name="layer1")
            layers1_dropout = tf.nn.dropout(layers1, keep_prob=self._dropout, name="dropout1")

            # 全连接层2
            layers2 = tf.layers.dense(layers1_dropout, 512, activation=tf.nn.relu, name="layer2")
            layers2_dropout = tf.nn.dropout(layers2, keep_prob=self._dropout, name="dropout2")

            # 全连接层3
            self._user_vector = tf.layers.dense(layers2_dropout, self._items_embed_size, activation=tf.nn.relu, name="layer3")

    def _top_k(self):
        with tf.name_scope("top_k"):
            normal_user_vector = tf.nn.l2_normalize(self._user_vector, dim=1)
            items_normal_matrix = tf.nn.l2_normalize(self._items_matrix, dim=1)
            dist = tf.matmul(normal_user_vector, items_normal_matrix, transpose_b=True, name="l2_distance")
            self._top_values, self._top_idxs = tf.nn.top_k(dist, k=100)

    def _create_loss(self):
        with tf.name_scope("dnn_process"):
            self._sample_softmax_biases = tf.get_variable('soft_biases',
                                                          initializer=tf.zeros([self._items_size]),
                                                          trainable=False)

            self._loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=self._items_matrix,
                                                                   biases=self._sample_softmax_biases,
                                                                   labels=self._dense_target,
                                                                   inputs=self._user_vector,
                                                                   num_sampled=20,
                                                                   num_true=1,
                                                                   num_classes=self._items_size,
                                                                   partition_strategy="mod"))
    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self._optimizer = tf.train.GradientDescentOptimizer(self._lr).minimize(self._loss, global_step=self.global_step)

    def build_graph(self):
        self._create_prepare_data()
        self._create_embedding()
        self._create_input_context()
        self._create_dnn_process()
        self._create_loss()
        self._top_k()
        self._create_optimizer()


    def train(self, saver, epoch):
        self._sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)))
        self._sess.run(tf.global_variables_initializer())
        total_loss = 0.0
        batch_cnt = 0
        for i in range(epoch):
            self._sess.run(self.data_iterator.initializer)
            while True:
                try:
                    batch_cnt += 1
                    loss, _ = self._sess.run([self._loss, self._optimizer])
                    if batch_cnt % self._log_size == 0:
                        logging.info("batch count: {}, loss :{}".format(batch_cnt, loss))
                    total_loss += loss
                except tf.errors.OutOfRangeError:
                    logging.info("Average loss: {} of the epoch: {}".format(total_loss / batch_cnt, i))
                    total_loss = 0.0
                    batch_cnt = 0
                    if i != 0 and i % self._check_point == 0:
                        self.check_point(saver, i)
                    break

    '''
    保存category embedding, tag embedding, producer embdding, items embedding
    category_dict:表示类别字典
    tag_dict:表示标签字典
    producer_dict:表示出品人字典
    items_dict:表示物品字典
    '''
    def save(self, category_dict, tag_dict, producer_dict, items_dict, suffix=""):
        index_category = dict(zip(category_dict.values(), category_dict.keys()))
        index_tag = dict(zip(tag_dict.values(), tag_dict.keys()))
        index_producer = dict(zip(producer_dict.values(), producer_dict.keys()))
        index_items = dict(zip(items_dict.values(), items_dict.keys()))
        model_path = os.path.join(self._model_path, "feature_vector" + suffix)
        if os.path.exists(model_path) is False:
            os.mkdir(model_path)
        tc_m, tag_m, prod_m, item_m = self._sess.run([self._categorys_matrix,
                    self._tags_matrix,
                    self._producers_matrix,
                    self._items_matrix])

        # 保存类别的向量
        category_path = os.path.join(model_path, "category.vec")
        category_write = open(category_path, "w")
        for i, category_vector in enumerate(tc_m):
            category_write.write(index_category[i] +
                                 "\t" + ",".join([str(dim) for dim in category_vector]) +
                                 "\n")

        # 保存标签的向量
        tag_path = os.path.join(model_path, "tag.vec")
        tag_write = open(tag_path, "w")
        for i, tag_vector in enumerate(tag_m):
            tag_write.write(index_tag[i] + "\t"
                            + ",".join([str(dim) for dim in tag_vector])
                            + "\n")

        # 保存出品人的向量
        producer_path = os.path.join(model_path, "producer.vec")
        producer_write = open(producer_path, "w")
        for i, producer_vector in enumerate(prod_m):
            producer_write.write(index_producer[i] + "\t"
                                 + ",".join([str(dim) for dim in producer_vector])
                                 + "\n")

        # 保存item的向量
        items_path = os.path.join(model_path, "items.vec" + suffix)
        items_write = open(items_path, "w")
        for i, items_vector in enumerate(item_m):
            items_write.write(index_items[i] + "\t"
                              + ",".join([str(dim) for dim in items_vector])
                              + "\n")

    def check_point(self, saver, epoch):
        if os.path.exists(self._model_path) is False:
            os.mkdir(self._model_path)
        check_point = os.path.join(self._model_path, "check_point/dl")
        saver.save(self._sess, check_point, epoch)
        logging.info("Check point path: {}".format(check_point))

def build_file_queue(input_dir):
    input_paths = os.listdir(input_dir)
    pathes = [os.path.join(input_dir, path) for path in input_paths]
    return pathes

'''
path
'''
def load_dict(path):
    category_dict_path = os.path.join(path, "category.index")
    category_read = open(category_dict_path, "rb")
    category_dict = pickle.load(category_read)

    ouid_dict_path = os.path.join(path, "ouid.index")
    ouid_read = open(ouid_dict_path, "rb")
    ouid_dict = pickle.load(ouid_read)

    tag_dict_path = os.path.join(path, "tag.index")
    tag_read = open(tag_dict_path, "rb")
    tag_dict = pickle.load(tag_read)

    items_dict_path = os.path.join(path, "items.index")
    items_read = open(items_dict_path, "rb")
    items_dict = pickle.load(items_read)

    return category_dict, ouid_dict, tag_dict, items_dict

def train():
    # 表示embedding的大小
    embed_size = args["embed_size"]
    # 表示学习率
    learn_rate = args["lr"]
    # 表示dropout的概率
    dropout = args["dropout"]
    # 表示迭代的轮数
    epoch = args["epoch"]
    # 表示训练的根路径
    home_path = args["home"]
    #batch_size
    batch_size = args["batch_size"]

    #prefectch size
    prefectch_size = args["prefetch_size"]

    check_point = args["checkpoint"]

    log_size = args["log_size"]
    # 表示训练数据的路径
    data_path = os.path.join(home_path, "tf_record")

    pathes = build_file_queue(data_path)
    category_dict, ouid_dict, tag_dict, items_dict = load_dict(home_path)
    model = YouTubeDnnModel(home_path, pathes,
                            batch_size, prefectch_size,
                            check_point, log_size,
                            len(items_dict), embed_size,
                            len(tag_dict), embed_size,
                            len(ouid_dict), embed_size,
                            len(category_dict), embed_size,
                            None,
                            learn_rate,
                            dropout)
    model.build_graph()
    saver = tf.train.Saver()
    model.train(saver, epoch)
    model.save(category_dict, tag_dict, ouid_dict, items_dict)
    model.check_point(saver, epoch - 1)

def predict():
    # 表示embedding的大小
    embed_size = args["embed_size"]
    # 表示学习率
    learn_rate = args["lr"]
    # 表示dropout的概率
    dropout = args["dropout"]
    # 表示迭代的轮数
    epoch = args["epoch"]
    # 表示训练的根路径
    home_path = args["home"]
    # batch_size
    batch_size = args["batch_size"]

    # prefectch size
    prefectch_size = args["prefetch_size"]

    check_point = args["checkpoint"]

    log_size = args["log_size"]
    # 表示训练数据的路径
    data_path = os.path.join(home_path, "predict_tf_record")
    output_path = os.path.join(home_path, "predict.sim")
    output_write = open(output_path, "w")
    pathes = build_file_queue(data_path)
    category_dict, ouid_dict, tag_dict, items_dict = load_dict(home_path)
    model = YouTubeDnnModel(home_path, pathes,
                            batch_size, prefectch_size,
                            check_point, log_size,
                            len(items_dict), embed_size,
                            len(tag_dict), embed_size,
                            len(ouid_dict), embed_size,
                            len(category_dict), embed_size,
                            None,
                            learn_rate,
                            dropout)
    model.build_graph()
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)))

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(os.path.join(os.path.join(home_path, "model"), "check_point/checkpoint")))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    logging.info("tf model init successfully")
    index_items = dict(zip(items_dict.values(), items_dict.keys()))
    sess.run(model.data_iterator.initializer)
    while True:
        try:
            centers, top_idxs = sess.run([model._center, model._top_idxs])
            for center, value in zip(centers.values,top_idxs):
                center_str = str(center, encoding="utf-8")
                output_write.write(str(center_str) + "\t" + ",".join([index_items[i] for i in value]) + "\n")
        except tf.errors.OutOfRangeError:
            logging.info("predict finished")
            break

def main():
    medthod = args["method"]
    if medthod == "train":
        train()
    else:
        predict()


if __name__=="__main__":
    ap = argparse.ArgumentParser(prog="YouTube DNN Processing")

    #模型训练的根路径
    ap.add_argument("--home",
                    type=str,
                    default="C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding",
                    help="home path of the YouTube DNN")

    # 训练迭代的轮数
    ap.add_argument("--epoch", type=int, default=2, help="epoch of the YouTube DNN")

    # 学习率
    ap.add_argument("--lr", type=float, default=1.0, help="learn rate of the YouTube DNN Model")

    # embedding 维度的大小
    ap.add_argument("--embed_size", type=int, default=300, help="embed size of the feature")

    # dropout概率
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout of the YouTube DNN Model")

    # batch_size 大小
    ap.add_argument("--batch_size", type=int, default=300, help="batch size of the train model")

    # prefetch_size
    ap.add_argument("--prefetch_size", type=int, default=300000, help="prefetch size of the preprocess data")

    # 模型保存的间隔
    ap.add_argument("--checkpoint", type=int, default=5, help="check out point during train")

    # 单轮迭代batch的打印
    ap.add_argument("--log_size", type=int, default=300, help="log batch count")

    # 表示预测或者训练
    ap.add_argument("--method", type=str, default="predict", help="train or predict")

    args = vars(ap.parse_args())
    main()
