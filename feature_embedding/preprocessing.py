#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import types
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import Counter
from multiprocessing import Pool, Pipe
import codecs
import logging
import pickle
import argparse

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

VIDEO_FEATURE_TYPE = {
    'vid'   :str,
    'tc'    :str,
    'aid'   :str,
    'kw'    :str,
    'tag'   :str,
    'label' :str,
    'ouid'  :str,
    'expose':np.int64,
    'click' :np.int64,
    'score' :str,
}

VIDEO_FEATURE_NAME = VIDEO_FEATURE_TYPE.keys()

class FeatureEmbeddingProcessing(object):

    '''
    cate_index_dict: 表示类别索引字典
    prod_index_dict: 表示出品人索引字典
    label_index_dict: 表示标签索引字典
    '''
    def __init__(self,home_path, feature_path, view_path):
        self._home_path = home_path
        self._feature_path = feature_path
        self._view_path = view_path
        self._videos_feature_dict = None
        self._videos_index_dict = None

    def _BytesListFeature(self, x):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))

    def _Int64ListFeature(self, x):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

    def _FloatListFeature(self, x):
        return tf.train.Feature(float_list=tf.train.FloatList(value=x))

    def _FeaturesListFeature(self, x):
        return tf.train.FeatureList(feature=x)

    def build_batch(self):
        pass

    def _trainsform_index_record(self, x, category_dict, producer_dict, tag_dict):
        tc = x["tc"]
        ouid = x["ouid"]
        tags = x['tag'].strip().split(";")
        feature_dict = {}
        if tc in category_dict.keys():
            feature_dict["tc"] = category_dict.get(tc)
        else:
            feature_dict["tc"] = -1

        if ouid in producer_dict.keys():
            feature_dict["ouid"] = producer_dict.get(ouid)
        else:
            feature_dict["ouid"] = -1

        tag_index_list = list()
        for tag in tags:
            if tag in tag_dict.keys():
                index = tag_dict.get(tag)
                tag_index_list.append(index)
        if len(tag_index_list) > 0:
            feature_dict["tag"] = tag_index_list
        else:
            feature_dict["tag"] = tag_index_list
        x["index"] = feature_dict
        return x

    def _trainsform_index_record_map(self, x):
        tc = x[1]
        ouid = x[6]
        tags = x[4].strip().split(";")
        feature_dict = {}
        if tc in self._category_dict.keys():
            feature_dict["tc"] = self._category_dict.get(tc)
        else:
            feature_dict["tc"] = -1

        if ouid in self._ouid_dict.keys():
            feature_dict["ouid"] = self._ouid_dict.get(ouid)
        else:
            feature_dict["ouid"] = -1

        tag_index_list = list()
        for tag in tags:
            if tag in self._tag_dict.keys():
                index = self._tag_dict.get(tag)
                tag_index_list.append(index)
        if len(tag_index_list) > 0:
            feature_dict["tag"] = tag_index_list
        else:
            feature_dict["tag"] = tag_index_list
        return x[0], feature_dict

    def _transform_index(self, video_feature, thread):
        self._category_dict, self._ouid_dict, self._tag_dict = self._build_index(video_feature)
        video_feature_index = dict()
        length = video_feature.values.size

        with Pool(thread) as pool:
            for cnt, (vid, feature) in enumerate(pool.imap_unordered(self._trainsform_index_record_map, video_feature.values)):
                video_feature_index[vid] = feature
                if cnt % 100000 == 0:
                    logging.info("build video feature index: {}/{}".format(cnt, length))
        return video_feature_index


    '''
    feature_path: 索引字典的路径
    1. 建立类别索引字典
    2. 建立出品人索引字典
    3. 建立标签索引字典
    :returns 
    category_index_dict: 类别索引字典
    ouid_index_dict: 出品人索引字典
    tag_index_dict: 标签索引字典
    '''
    def _build_index(self, video_feature):
        #构建三级类索引并保存
        third_category_index = video_feature["tc"].drop_duplicates().reset_index(drop=True)
        category_index_path = os.path.join(self._home_path, "category.index")
        third_category_index.to_csv(category_index_path, sep="\t")
        index_category_dict = third_category_index.to_dict()
        category_index_dict = dict(zip(index_category_dict.values(), index_category_dict.keys()))
        del third_category_index, index_category_dict
        logging.info("build third category index successful, dictionary size: {}".format(len(category_index_dict)))

        category_path = os.path.join(self._home_path, "category.index")
        category_write = open(category_path, "wb")
        pickle.dump(category_index_dict, category_write)
        logging.info("dump category index in file: {}".format(category_path))

        #构建用户索引并保存
        index_ouid = video_feature["ouid"].drop_duplicates().reset_index(drop=True)
        ouid_path = os.path.join(self._home_path, "ouid.index")
        index_ouid.to_csv(ouid_path, sep='\t')
        index_ouid_dict = index_ouid.to_dict()
        ouid_index_dict = dict(zip(index_ouid_dict.values(), index_ouid_dict.keys()))
        del index_ouid, index_ouid_dict,
        logging.info("build producer index successful, dictionary size: {}".format(len(ouid_index_dict)))
        ouid_path = os.path.join(self._home_path, "ouid.index")
        ouid_write = open(ouid_path, "wb")
        pickle.dump(ouid_index_dict, ouid_write)
        logging.info("dump producer index in file:{}".format(ouid_path))

        #构建标签索引并保存
        video_feature_tag = video_feature.drop(["tag"], axis=1)\
            .join(video_feature["tag"].str.split(";", expand=True).stack().reset_index(level=1, drop=True).rename("tag"))
        tag_index = video_feature_tag["tag"].drop_duplicates().reset_index(drop=True)
        tag_path = os.path.join(self._home_path, "tag.index")
        tag_index.to_csv(tag_path, sep="\t")
        index_tag_dict = tag_index.to_dict()
        tag_index_dict = dict(zip(index_tag_dict.values(), index_tag_dict.keys()))
        del video_feature_tag, tag_index, index_tag_dict
        logging.info("build tag index successful, dictionary size: {}".format(len(tag_index_dict)))
        tag_path = os.path.join(self._home_path, "tag.index")
        tag_write = open(tag_path, "wb")
        pickle.dump(tag_index_dict, tag_write)
        logging.info("dump tag index in file: {}".format(tag_path))

        return category_index_dict, ouid_index_dict, tag_index_dict


    def _normal(self, x, max_count, min_count):
        expose = x["expose"]
        click = x["click"]
        click = min(expose, click)
        x["expose_normal"] = (expose - min_count) / (max_count - min_count)
        x["click_normal"] = (click - min_count) / (max_count - min_count)
        return x

    '''
    对点击和曝光做min-max归一化
    :return
    '''
    def _build_normal_score(self, video_feature):
        expose = video_feature["expose"]
        click = video_feature["click"]
        max_count = expose.max()
        min_count = click.min()
        video_feature_normal = video_feature.apply(self._normal, max_count=max_count, min_count=min_count, axis=1)
        return video_feature_normal

    def _generate_sequence_pair(self, items, windows_size):
        centers_result = []
        targets_result = []
        for (i, center) in enumerate(items):
            size = np.random.randint(1, windows_size)
            targets = [] + items[max(i - size, 0): i] + items[i + 1: min(i + size, len(items))]
            for target in targets:
                centers_result.append(center)
                targets_result.append(target)

        return (centers_result, targets_result)

    '''
    预处理视频特征数据包括以下几个部分:
    1 读取视频特征数据
    2 对视频特征数据进行归一化
    3 生成视频特征索引化后的字典
    :return 特征预处理后的字典
    '''
    def video_feature_preprocessing(self, thread):
        video_feature = pd.read_csv(self._feature_path, sep="\t", names=VIDEO_FEATURE_NAME, dtype=VIDEO_FEATURE_TYPE)
        #video_feature_normal = self._build_normal_score(video_feature)
        video_feature_dict = self._transform_index(video_feature, thread=thread)
        logging.info("build video feature dictionary successful:{}".format(len(video_feature_dict)))
        return video_feature_dict

    '''
    统计每一个block的词汇个数
    block: 单个的block序列
    '''
    def _block_vocab_counter(self, block):
        counter = Counter()
        for line in block:
            counter.update(line.strip().split(" "))
        return counter

    '''
    过滤每一个block不在字典内的item
    block: 每一个需要处理的items序列集合
    vocab_dict:词汇字段
    :return 用户观影序列
    '''
    def _transfer(self, block, vocab_dict):
        sequences = []
        for index, line in enumerate(block):
            items = [i for i in line.strip().split(" ") if i in vocab_dict.keys()]
            if len(items) >= 2:
                sequences.append(items)
            if index % 100000 == 0:
                logging.info("transfer sequences: {}, in pid: {}".format(index, os.getpid()))
        return sequences

    '''
    处理用户观影序列
    thread: 处理用户观影序列的线程数
    min_count: 用户观影序列里面最小出现的词汇次数
    :returns
    vocab_dict: 视频对应的索引字典
    sequences: 过滤最小共现次数的用户序列
    '''
    def user_sequence_preprocessing(self, thread, min_count):
        sequences = []
        vocab_dict = dict()
        counter = Counter()
        lines = codecs.open(self._view_path, 'r', 'utf-8').readlines()
        all_len = len(lines)
        block_len = int(all_len / thread)
        blocks = []
        for i in range(thread):
            start_offset = block_len * i
            end_offset = block_len * (i + 1)
            if i + 1 == thread:
                end_offset = all_len
            blocks.append(lines[start_offset:end_offset])
        with Pool(thread) as pool:
            for block_counter in pool.imap_unordered(self._block_vocab_counter, blocks):
                counter.update(block_counter)
            for (key, value) in counter.items():
                if value >= min_count:
                    vocab_dict[key] = len(vocab_dict)
        logging.info("build vocabulary sucess, vocabulary size:{}".format(len(vocab_dict)))
        pool.join()
        pool.close()
        with Pool(thread) as pool:
            results = [pool.apply_async(self._transfer, (block, vocab_dict)) for block in blocks]
            pool.close()
            pool.join()

        for result in results:
            sequences.extend(result.get())
        return vocab_dict, sequences

    '''
    对一个序列中的items按照window size的大小组成pair对
    items: 表示物品序列
    window_size: 表示滑动窗口大小
    :returns 返回按照顺序组成pair对的两个链表
    '''
    def _generate_sequence_pair(self, items, window_size):
        centers_result = []
        targets_result = []
        for (i, center) in enumerate(items):
            size = np.random.randint(1, window_size)
            targets = [] + items[max(i - size, 0): i] + items[i + 1: min(i + size, len(items))]
            for target in targets:
                centers_result.append(center)
                targets_result.append(target)
        return (centers_result, targets_result)

    '''
    对每一个组成pair的输入输出转化成TF Record的格式并输出
    center: 表示输入的上下文语境
    target: 表示中心词
    '''
    def _serial_pair(self, center, target):
        if center in self._videos_feature_dict.keys() and target in self._vocab_dict.keys():
            feature = self._videos_feature_dict[center]
            ouid = feature["ouid"]
            category = feature["tc"]
            tag = feature["tag"]
            target_index = self._vocab_dict.get(target)
            bytes_center = bytes(center, encoding="utf-8")
            # 每一个输入输出对应的数据结构
            # ouid: 表示作者索引
            # tc: 表示类别索引
            # tag: 表示标签索引
            # target: 表示目标结果
            feature = {
                "center": self._BytesListFeature(x=[bytes_center]),
                "ouid": self._Int64ListFeature(x=[ouid]),
                "tc":   self._Int64ListFeature(x=[category]),
                "tag":  self._Int64ListFeature(x=tag),
                "target": self._Int64ListFeature(x=[target_index]),
            }

            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()
        else:
            return None

    '''
    对每一个block块建立有效的网络输入输出
    block: 表示输入的数据块
    window_size: 表示窗口的大小
    store_size: 表示一个文件中存储的pair size的大小
    start: 表示开始的后缀索引
    thread: 表示线程数
    child_conn: 表示子连接
    '''
    def generate_block_pair(self, data_path, block, window_size, store_size, start, thread, child_conn):
        suffix = start
        path = os.path.join(data_path, "word2vec_{}.tfrecord".format(suffix))
        logging.info("build word2vec_{}.tfrecord successful".format(suffix))
        writer = tf.python_io.TFRecordWriter(path)
        count = 0
        for items in block:
            try:
                (center_result, target_result) = self._generate_sequence_pair(items, window_size)
                for (center, target) in zip(center_result, target_result):
                    example = self._serial_pair(center, target)
                    if example is not None:
                        writer.write(example)
                        count += 0
                    if count > store_size:
                        writer.flush()
                        writer.close()
                        suffix += thread
                        logging.info("build word2vec_{}.tfrecord successful".format(suffix))
                        path = os.path.join(data_path, "word2vec_{}.tfrecord".format(suffix))
                        writer = tf.python_io.TFRecordWriter(path)
                        count = 0

            except Exception as e:
                child_conn.send(e)
                child_conn.close(e)
        writer.close()


    def build_batches_pair_tf_record(self, view_seqs, windows_size, thread, store_size):
        #多线程生成TF Record并保存在多个文件中
        data_path = os.path.join(self._home_path, "tf_record")
        if os.path.exists(data_path) is False:
            os.mkdir(data_path)
        try:
            parent_conn, child_conn = Pipe()

            #划分成thread大小的block数目
            all_len = len(view_seqs)
            block_len = int(all_len / thread)
            blocks = []
            for i in range(thread):
                start_offset = block_len * i
                end_offset = block_len * (i + 1)
                if i + 1 == thread:
                    end_offset = all_len
                blocks.append(view_seqs[start_offset:end_offset])

            #每个线程池处理一个block
            with Pool(thread) as pool:
                [pool.apply_async(self.generate_block_pair, \
                                  (data_path,
                                   block,
                                   windows_size,
                                   store_size,
                                   index,
                                   thread,
                                   child_conn)) for (index, block) in enumerate(blocks)]
                pool.close()
                pool.join()
        except Exception as e:
            logging.error(e)

    def generate_train_data(self, windows_size, min_count, thread, store_size):
        self._videos_feature_dict = self.video_feature_preprocessing(thread)
        self._vocab_dict, self._sequences = self.user_sequence_preprocessing(thread, min_count)
        items_path = os.path.join(self._home_path, "items.index")
        item_write = open(items_path, "wb")
        pickle.dump(self._vocab_dict, item_write)
        logging.info("dump items index in file: {}".format(items_path))
        self.build_batches_pair_tf_record(self._sequences, windows_size, thread, store_size)
        # data_path = os.path.join(self._home_path, "tf_record")
        # self.generate_block_pair(data_path, self._sequences, 5, 3000, 0, 1, None)


def main():
    home_path = args["home"]
    feature_path = args["feature"]
    view_path = args["seqs"]
    window_size = args["window_size"]
    min_count = args["min_count"]
    store_size = args["store"]
    thread_size = args["thread"]
    video_preprocessing = FeatureEmbeddingProcessing(home_path, feature_path, view_path)
    video_preprocessing.generate_train_data(window_size, min_count, thread_size, store_size)

if __name__=="__main__":
    ap = argparse.ArgumentParser(prog="YouTube DNN Processing")

    #训练目录的根路径
    ap.add_argument("--home",
                    type=str,
                    default="C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding",
                    help="home path of the YouTube DNN")

    #训练特征的路径
    ap.add_argument("--feature",
                    type=str,
                    default="C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding\\feature.sample",
                    help="feature path of the video")

    #训练观影序列的路径
    ap.add_argument("--seqs",
                    type=str,
                    default="C:\\Users\\xuezhengyin210834\\Desktop\\feature_embedding\\seqs.sample",
                    help="sequence path of the user action")

    #训练的窗口大小
    ap.add_argument("--window_size", type=int, default=10, help="window size of the YouTube DNN")

    #
    ap.add_argument("--min_count", type=int, default=5, help="min count of the items in action")

    ap.add_argument("--store", type=int, default=300000, help="store size of the file")

    ap.add_argument("--thread", type=int, default=2, help="thread size of the application")

    args = vars(ap.parse_args())

    main()