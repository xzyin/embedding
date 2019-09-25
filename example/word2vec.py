#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np
import tensorflow as tf
from os.path import abspath, dirname
sys.path.insert(0, abspath(dirname(dirname(__file__))))
from word2vec.Word2vecTokenzier import Word2vecTokenizer as wt
from word2vec.Word2vecModel import Word2vecModel
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def train(epoch, window_size, batch_size, learn_rate):
    vocab_dict, view_seqs = wt.build_vocab("C:\\Users\\xuezhengyin210834\\Desktop\\text_seqs", 10, False)
    word2vec_model = Word2vecModel(len(vocab_dict) + 1, 10, 10, 1.0, "C:\\Users\\xuezhengyin210834\\Desktop\\test_log_dir")
    ratio = learn_rate / epoch
    word2vec_model.build_graph()
    #saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            batch_count = 0
            k = 0
            batches_loss = []
            for context, target in wt.generate_batch(window_size, batch_size, view_seqs):
                feed_dict = {word2vec_model.context_words: context,
                             word2vec_model.target_words: target}
                loss_batch, _ = sess.run([word2vec_model.loss,
                                          word2vec_model.optimizer],
                                         feed_dict=feed_dict)
                batches_loss.append(loss_batch)
                batch_count += context.shape[0]

                k += 1
                if k % 60 == 0:
                    logging.info("epoch: {}, batch count: {}, pair count: {}, loss: {:5.5f}"
                                 .format(i, k, batch_count, loss_batch))
            feed_dict = {word2vec_model.batches_loss: np.array(batches_loss)}
            average_loss, merge = sess.run([word2vec_model.average_loss,word2vec_model.merge], feed_dict=feed_dict)
            word2vec_model.train_writer.add_summary(merge, i)
            logging.info("Average loss at epoch {}, pair count {}, current loss {:5.5f}, learn rate {:3.3f}"
                         .format(i, batch_count, average_loss, learn_rate))

        word2vec_model.train_writer.close()



if __name__ == "__main__":
    train(10, 10, 300, 1.0)