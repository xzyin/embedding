#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import logging
import argparse
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

'''
训练word2vec模型
'''
def train(vocab_dict, view_seqs):
    epoch = args["iter"]
    window_size = args["window_size"]
    batch_size = args["batch_size"]
    learn_rate = args["lr"]
    log_dir = args["log_dir"]
    embed_size = args["size"]
    num_sampled = args["num_sampled"]
    word2vec_model = Word2vecModel(vocab_size=len(vocab_dict)+1, embed_size=embed_size,
                                   num_sampled=num_sampled, learn_rate=learn_rate, log_dir=log_dir)
    word2vec_model.build_graph()
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
        dump(sess, word2vec_model, vocab_dict=vocab_dict)
        word2vec_model.train_writer.close()
    return sess, word2vec_model

def dump(sess, model, vocab_dict):
    output = open(args["output"], "w")
    sess.run(tf.global_variables_initializer())
    index_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    word_vector = sess.run(model.embed_matrix)
    for i, vec in enumerate(word_vector):
        if i in index_dict.keys():
            vid = index_dict.get(i)
            vector = " ".join([str(dim) for dim in vec])
            line = "{} {}\n".format(vid, vector)
            output.write(line)
    output.close()


def main():
    vocab_dict, view_seqs = wt.build_vocab(args["input"], args["min_count"], False)
    logging.info("build vocabulary successful, vocabulary size: {}".format(len(vocab_dict)))
    train(vocab_dict, view_seqs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="Word2vec")
    ap.add_argument("--input", help="input path of the word sequences ")
    ap.add_argument("--iter", type=int, default=20, help="max iteration of the word2vec")
    ap.add_argument("--window_size", type=int, default=5, help="window size of the word2vec model")
    ap.add_argument("--batch_size", type=int, default=300, help="batch size of train model data")
    ap.add_argument("--lr", type=float, default=1.0, help="learn rate of the word2vec model")
    ap.add_argument("--size", type=int, default=128, help="dimensions size of the word embedding space")
    ap.add_argument("--min_count", type=int, default=10, help="minimum word frequency")
    ap.add_argument("--log_dir", help="directory of tensor board log")
    ap.add_argument("--num_sampled", type=int, default=64, help="num sampled of the word2vec model")
    ap.add_argument("--output", help="output path of the vector")
    args = vars(ap.parse_args())
    main()
