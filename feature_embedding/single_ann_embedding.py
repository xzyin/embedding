#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s % (message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class FeatureEmbedding(object):
    def __init__(self):
        pass

    def _parse_function(self, serialize_string):
        feature_description = {
            "v_label": tf.VarLenFeature(dtype=tf.int32),
            "v_producer": tf.VarLenFeature(dtype=tf.int64),
        }
    def _create_prepare_data(self):
        pass

    def _create_embedding(self):
        pass