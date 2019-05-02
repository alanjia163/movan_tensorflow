#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
import os
import  numpy as np
import re

from PIL import Image

PATH = 'retrain/output_labels.txt'
lines  = tf.gfile.GFile(PATH)

#read data line by line
for uid,line in enumerate(lines):
    line = line.strip('\n')
    uid_to_human[uid] = line

def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


#创建一个图来存放保存好的模型
with tf.gfile.FastGFile('retrain/output_graph.pb',name='pb') as f:
    graph_def  = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')


#sess and do not need init

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    #遍历目录
    for root,dirs,files in os.walk('retrain/images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.)
