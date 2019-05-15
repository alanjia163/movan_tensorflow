#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
output1 = tf.matmul(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[2.,2.,2.,],input2:[3.,2.,3.]}))
    print(sess.run(output,feed_dict={input2:[2.,2.,2.,],input2:[3.,2.,3.]}))
