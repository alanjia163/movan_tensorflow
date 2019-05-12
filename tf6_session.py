#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
prediction = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    p = sess.run(prediction)
    print(p)



