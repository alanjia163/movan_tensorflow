#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
import numpy as np

#data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

#placeholder
# x_p = tf.placeholder(np.float32,[100,])
#structure
W =tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))

y =W*x_data+b


#loss and train_op
loss = tf.losses.mean_squared_error(y_data,y)
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

#sess and init
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    #epoch
    for epoch in range(201):
        sess.run(train_op)
        if epoch%10==0:
            loss_=sess.run(loss)
            print('epoch:%d | loss:%f'%(epoch,loss_),'| weight:',sess.run(W)," | bias:",sess.run(b))

