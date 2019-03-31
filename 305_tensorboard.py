#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

# import package
import tensorflow as tf
import numpy as np

# data
x = np.linspace(-1, 1, 100)[:, np.newaxis]  # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise  # shape (100, 1) + some noise
tf.set_random_seed(1)
np.random.seed(1)

# placeholders
with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, x.shape, name='x')
    tf_y = tf.placeholder(tf.float32, y.shape, name='y')
# add layers
with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(l1, 1, name='output_layer')
    # add to histogram summary
    tf.summary.histogram('h_out', l1)
    tf.summary.histogram('pred', output)

# loss and train_op
loss = tf.losses.mean_squared_error(tf_y, output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
tf.summary.scalar('loss', loss)  # add loss to scalar summary

# sess and init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# record and write
writer = tf.summary.FileWriter('./log', sess.graph)  # write to file
merge_op = tf.summary.merge_all()

# train_step
for step in range(100):
    # train and net output
    _, result = sess.run([train_op, merge_op], {tf_x: x, tf_y: y})
    writer.add_summary(result, step)
