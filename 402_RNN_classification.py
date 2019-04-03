#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# hyper parameter
BATCH_SIZE = 64
TIME_STEP = 28  # RNN time step / image height
INPUT_SIZE = 28  # RNN input size /image width
LR = 0.01

# data
mnist = input_data.read_data_sets('./mnist', Exception=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot


# placheholders
tf_x = tf.placeholder(tf.float32, shape=[None, TIME_STEP * INPUT_SIZE])
tf_y = tf.placeholder(tf.int32, shape=[None, 10])

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,
    image,
    initial_state=None,
    dtype=tf.float32,
    time_major=False,
)

#output based on the last output step
output = tf.layers.dense(outputs[:,-1,:],10)

#loss and train_op
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

#return acc,update_op
accuracy  = tf.metrics.accuracy(
    labels = tf.argmax(tf_y,axis =1),predictions=tf.argmax(output,axis=1),
)[1]

#sess and init
sess  = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init_op)

for step in range(1200):
    b_x,b_y = mnist.train.next_batch(BATCH_SIZE)
    _,loss = tess.run([train_op,loss]),{tf_x:test_x,tf_y:test_y}
    if step % 50 ==0:
        accuracy_=sess.run(accuracy,{tf_x:test_x,tf_y:test_y})
        print('train loss:%.4f'%loss_,'|test accuracy: %.2f' % accuracy_)


    #print 10 predictions from test data
    test_output  = sess.run(output,{tf_x:test_x[:10]})
    pred_y = np.argmax(test_output, 1)
    print(pred_y, 'prediction number')
    print(np.argmax(test_y[:10], 1), 'real number')










