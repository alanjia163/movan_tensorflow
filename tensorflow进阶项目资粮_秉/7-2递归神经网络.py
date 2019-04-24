#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# hyper parameter
n_inputs = 28  # 输入一行,一行有28个数据
max_time = 28  # 一共28行
lstm_size = 100
n_classes = 10
batch_size = 50
n_batch = mnist.train.num_examples // batch_size

# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# W_and_b
W = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# rnn
def RNN(X, W):
    # inputs = [batch_size,max_time,n_inputs]
    inputs = tf.reshape(x, [-1, max_time, n_inputs])
    # define_lstm_cell
    lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + b)
    return results


prediction = RNN(x, W, b)
cross_entropy = tf.redcue_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels= y))
train_step = tf.train.AdamOptimizer(le-4).minimize(cross_entropy)
#compute_accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#init
init = tf.global_variables_initializer()

#train_step
with tf.Session() as sess:
    sess.run(init)

    #epoch
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('iter',str(epoch)+',Testing accuracy:'+str(acc))
