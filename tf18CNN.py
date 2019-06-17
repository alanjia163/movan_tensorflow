#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    '''
    传入训练数据v_xs,计算y_pre，比较真值v_ys，计算准确率
    :param v_xs:
    :param v_ys:
    :return:result:accuracy
    '''
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    '''
    :param x: 输出
    :param W: 卷积核大小
    :return:conv
    '''
    # stride [1,x_movement,y_movement,1 ]
    # must have strides[0] = strides[3] = 1
    conv = tf.nn.conv2d(
        input=x,
        filter=W,
        strides=[1, 1, 1, 1],
        padding='SAME',
    )
    return conv


def max_pool_2x2(x):
    # stride [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# placeholders
xs = tf.placeholder(tf.float32, [None, 784]) / 255.
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# reshape
x_image = tf.reshape(x, [-1, 28, 28, 1])
# print(x_image.shape)#[n_samples,28,28,1]

# add_layers
# conv1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32
# conv1
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x32

h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# fc1 layer
W_fc1 = weight_variable([7 * 7 * 64, 10])
b_fc1 = bias_variable([10])
prediciton = tf.nn.softmax(tf.matmul(h_flat, W_fc1) + b_fc1)

# loss and trian_op
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediciton, ), reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

# sess and init
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # epochs
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.6})

        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
