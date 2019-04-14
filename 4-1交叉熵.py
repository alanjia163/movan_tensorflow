#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#load data
mnist = input_data.read_data_sets("MNIST_dada",one_hot=True)

#batch_size
batch_size =100
#batchs
n_batch = mnist.train.num_examples//batch_size

#placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#add layers
W =tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lavels=y,logits=prediction))
train_step =tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

#argmax()返回一维度张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
