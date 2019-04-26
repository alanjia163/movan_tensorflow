#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist
mnist = input_data.read_data_sets('./MNIST_data',one_hot=True)

#hyper parameter
batch_size =100
n_batch = mnist.train.num_examples // batch_size

#placehoder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#W_b_
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#layers
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#loss and train_op
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#init
init = tf.global_variables_initializer()

#accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

#train
with tf.Session() as sess:
    sess.run(init)
    #
    for epoch in range(11):
        for bach in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))

    saver.save(sess,'./Save_Net/my_net.ckpt')