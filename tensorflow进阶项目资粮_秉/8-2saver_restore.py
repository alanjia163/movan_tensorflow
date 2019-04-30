#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#data
mnist =input_data.read_data_sets('MNIST_data',one_hot=True)


#hyper parameter
batch_size =100
n_batch =mnist.train.num_examples // batch_size

#placeholder
X = tf.placeholder(tf.float32,[None,28*28])
Y = tf.placeholder(tf.float32,[None,10])

#add_layer
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#prediction
prediction = tf.nn.softmax(tf.matmul(X,W)+b)

#loss and train_op
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_op tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#accuracy
correct_prediction =tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Saver
saver = tf.train.Saver()

#train_step
with tf.Session() as sess:
    #init_sess
    init =tf.global_variables_initializer()
    #epoch
    sess.run(init)
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    saver.restore(sess,'net/my_net.ckpt')
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

        #save