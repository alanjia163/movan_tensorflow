#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data


#data
mnist =input_data.read_data_sets("MNIST_data",one_hot=True)

#hyper parameter
batch_size =100  #每个批次的大小
n_batch = mnist.train.num_examples // batch_size  #多少个batch
LR = 0.01
#placeholders
x= tf.placeholder(tf.float32,shape=784)
y = tf.placeholder(tf.float32,shape=10) #

#add_layers
W = tf.Variable(tf.zeros(shape=[784,10]))
b = tf.Variable(tf.zeros(shape=[10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#loss and train_op
loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_op = tf.train.AdamOptimizer(LR).minimize(loss)


#accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy  =tf.reduce_mean(correct_prediction,tf.float32)

#init_op
init = tf.global_variables_initializer()

#sess
    #init
    #train_epoch_batch
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
