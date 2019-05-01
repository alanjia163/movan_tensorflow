#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#hyper parameter
batch_size =100
n_batch =mnist.train.num_examples // batch_size

#placeholder
x = tf.placeholder(tf.float32,[None,784],name = 'x_input')
y = tf.placeholder(tf.float32,[None,10])

#add_layer
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b,name = 'output')


#loss and train_step
loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


#accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#sess and init
with tf.Session() as sess:
    init  = tf.group(tf.global_variables_initializer())
    sess.run(init)

    #epoch
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})

        print('iter'+str(epoch)+',testing accruacy'+str(acc))

    #保存模型参数和结构
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['output'])
    #保存模型到目录下 的model文件夹中

    with tf.gfile.FastGFile('./Save_Net/tfmodel.pb',mode='wb') as f:
        f.write(output_graph_def.SerializeToString())



