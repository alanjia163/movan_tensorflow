#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#seed
tf.random.set_random_seed(1)

#load data
mnist = input_data.read_data_sets('MNIST.data',one_hot=True)

#hyper parameter
lr = 0.01
tarining_iters = 100
batch_size =128
n_inputs =28
n_steps =28
n_hidden_units =128
n_classes =10


#placeholders
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

#W
weight ={
    #(28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    #(128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases = {
    #(128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,1]))
    #(10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X,weights,biases):
    #hidden layer
    #input_reshape - X->(128 batch * 28 steps, 28 inputs)
    X =tf.reshape(X,[-1,n_inputs])
    #hidden1
    X_in = tf.matmul(X,weights['in'])+biases['in']
        #X_in->(128 batch,28 steps,128 hidden)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])#reshape again shape=[-1,n_steps,n_hidden_units]


    #cell
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    #lstm cell is divides into two parts(c_state,h_state)
    init_state =cell.zero_state(batch_size,dtype=tf.float32)
    outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
    results = tf.matmul(outputs[-1],weight['out']+biases['out'])#shape = [128,10]
    return results

pred = RNN(x,weights,biases)
#loss and train_op
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

#accuracy
correct_pred = tf.equal


