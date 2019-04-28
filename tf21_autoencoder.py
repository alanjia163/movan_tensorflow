#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jia ShiLin

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

#data
mnist = input_data.read_data_sets('../MNIST_data',one_hot=True)

#hyper parameter
LR = 0.01
trainning_epochs = 5
batch_size =256
batch = mnist.train.num_examples//batch_size
display_step =1
examples_to_show =10


n_input =28*28
#hidden layers_settings
n_hidden_1 = 256#lst layer number features
n_hidden_2 = 128#2nd layer number features

#placeholders
X =tf.placeholder(tf.float32,shape=[None,n_input])


#W and b
weights = {#dict
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

bias = {
    'encoders_b1':tf.Variable(tf.random_normal(shape=[n_hidden_1])),
    'encoders_b2':tf.Variable(tf.random_normal(shape=[n_hidden_2])),
    'decoders_b1':tf.Variable(tf.random_normal(shape=[n_hidden_1])),
    'decoders_b2':tf.Variable(tf.random_normal(shape=[n_input]))

}

#encoder and decoder
def encoder(x):
    layer_1=tf.nn.sigmoid(tf.matmul(x,weights['encoder_h1'])+bias['encoders_b1'])
    layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights['encoder_h2'])+bias['encoders_b2'])
    return layer_2

def decoder(x):
    layer_1=tf.nn.sigmoid(tf.matmul(x,weights['decoder_h1'])+bias['decoders_b1'])
    layer_2=tf.nn.sigmoid(tf.matmul(layer_1,weights['decoder_h2'])+bias['decoders_b2'])
    return layer_2


#add layers
encoder_op =encoder(X)
decoder =decoder(encoder_op)

#predictin
y_pred =decoder

#loss and train_op
loss = tf.losses.mean_squared_error(y_pred,X)
train_op =tf.train.AdamOptimizer(LR).minimize(loss)

#train_step
with tf.Session() as sess:
    #init
    init  = tf.global_variables_initializer()
    sess.run(init)

    #epoch and iter
    for epoch in range(trainning_epochs):
        for i in range(batch):
            batch_xs,batch_ys =mnist.train.next_batch(batch_size)
            _,cost =sess.run([train_op,loss],feed_dict={X:batch_xs})

        #display logs per epoch step
        print('Epoch:','%04d'%(epoch+1),'loss:','%.9f'%cost)

    print('Finished')


    # # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()

    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()














