#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()

x = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


def add_layer(inputs, in_size, out_size, layer_name, activation=None):
    '''

    :param inputs:
    :param in_size:
    :param out_size:
    :param layer_name:
    :param activation:
    :return: outputs
    '''

    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    W_plus_b = tf.matmul(inputs, Weights) + biases
    # dropout_layer
    W_plus_b = tf.nn.dropout(W_plus_b, keep_prob=keep_prob)

    if activation is None:
        outputs = W_plus_b
    else:
        outputs = activation(W_plus_b)
    return outputs


# placeholders
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

# layers
l1 = add_layer(xs, 64, 50, 'l1', activation=tf.nn.relu)
prediction = add_layer(l1, 50, 10, 'l2', activation=tf.nn.softmax)

# loss and train op
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()  # 收集
#summary writer
train_writer = tf.summary.FileWriter('loss/train',sess.graph)
test_writer = tf.summary.FileWriter('loss/train',sess.graph)

#init
init = tf.global_variables_initializer()
sess.run(init)

#train_step
for i  in range(500):
    sess.run(train_step,feed_dict={xs:x_train,ys:y_train,keep_prob : 0.7})

    if i %50 == 0:
        #record loss此时正则化比例为1，不需要正则化
        train_result = sess.run(merged,feed_dict={xs:x_train,ys:y_train,keep_prob : 1})
        test_result = sess.run(merged,feed_dict={xs:x_test,ys:y_test,keep_prob : 1})

        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)





