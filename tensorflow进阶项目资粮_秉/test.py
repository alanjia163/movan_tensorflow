#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import  tensorflow as tf

#
def myregression():
    '''
    d自定义一个线性回归
    :return: None
    '''
    #data
    with tf.variable_scope('data'):
        x = tf.random_normal([100,1],mean=1.11,stddev=0.5,name='x_data')
        y_true = tf.matmul(x,[[0.7]])+0.8
    #建立权值和偏置,y=xw+b
    with tf.variable_scope('W_and_b'):
        W = tf.Variable(tf.random_normal([1,1],mean=0,stddev=1),name='W')
        b = tf.Variable(0.0,name='b')
    #add_layers
    with tf.variable_scope('layers_prediction'):
        prediction = tf.matmul(x,W)+b
    #loss
    with tf.variable_scope('loss'):
        loss = tf.losses.mean_squared_error(prediction,y_true)
    #optimizer
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.05)
    #train_op
    with tf.variable_scope('train_op'):
        train_op = optimizer.minimize((loss))

    #collect tensor
    tf.summary.scalar('loss',loss)#用于收集一维标量
    tf.summary.histogram('weights',W)#用于收集tensor
    #合并写入变量,定义OP
    merged =tf.summary.merge_all()
    #sess and init
             #sess = tf.Session()
    #init
    init = tf.global_variables_initializer()
    #sess.run
    with tf.Session() as sess:
        sess.run(init)
        for step in range(500):
            sess.run(train_op)
            #运行合并OP
            summary = tf.run(merged)
            #建立一个写入事件文件
            filewriter = tf.summary.FileWriter('./data/summary',graph=sess.graph)
            filewriter.add_summary(summary,step)
            #print
            if step % 10 == 0:
                #print
                print(step,W.eval(),b.eval())
    return None
if __name__ == '__main__':
    myregression()