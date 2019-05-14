#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf

state = tf.Variable(0,name='counter')
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

#sess and init
with tf.Session() as sess:
    init = tf.global_variables_initializer()#由于有新建变量
    sess.run(init)

    for epoch in range(3):
        sess.run(update)
        print(sess.run(state))

