#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
x = np.linspace(-5, 5, 200)  # shape(100,1)

# following are popular activation functions
y_relu = tf.nn.relu(x)

# y_softmax = tf.nn.sotfmax(x),sess
sess = tf.Session()
y_relu = sess.run(y_relu)
# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.show()
