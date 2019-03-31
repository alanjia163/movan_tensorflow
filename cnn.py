import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# hyper parameter,
BATCH_SIZE = 50
LR = 0.001

# load data
mnist = input_data.read_data_sets('./mnist', one_hot=True)  # nomalized to range(0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot example
print(mnist.train.images.shape)  # (55000,28*28)
print(mnist.train.labels.shape)  # (55000,10)

# placeholders
tf_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255
tf_y = tf.placeholder(tf.int32, [None, 10])  # input y

# reshape,因为卷积的入口中input的输入格式,
# input的要求格式是[batch_size] + input_spatial_shape + [in_channels],
# 也就是要求第一维是batch，最后一维是channel，中间是真正的卷积维度
image = tf.reshape(tf_x, [-1, 28, 28, 1])

# CNN
# CNN
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)  # shape->(28,28,16)
