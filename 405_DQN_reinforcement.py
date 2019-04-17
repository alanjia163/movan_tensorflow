#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
import  numpy as np
import  gym

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
MEMORY_COUNTER = 0          # for store experience
LEARNING_STEP_COUNTER = 0   # for target updating
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
MEMORY = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory

# tf placeholders
tf_s = tf.placeholder(tf.float32, [None, N_STATES])
tf_a = tf.placeholder(tf.int32, [None, ])
tf_r = tf.placeholder(tf.float32, [None, ])
tf_s_ = tf.placeholder(tf.float32, [None, N_STATES])

with tf.variable_scope('q'):        # evaluation network
    l_eval = tf.layers.dense(tf_s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
    q = tf.layers.dense(l_eval, N_ACTIONS, kernel_initializer=tf.random_normal_initializer(0, 0.1))

with tf.variable_scope('q_next'):   # target network, not to train
    l_target = tf.layers.dense(tf_s_, 10, tf.nn.relu, trainable=False)
    q_next = tf.layers.dense(l_target, N_ACTIONS, trainable=False)

q_target = tf_r + GAMMA * tf.reduce_max(q_next, axis=1)                   # shape=(None, ),