#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

ACTIVATION = tf.nn.relu
N_LAYERS =7
N_HIDDEN_UNITS = 30


def fix_seed(seed =1):
    #
    np.random.seed(seed)
    tf.set_random_seed(seed)

def plot_his(inputs,inputs_norm):
    #plot histogram for hte inputs of every layer
    for j ,all_inputs in enumerate([inputs,inputs_norm]):
        for i ,inputs in enumerate(all_inputs):
            plt.subplot(2,len(all_inputs),j*len(all_inputs)+(i+1))
            plt.cla()

            if i == 0:
                the_range = (-7,10)
                else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
        plt.draw()
        plt.pause(0.01)


def build_net(xs,ys,norm):
    def add_layer(inputs,in_size,out_size,activation_function =None,norm=False):
        Weights =tf.Variable(tf.random_normal([in_size,out_size],mean=0,stddev=1.))
        biases = tf.Variable(tf.zeros([1,out_size])+0.1)

        #fully connected product
        Wx_plus_b = tf.matmul(input(s,Weights)+biases)

        #normalize fully connected product
        if norm:
            #batch Normalize
            fc_mean,fc_var = tf.nn.moments(

            )
