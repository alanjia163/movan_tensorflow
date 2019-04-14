#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import tensorflow as tf
import  numpy as np
import  gym

tf.set_random_seed(1)
np.random.seed(1)

#pyper parameters
BATCH_SIZE=32   #
LR =0.01        #
EPSILON =0.9    #
GAMMA =0.9

