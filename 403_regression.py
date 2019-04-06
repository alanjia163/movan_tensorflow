#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin


#import
import tensorflow as tf
import  numpy as np
from   matplotlib import  pyplot as plt

#hyper parameter
TIME_STEP =10
INPUT_SIZE =1
CELL_SIZE =32
LR = 0.02

#show data
steps = np.linspace(0,np.pi*2,100,dtype=np.float32)

