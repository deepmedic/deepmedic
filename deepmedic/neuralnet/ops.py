# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

from math import ceil
import numpy as np
import random

import tensorflow as tf

try:
    from sys import maxint as MAX_INT
except ImportError:
    # python3 compatibility
    from sys import maxsize as MAX_INT


###############################################################
# Functions used by layers but do not change Layer Attributes #
###############################################################

def relu(input):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    return tf.maximum(0., input)

def prelu(input, a):
    # a = tensor of floats, [1, n_channels, 1, 1, 1]
    pos = tf.maximum(0., input)
    neg = a * (input - abs(input)) * 0.5
    return pos + neg

def elu(input):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    return tf.nn.elu(input)

def selu(input):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    lambda01 = 1.0507 # calc in p4 of paper.
    alpha01 = 1.6733 # WHERE IS THIS USED? I AM DOING SOMETHING WRONG I THINK.
    raise NotImplementedError()
    return lambda01 * tf.nn.elu(input)


# Currently only used for pooling3d
def mirrorFinalBordersOfImage(image3dBC012, mirrorFinalBordersForThatMuch) :
    image3dBC012WithMirrorPad = image3dBC012
    for time_i in range(0, mirrorFinalBordersForThatMuch[0]) :
        image3dBC012WithMirrorPad = tf.concat([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,-1:,:,:] ], axis=2)
    for time_i in range(0, mirrorFinalBordersForThatMuch[1]) :
        image3dBC012WithMirrorPad = tf.concat([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,:,-1:,:] ], axis=3)
    for time_i in range(0, mirrorFinalBordersForThatMuch[2]) :
        image3dBC012WithMirrorPad = tf.concat([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,:,:,-1:] ], axis=4)
    return image3dBC012WithMirrorPad


def pool3dMirrorPad(image3dBC012, poolParams) :
    # image3dBC012 dimensions: (batch, fms, r, c, z)
    # poolParams: [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode]
    ws = poolParams[0] # window size
    stride = poolParams[1] # stride
    mode1 = poolParams[3] # MAX or AVG
    
    image3dBC012WithMirrorPad = mirrorFinalBordersOfImage(image3dBC012, poolParams[2])
    
    pooled_out = tf.nn.pool( input = tf.transpose(image3dBC012WithMirrorPad, perm=[0,4,3,2,1]),
                            window_shape=ws,
                            strides=stride,
                            padding="VALID", # SAME or VALID
                            pooling_type=mode1,
                            data_format="NDHWC") # AVG or MAX
    pooled_out = tf.transpose(pooled_out, perm=[0,4,3,2,1])
    
    return pooled_out

