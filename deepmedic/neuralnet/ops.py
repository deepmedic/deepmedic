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

def applyDropout(rng, dropoutRate, inputTrain, inputVal, inputTest) :
    if dropoutRate > 0.001 : #Below 0.001 I take it as if there is no dropout at all. (To avoid float problems with == 0.0. Although my tries show it actually works fine.)
        keep_prob = (1-dropoutRate)
        
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(shape=tf.shape(inputTrain), minval=0., maxval=1., seed=rng.randint(999999), dtype="float32")
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        dropoutMask = tf.floor(random_tensor)
    
        # tf.nn.dropout(x, keep_prob) scales kept values UP, so that at inference you dont need to scale then. 
        inputImgAfterDropoutTrain = inputTrain * dropoutMask
        inputImgAfterDropoutVal = inputVal * keep_prob
        inputImgAfterDropoutTest = inputTest * keep_prob
    else :
        inputImgAfterDropoutTrain = inputTrain
        inputImgAfterDropoutVal = inputVal
        inputImgAfterDropoutTest = inputTest
    return (inputImgAfterDropoutTrain, inputImgAfterDropoutVal, inputImgAfterDropoutTest)


def relu(input):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    return tf.maximum(0., input)

def prelu(input, a):
    # a = float (tf or np)
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

def createAndInitializeWeightsTensor(filterShape, convWInitMethod, rng) :
    # filterShape of dimensions: [#FMs in this layer, #FMs in input, rKernelDim, cKernelDim, zKernelDim]
    if convWInitMethod[0] == "normal" :
        stdForInit = convWInitMethod[1] # commonly 0.01 from Krizhevski
    elif convWInitMethod[0] == "fanIn" :
        varianceScale = convWInitMethod[1] # 2 for init ala Delving into Rectifier, 1 for SNN.
        stdForInit = np.sqrt( varianceScale / (filterShape[1] * filterShape[2] * filterShape[3] * filterShape[4]) )
        
    wInitNpArray = np.asarray( rng.normal(loc=0.0, scale=stdForInit, size=(filterShape[0],filterShape[1],filterShape[2],filterShape[3],filterShape[4])), dtype='float32' )
    W = tf.Variable( wInitNpArray, dtype="float32", name="W")
    # W shape: [#FMs of this layer, #FMs of Input, rKernFims, cKernDims, zKernDims]
    return W

def convolveWithGivenWeightMatrix(W, inputToConvTrain, inputToConvVal, inputToConvTest):
    # input weight matrix W has shape: [ #ChannelsOut, #ChannelsIn, R, C, Z ]
    # Input signal given in shape [BatchSize, Channels, R, C, Z]
    
    # Tensorflow's Conv3d requires filter shape: [ D/Z, H/C, W/R, C_in, C_out ] #ChannelsOut, #ChannelsIn, Z, R, C ]
    wReshapedForConv = tf.transpose(W, perm=[4,3,2,1,0])
    
    # Conv3d requires signal in shape: [BatchSize, Channels, Z, R, C]
    inputToConvReshapedTrain = tf.transpose(inputToConvTrain, perm=[0,4,3,2,1])
    outputOfConvTrain = tf.nn.conv3d(input = inputToConvReshapedTrain, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = wReshapedForConv, # TF: Depth, Height, Wight, Chans_in, Chans_out
                                  strides = [1,1,1,1,1],
                                  padding = "VALID",
                                  data_format = "NDHWC"
                                  )
    #Output is in the shape of the input image (signals_shape).
    outputTrain = tf.transpose(outputOfConvTrain, perm=[0,4,3,2,1]) #reshape the result, back to the shape of the input image.
    
    #Validation
    inputToConvReshapedVal = tf.transpose(inputToConvVal, perm=[0,4,3,2,1])
    outputOfConvVal = tf.nn.conv3d(input = inputToConvReshapedVal,
                                  filters = wReshapedForConv,
                                  strides = [1,1,1,1,1],
                                  padding = "VALID",
                                  data_format = "NDHWC"
                                  )
    outputVal = tf.transpose(outputOfConvVal, perm=[0,4,3,2,1])
    
    #Testing
    inputToConvReshapedTest = tf.transpose(inputToConvTest, perm=[0,4,3,2,1])
    outputOfConvTest = tf.nn.conv3d(input = inputToConvReshapedTest,
                                  filters = wReshapedForConv,
                                  strides = [1,1,1,1,1],
                                  padding = "VALID",
                                  data_format = "NDHWC"
                                  )
    outputTest = tf.transpose(outputOfConvTest, perm=[0,4,3,2,1])
    
    return (outputTrain, outputVal, outputTest)


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

