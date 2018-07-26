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

def applyDropout(rng, dropoutRate, inputTrainShape, inputTrain, inputVal, inputTest) :
    if dropoutRate > 0.001 : #Below 0.001 I take it as if there is no dropout at all. (To avoid float problems with == 0.0. Although my tries show it actually works fine.)
        keep_prob = (1-dropoutRate)
        
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(shape=inputTrainShape, minval=0., maxval=1., seed=rng.randint(999999), dtype="float32")
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


def applyBn(rollingAverageForBatchNormalizationOverThatManyBatches, inputTrain, inputVal, inputTest, inputShapeTrain) :
    numOfChanns = inputShapeTrain[1]
    
    gBn = tf.Variable( np.ones( (numOfChanns), dtype='float32'), name="gBn" )
    bBn = tf.Variable( np.zeros( (numOfChanns), dtype='float32'), name="bBn" )
    gBn_resh = tf.reshape(gBn, shape=[1,numOfChanns,1,1,1])
    bBn_resh = tf.reshape(bBn, shape=[1,numOfChanns,1,1,1])
    
    #for rolling average:
    muBnsArrayForRollingAverage = tf.Variable( np.zeros( (rollingAverageForBatchNormalizationOverThatManyBatches, numOfChanns), dtype='float32' ), name="muBnsForRollingAverage" )
    varBnsArrayForRollingAverage = tf.Variable( np.ones( (rollingAverageForBatchNormalizationOverThatManyBatches, numOfChanns), dtype='float32' ), name="varBnsForRollingAverage" )        
    sharedNewMu_B = tf.Variable(np.zeros( (numOfChanns), dtype='float32'), name="sharedNewMu_B")
    sharedNewVar_B = tf.Variable(np.ones( (numOfChanns), dtype='float32'), name="sharedNewVar_B")
    
    e1 = np.finfo(np.float32).tiny 
    
    mu_B, var_B = tf.nn.moments(inputTrain, axes=[0,2,3,4])
    
    #---computing mu and var for inference from rolling average---
    mu_MoveAv = tf.reduce_mean(muBnsArrayForRollingAverage, axis=0)
    mu_MoveAv = tf.reshape(mu_MoveAv, shape=[1,numOfChanns,1,1,1])
    effectiveSize = inputShapeTrain[0]*inputShapeTrain[2]*inputShapeTrain[3]*inputShapeTrain[4] #batchSize*voxels in a featureMap. See p5 of the paper.
    var_MoveAv = (effectiveSize/(effectiveSize-1)) * tf.reduce_mean(varBnsArrayForRollingAverage, axis=0)
    var_MoveAv = var_MoveAv + e1
    var_MoveAv = tf.reshape(var_MoveAv, shape=[1,numOfChanns,1,1,1])
    
    #OUTPUT FOR TRAINING
    mu_B_resh = tf.reshape(mu_B, shape=[1,numOfChanns,1,1,1])
    var_B_resh = tf.reshape(var_B, shape=[1,numOfChanns,1,1,1])
    normXi_train = (inputTrain - mu_B_resh ) /  tf.sqrt(var_B_resh + e1) # e1 should come OUT of the sqrt! 
    normYi_train = gBn_resh * normXi_train + bBn_resh
    #OUTPUT FOR VALIDATION
    normXi_val = (inputVal - mu_MoveAv) /  tf.sqrt(var_MoveAv) 
    normYi_val = gBn_resh * normXi_val + bBn_resh
    #OUTPUT FOR TESTING
    normXi_test = (inputTest - mu_MoveAv) /  tf.sqrt(var_MoveAv) 
    normYi_test = gBn_resh * normXi_test + bBn_resh
    
    return (normYi_train,
            normYi_val,
            normYi_test,
            gBn,
            bBn,
            # For rolling average
            muBnsArrayForRollingAverage,
            varBnsArrayForRollingAverage,
            sharedNewMu_B,
            sharedNewVar_B,
            mu_B, # this is the current value of muB calculated in this training iteration. It will be saved in the "sharedNewMu_B" (update), in order to be used for updating the rolling average. Something could be simplified here.
            var_B
            )
    
    
def makeBiasParamsAndApplyToFms( fmsTrain, fmsVal, fmsTest, numberOfFms ) :
    b_values = np.zeros( (numberOfFms), dtype = 'float32')
    b = tf.Variable(b_values, name="b")
    b_resh = tf.reshape(b, shape=[1,numberOfFms,1,1,1])
    fmsWithBiasAppliedTrain = fmsTrain + b_resh
    fmsWithBiasAppliedVal = fmsVal + b_resh
    fmsWithBiasAppliedTest = fmsTest + b_resh
    return (b, fmsWithBiasAppliedTrain, fmsWithBiasAppliedVal, fmsWithBiasAppliedTest)

def applyRelu(inputTrain, inputVal, inputTest):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    outputTrain= tf.maximum(0., inputTrain)
    outputVal = tf.maximum(0., inputVal)
    outputTest = tf.maximum(0., inputTest)
    return ( outputTrain, outputVal, outputTest )

def applyPrelu( inputTrain, inputVal, inputTest, numberOfInputChannels ) :
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    aPreluValues = np.ones( (numberOfInputChannels), dtype = 'float32' ) * 0.01 #"Delving deep into rectifiers" initializes it like this. LeakyRelus are at 0.01
    aPrelu = tf.Variable(aPreluValues, name="aPrelu") #One separate a (activation) per feature map.
    aPrelu5D = tf.reshape(aPrelu, shape=[1, numberOfInputChannels, 1, 1, 1] )
    
    posTrain = tf.maximum(0., inputTrain)
    negTrain = aPrelu5D * (inputTrain - abs(inputTrain)) * 0.5
    outputTrain = posTrain + negTrain
    posVal = tf.maximum(0., inputVal)
    negVal = aPrelu5D * (inputVal - abs(inputVal)) * 0.5
    outputVal = posVal + negVal
    posTest = tf.maximum(0., inputTest)
    negTest = aPrelu5D * (inputTest - abs(inputTest)) * 0.5
    outputTest = posTest + negTest
    
    return ( aPrelu, outputTrain, outputVal, outputTest )

def applyElu(inputTrain, inputVal, inputTest):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    outputTrain = tf.nn.elu(inputTrain)
    outputVal = tf.nn.elu(inputVal)
    outputTest = tf.nn.elu(inputTest)
    return ( outputTrain, outputVal, outputTest )

def applySelu(inputTrain, inputVal, inputTest):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    lambda01 = 1.0507 # calc in p4 of paper.
    alpha01 = 1.6733
    
    outputTrain = lambda01 * tf.nn.elu(inputTrain)
    outputVal = lambda01 *  tf.nn.elu(inputVal)
    outputTest = lambda01 * tf.nn.elu(inputTest)
    
    return ( outputTrain, outputVal, outputTest )

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

def convolveWithGivenWeightMatrix(W, filterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) :
    # input weight matrix W has shape: [ #ChannelsOut, #ChannelsIn, R, C, Z ] == filterShape
    # filterShape is the shape of W.
    # Input signal given in shape [BatchSize, Channels, R, C, Z]
    
    # Tensorflow's Conv3d requires filter shape: [ D/Z, H/C, W/R, C_in, C_out ] #ChannelsOut, #ChannelsIn, Z, R, C ]
    wReshapedForConv = tf.transpose( W, perm=[4,3,2,1,0] )
    
    # Conv3d requires signal in shape: [BatchSize, Channels, Z, R, C]
    inputToConvReshapedTrain = tf.transpose( inputToConvTrain, perm=[0,4,3,2,1] )
    outputOfConvTrain = tf.nn.conv3d(input = inputToConvReshapedTrain, # batch_size, time, num_of_input_channels, rows, columns
                                  filter = wReshapedForConv, # TF: Depth, Height, Wight, Chans_in, Chans_out
                                  strides = [1,1,1,1,1],
                                  padding = "VALID",
                                  data_format = "NDHWC"
                                  )
    #Output is in the shape of the input image (signals_shape).
    outputTrain = tf.transpose( outputOfConvTrain, perm=[0,4,3,2,1] ) #reshape the result, back to the shape of the input image.
    
    #Validation
    inputToConvReshapedVal = tf.transpose( inputToConvVal, perm=[0,4,3,2,1] )
    outputOfConvVal = tf.nn.conv3d(input = inputToConvReshapedVal,
                                  filter = wReshapedForConv,
                                  strides = [1,1,1,1,1],
                                  padding = "VALID",
                                  data_format = "NDHWC"
                                  )
    outputVal = tf.transpose( outputOfConvVal, perm=[0,4,3,2,1] )
    
    #Testing
    inputToConvReshapedTest = tf.transpose( inputToConvTest, perm=[0,4,3,2,1] )
    outputOfConvTest = tf.nn.conv3d(input = inputToConvReshapedTest,
                                  filter = wReshapedForConv,
                                  strides = [1,1,1,1,1],
                                  padding = "VALID",
                                  data_format = "NDHWC"
                                  )
    outputTest = tf.transpose( outputOfConvTest, perm=[0,4,3,2,1] )
    
    outputShapeTrain = [inputToConvShapeTrain[0],
                        filterShape[0],
                        inputToConvShapeTrain[2]-filterShape[2]+1,
                        inputToConvShapeTrain[3]-filterShape[3]+1,
                        inputToConvShapeTrain[4]-filterShape[4]+1]
    outputShapeVal = [  inputToConvShapeVal[0],
                        filterShape[0],
                        inputToConvShapeVal[2]-filterShape[2]+1,
                        inputToConvShapeVal[3]-filterShape[3]+1,
                        inputToConvShapeVal[4]-filterShape[4]+1]
    outputShapeTest = [ inputToConvShapeTest[0],
                        filterShape[0],
                        inputToConvShapeTest[2]-filterShape[2]+1,
                        inputToConvShapeTest[3]-filterShape[3]+1,
                        inputToConvShapeTest[4]-filterShape[4]+1]
    
    return (outputTrain, outputVal, outputTest, outputShapeTrain, outputShapeVal, outputShapeTest)

        
def applySoftmaxToFmAndReturnProbYandPredY( inputToSoftmax, inputToSoftmaxShape, numberOfOutputClasses, softmaxTemperature):
    # The softmax function works on 2D tensors (matrices). It computes the softmax for each row. Rows are independent, eg different samples in the batch. Columns are the input features, eg class-scores.
    # Softmax's input 2D matrix should have shape like: [ datasamples, #Classess ]
    # My class-scores/class-FMs are a 5D tensor (batchSize, #Classes, r, c, z).
    # I need to reshape it to a 2D tensor.
    # The reshaped 2D Tensor will have dimensions: [ batchSize * r * c * z , #Classses ]
    # The order of the elements in the rows after the reshape should be :
    
    inputToSoftmaxReshaped = tf.transpose(inputToSoftmax, perm=[0,2,3,4,1]) # [batchSize, r, c, z, #classes), the classes stay as the last dimension.
    inputToSoftmaxFlattened = tf.reshape(inputToSoftmaxReshaped, shape=[-1]) 
    # flatten is "Row-major" 'C' style. ie, starts from index [0,0,0] and grabs elements in order such that last dim index increases first and first index increases last. (first row flattened, then second follows, etc)
    numberOfVoxelsDenselyClassified = inputToSoftmaxShape[2]*inputToSoftmaxShape[3]*inputToSoftmaxShape[4]
    firstDimOfInputToSoftmax2d = inputToSoftmaxShape[0]*numberOfVoxelsDenselyClassified # batchSize*r*c*z.
    inputToSoftmax2d = tf.reshape(inputToSoftmaxFlattened, shape=[firstDimOfInputToSoftmax2d, numberOfOutputClasses]) # Reshape works in "Row-major", ie 'C' style too.
    # Predicted probability per class.
    p_y_given_x_2d = tf.nn.softmax(inputToSoftmax2d/softmaxTemperature, axis=-1)
    p_y_given_x_classMinor = tf.reshape(p_y_given_x_2d, shape=[inputToSoftmaxShape[0], inputToSoftmaxShape[2], inputToSoftmaxShape[3], inputToSoftmaxShape[4], inputToSoftmaxShape[1]]) #Result: batchSize, R,C,Z, Classes.
    p_y_given_x = tf.transpose(p_y_given_x_classMinor, perm=[0,4,1,2,3]) #Result: batchSize, Class, R, C, Z
    
    # Classification (EM) for each voxel
    y_pred = tf.argmax(p_y_given_x, axis=1) #Result: batchSize, R, C, Z
    
    return ( p_y_given_x, y_pred )


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


def pool3dMirrorPad(image3dBC012, image3dBC012Shape, poolParams) :
    # image3dBC012 dimensions: (batch, fms, r, c, z)
    # poolParams: [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode]
    ws = poolParams[0] # window size
    stride = poolParams[1] # stride
    mode1 = poolParams[3] # MAX or AVG
    
    image3dBC012WithMirrorPad = mirrorFinalBordersOfImage(image3dBC012, poolParams[2])
    
    pooled_out = tf.nn.pool( input = tf.transpose( image3dBC012WithMirrorPad, perm=[0,4,3,2,1] ),
                            window_shape=ws,
                            strides=stride,
                            padding="VALID", # SAME or VALID
                            pooling_type=mode1,
                            data_format="NDHWC") # AVG or MAX
    pooled_out = tf.transpose( pooled_out, perm=[0,4,3,2,1] )
    
    #calculate the shape of the image after the max pooling.
    #This calculation is for ignore_border=True! Pooling should only be done in full areas in the mirror-padded image.
    imgShapeAfterPoolAndPad = [ image3dBC012Shape[0],
                                image3dBC012Shape[1],
                                int(ceil( (image3dBC012Shape[2] + poolParams[2][0] - ds[0] + 1) / (1.0*stride[0])) ),
                                int(ceil( (image3dBC012Shape[3] + poolParams[2][1] - ds[1] + 1) / (1.0*stride[1])) ),
                                int(ceil( (image3dBC012Shape[4] + poolParams[2][2] - ds[2] + 1) / (1.0*stride[2])) )
                            ]
    return (pooled_out, imgShapeAfterPoolAndPad)

