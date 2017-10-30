# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange

from math import ceil
import numpy as np
import random

import theano
import theano.tensor as T

try:
    from sys import maxint as MAX_INT
except ImportError:
    # python3 compatibility
    from sys import maxsize as MAX_INT


###############################################################
# Functions used by layers but do not change Layer Attributes #
###############################################################

def applyDropout(rng, dropoutRate, inputTrainShape, inputTrain, inputInference, inputTesting) :
    if dropoutRate > 0.001 : #Below 0.001 I take it as if there is no dropout at all. (To avoid float problems with == 0.0. Although my tries show it actually works fine.)
        probabilityOfStayingActivated = (1-dropoutRate)
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        dropoutMask = srng.binomial(n=1, size=inputTrainShape, p=probabilityOfStayingActivated, dtype=theano.config.floatX)
        inputImgAfterDropout = inputTrain * dropoutMask
        inputImgAfterDropoutInference = inputInference * probabilityOfStayingActivated
        inputImgAfterDropoutTesting = inputTesting * probabilityOfStayingActivated
    else :
        inputImgAfterDropout = inputTrain
        inputImgAfterDropoutInference = inputInference
        inputImgAfterDropoutTesting = inputTesting
    return (inputImgAfterDropout, inputImgAfterDropoutInference, inputImgAfterDropoutTesting)


def applyBn(rollingAverageForBatchNormalizationOverThatManyBatches, inputTrain, inputVal, inputTest, inputShapeTrain) :
    numberOfChannels = inputShapeTrain[1]
    
    gBn_values = np.ones( (numberOfChannels), dtype = 'float32' )
    gBn = theano.shared(value=gBn_values, borrow=True)
    bBn_values = np.zeros( (numberOfChannels), dtype = 'float32')
    bBn = theano.shared(value=bBn_values, borrow=True)
    
    #for rolling average:
    muBnsArrayForRollingAverage = theano.shared(np.zeros( (rollingAverageForBatchNormalizationOverThatManyBatches, numberOfChannels), dtype = 'float32' ), borrow=True)
    varBnsArrayForRollingAverage = theano.shared(np.ones( (rollingAverageForBatchNormalizationOverThatManyBatches, numberOfChannels), dtype = 'float32' ), borrow=True)
    sharedNewMu_B = theano.shared(np.zeros( (numberOfChannels), dtype = 'float32'), borrow=True)
    sharedNewVar_B = theano.shared(np.ones( (numberOfChannels), dtype = 'float32'), borrow=True)
    
    e1 = np.finfo(np.float32).tiny 
    #WARN, PROBLEM, THEANO BUG. The below was returning (True,) instead of a vector, if I have only 1 FM. (Vector is (False,)). Think I corrected this bug.
    mu_B = inputTrain.mean(axis=[0,2,3,4]) #average over all axis but the 2nd, which is the FM axis.
    mu_B = T.unbroadcast(mu_B, (0)) #The above was returning a broadcastable (True,) tensor when FM-number=1. Here I make it a broadcastable (False,), which is the "vector" type. This is the same type with the sharedNewMu_B, which we are updating with this. They need to be of the same type.
    var_B = inputTrain.var(axis=[0,2,3,4])
    var_B = T.unbroadcast(var_B, (0))
    var_B_plusE = var_B + e1
    
    #---computing mu and var for inference from rolling average---
    mu_RollingAverage = muBnsArrayForRollingAverage.mean(axis=0)
    effectiveSize = inputShapeTrain[0]*inputShapeTrain[2]*inputShapeTrain[3]*inputShapeTrain[4] #batchSize*voxels in a featureMap. See p5 of the paper.
    var_RollingAverage = (effectiveSize/(effectiveSize-1))*varBnsArrayForRollingAverage.mean(axis=0)
    var_RollingAverage_plusE = var_RollingAverage + e1
    
    #OUTPUT FOR TRAINING
    normXi_train = (inputTrain - mu_B.dimshuffle('x', 0, 'x', 'x', 'x')) /  T.sqrt(var_B_plusE.dimshuffle('x', 0, 'x', 'x', 'x')) 
    normYi_train = gBn.dimshuffle('x', 0, 'x', 'x', 'x') * normXi_train + bBn.dimshuffle('x', 0, 'x', 'x', 'x') # dimshuffle makes b broadcastable.
    #OUTPUT FOR VALIDATION
    normXi_val = (inputVal - mu_RollingAverage.dimshuffle('x', 0, 'x', 'x', 'x')) /  T.sqrt(var_RollingAverage_plusE.dimshuffle('x', 0, 'x', 'x', 'x')) 
    normYi_val = gBn.dimshuffle('x', 0, 'x', 'x', 'x') * normXi_val + bBn.dimshuffle('x', 0, 'x', 'x', 'x')
    #OUTPUT FOR TESTING
    normXi_test = (inputTest - mu_RollingAverage.dimshuffle('x', 0, 'x', 'x', 'x')) /  T.sqrt(var_RollingAverage_plusE.dimshuffle('x', 0, 'x', 'x', 'x')) 
    normYi_test = gBn.dimshuffle('x', 0, 'x', 'x', 'x') * normXi_test + bBn.dimshuffle('x', 0, 'x', 'x', 'x')
    
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
    b = theano.shared(value=b_values, borrow=True)
    fmsWithBiasAppliedTrain = fmsTrain + b.dimshuffle('x', 0, 'x', 'x', 'x')
    fmsWithBiasAppliedVal = fmsVal + b.dimshuffle('x', 0, 'x', 'x', 'x')
    fmsWithBiasAppliedTest = fmsTest + b.dimshuffle('x', 0, 'x', 'x', 'x')
    return (b, fmsWithBiasAppliedTrain, fmsWithBiasAppliedVal, fmsWithBiasAppliedTest)

def applyRelu(inputTrain, inputVal, inputTest):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    outputTrain= T.maximum(0, inputTrain)
    outputVal = T.maximum(0, inputVal)
    outputTest = T.maximum(0, inputTest)
    return ( outputTrain, outputVal, outputTest )

def applyPrelu( inputTrain, inputVal, inputTest, numberOfInputChannels ) :
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    aPreluValues = np.ones( (numberOfInputChannels), dtype = 'float32' )*0.01 #"Delving deep into rectifiers" initializes it like this. LeakyRelus are at 0.01
    aPrelu = theano.shared(value=aPreluValues, borrow=True) #One separate a (activation) per feature map.
    aPreluBroadCastedForMultiplWithChannels = aPrelu.dimshuffle('x', 0, 'x', 'x', 'x')
    
    posTrain = T.maximum(0, inputTrain)
    negTrain = aPreluBroadCastedForMultiplWithChannels * (inputTrain - abs(inputTrain)) * 0.5
    outputTrain = posTrain + negTrain
    posVal = T.maximum(0, inputVal)
    negVal = aPreluBroadCastedForMultiplWithChannels * (inputVal - abs(inputVal)) * 0.5
    outputVal = posVal + negVal
    posTest = T.maximum(0, inputTest)
    negTest = aPreluBroadCastedForMultiplWithChannels * (inputTest - abs(inputTest)) * 0.5
    outputTest = posTest + negTest
    
    return ( aPrelu, outputTrain, outputVal, outputTest )

def applyElu(alpha, inputTrain, inputVal, inputTest):
    outputTrain = T.basic.switch(inputTrain > 0, inputTrain, alpha * T.basic.expm1(inputTrain))
    outputVal = T.basic.switch(inputVal > 0, inputVal, alpha * T.basic.expm1(inputVal))
    outputTest = T.basic.switch(inputTest > 0, inputTest, alpha * T.basic.expm1(inputTest))
    return ( outputTrain, outputVal, outputTest )

def applySelu(inputTrain, inputVal, inputTest):
    #input is a tensor of shape (batchSize, FMs, r, c, z)
    lambda01 = 1.0507 # calc in p4 of paper.
    alpha01 = 1.6733
    
    ( outputTrain, outputVal, outputTest ) = applyElu(alpha01, inputTrain, inputVal, inputTest)
    outputTrain = lambda01 * outputTrain
    outputVal = lambda01 *  outputVal
    outputTest = lambda01 * outputTest
    
    return ( outputTrain, outputVal, outputTest )

def createAndInitializeWeightsTensor(filterShape, convWInitMethod, rng) :
    # filterShape of dimensions: [#FMs in this layer, #FMs in input, rKernelDim, cKernelDim, zKernelDim]
    if convWInitMethod[0] == "normal" :
        stdForInit = convWInitMethod[1] # commonly 0.01 from Krizhevski
    elif convWInitMethod[0] == "fanIn" :
        varianceScale = convWInitMethod[1] # 2 for init ala Delving into Rectifier, 1 for SNN.
        stdForInit = np.sqrt( varianceScale / (filterShape[1] * filterShape[2] * filterShape[3] * filterShape[4]) )
        
    # Perhaps I want to use: theano.config.floatX in the below
    wInitNpArray = np.asarray( rng.normal(loc=0.0, scale=stdForInit, size=(filterShape[0],filterShape[1],filterShape[2],filterShape[3],filterShape[4])), dtype='float32' )
    W = theano.shared( wInitNpArray, borrow=True )
    # W shape: [#FMs of this layer, #FMs of Input, rKernFims, cKernDims, zKernDims]
    return W

def convolveWithGivenWeightMatrix(W, filterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) :
    # input weight matrix W has shape: [ #ChannelsOut, #ChannelsIn, R, C, Z ] == filterShape
    # filterShape is the shape of W.
    # Input signal given in shape [BatchSize, Channels, R, C, Z]
    
    # Conv3d requires filter shape: [ #ChannelsOut, #ChannelsIn, Z, R, C ]
    wReshapedForConv = W.dimshuffle(0,1,4,2,3)
    wReshapedForConvShape = (filterShape[0], filterShape[1], filterShape[4], filterShape[2], filterShape[3])
    
    # Conv3d requires signal in shape: [BatchSize, Channels, Z, R, C]
    inputToConvReshapedTrain = inputToConvTrain.dimshuffle(0, 1, 4, 2, 3)
    inputToConvReshapedShapeTrain = (inputToConvShapeTrain[0], inputToConvShapeTrain[1], inputToConvShapeTrain[4], inputToConvShapeTrain[2], inputToConvShapeTrain[3]) # batch_size, time, num_of_input_channels, rows, columns
    outputOfConvTrain = T.nnet.conv3d(input = inputToConvReshapedTrain, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = wReshapedForConv, # Number_of_output_filters, Z, Numb_of_input_Channels, r, c
                                  input_shape = inputToConvReshapedShapeTrain, # Can be None. Used for optimization.
                                  filter_shape = wReshapedForConvShape, # Can be None. Used for optimization.
                                  border_mode = 'valid',
                                  subsample = (1,1,1), # strides
                                  filter_dilation=(1,1,1) # dilation rate
                                  )
    #Output is in the shape of the input image (signals_shape).
    
    #Validation
    inputToConvReshapedVal = inputToConvVal.dimshuffle(0, 1, 4, 2, 3)
    inputToConvReshapedShapeVal = (inputToConvShapeVal[0], inputToConvShapeVal[1], inputToConvShapeVal[4], inputToConvShapeVal[2], inputToConvShapeVal[3])
    outputOfConvVal = T.nnet.conv3d(input = inputToConvReshapedVal,
                                  filters = wReshapedForConv,
                                  input_shape = inputToConvReshapedShapeVal,
                                  filter_shape = wReshapedForConvShape,
                                  border_mode = 'valid',
                                  subsample = (1,1,1),
                                  filter_dilation=(1,1,1)
                                  )
    #Testing
    inputToConvReshapedTest = inputToConvTest.dimshuffle(0, 1, 4, 2, 3)
    inputToConvReshapedShapeTest = (inputToConvShapeTest[0], inputToConvShapeTest[1], inputToConvShapeTest[4], inputToConvShapeTest[2], inputToConvShapeTest[3])
    outputOfConvTest = T.nnet.conv3d(input = inputToConvReshapedTest,
                                  filters = wReshapedForConv,
                                  input_shape = inputToConvReshapedShapeTest,
                                  filter_shape = wReshapedForConvShape,
                                  border_mode = 'valid',
                                  subsample = (1,1,1),
                                  filter_dilation=(1,1,1)
                                  )
    
    outputTrain = outputOfConvTrain.dimshuffle(0, 1, 3, 4, 2) #reshape the result, back to the shape of the input image.
    outputVal = outputOfConvVal.dimshuffle(0, 1, 3, 4, 2)
    outputTest = outputOfConvTest.dimshuffle(0, 1, 3, 4, 2)
    
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
    
    inputToSoftmaxReshaped = inputToSoftmax.dimshuffle(0, 2, 3, 4, 1) # [batchSize, r, c, z, #classes), the classes stay as the last dimension.
    inputToSoftmaxFlattened = inputToSoftmaxReshaped.flatten(1) 
    # flatten is "Row-major" 'C' style. ie, starts from index [0,0,0] and grabs elements in order such that last dim index increases first and first index increases last. (first row flattened, then second follows, etc)
    numberOfVoxelsDenselyClassified = inputToSoftmaxShape[2]*inputToSoftmaxShape[3]*inputToSoftmaxShape[4]
    firstDimOfInputToSoftmax2d = inputToSoftmaxShape[0]*numberOfVoxelsDenselyClassified # batchSize*r*c*z.
    inputToSoftmax2d = inputToSoftmaxFlattened.reshape((firstDimOfInputToSoftmax2d, numberOfOutputClasses)) # Reshape works in "Row-major", ie 'C' style too.
    # Predicted probability per class.
    p_y_given_x_2d = T.nnet.softmax(inputToSoftmax2d/softmaxTemperature)
    p_y_given_x_classMinor = p_y_given_x_2d.reshape((inputToSoftmaxShape[0], inputToSoftmaxShape[2], inputToSoftmaxShape[3], inputToSoftmaxShape[4], inputToSoftmaxShape[1])) #Result: batchSize, R,C,Z, Classes.
    p_y_given_x = p_y_given_x_classMinor.dimshuffle(0,4,1,2,3) #Result: batchSize, Class, R, C, Z
    
    # Classification (EM) for each voxel
    y_pred = T.argmax(p_y_given_x, axis=1) #Result: batchSize, R, C, Z
    
    return ( p_y_given_x, y_pred )


# Currently only used for pooling3d
def mirrorFinalBordersOfImage(image3dBC012, mirrorFinalBordersForThatMuch) :
    image3dBC012WithMirrorPad = image3dBC012
    for time_i in range(0, mirrorFinalBordersForThatMuch[0]) :
        image3dBC012WithMirrorPad = T.concatenate([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,-1:,:,:] ], axis=2)
    for time_i in range(0, mirrorFinalBordersForThatMuch[1]) :
        image3dBC012WithMirrorPad = T.concatenate([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,:,-1:,:] ], axis=3)
    for time_i in range(0, mirrorFinalBordersForThatMuch[2]) :
        image3dBC012WithMirrorPad = T.concatenate([ image3dBC012WithMirrorPad, image3dBC012WithMirrorPad[:,:,:,:,-1:] ], axis=4)
    return image3dBC012WithMirrorPad


def pool3dMirrorPad(image3dBC012, image3dBC012Shape, poolParams) :
    # image3dBC012 dimensions: (batch, fms, r, c, z)
    # poolParams: [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode]
    ws = poolParams[0] # window size
    stride = poolParams[1] # stride
    mode1 = poolParams[3] # max, sum, average_inc_pad, average_exc_pad
    
    image3dBC012WithMirrorPad = mirrorFinalBordersOfImage(image3dBC012, poolParams[2])
    
    T.signal.pool.pool_3d( input=image3dBC012WithMirrorPad,
                            ws=ws,
                            ignore_border=True,
                            st=stride,
                            pad=(0,0,0),
                            mode=mode1)
    
    #calculate the shape of the image after the max pooling.
    #This calculation is for ignore_border=True! Pooling should only be done in full areas in the mirror-padded image.
    imgShapeAfterPoolAndPad = [ image3dBC012Shape[0],
                                image3dBC012Shape[1],
                                int(ceil( (image3dBC012Shape[2] + poolParams[2][0] - ds[0] + 1) / (1.0*stride[0])) ),
                                int(ceil( (image3dBC012Shape[3] + poolParams[2][1] - ds[1] + 1) / (1.0*stride[1])) ),
                                int(ceil( (image3dBC012Shape[4] + poolParams[2][2] - ds[2] + 1) / (1.0*stride[2])) )
                            ]
    return (pooled_out, imgShapeAfterPoolAndPad)

