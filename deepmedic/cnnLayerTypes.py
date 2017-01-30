# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import numpy
import numpy as np

import theano
import theano.tensor as T

from theano.tensor.nnet import conv
import theano.tensor.nnet.conv3d2d #conv3d2d fixed in bleeding edge version of theano.
import random

from sys import maxint as MAX_INT

from deepmedic.maxPoolingModule import myMaxPooling3d

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

def createAndInitializeWeightsTensor(filterShape, initializationTechniqueClassic0orDelvingInto1, rng) :
    # filterShape of dimensions: [#FMs in this layer, #FMs in input, rKernelDim, cKernelDim, zKernelDim]
    if initializationTechniqueClassic0orDelvingInto1 == 0 :
        stdForInitialization = 0.01
    elif initializationTechniqueClassic0orDelvingInto1 == 1 :
        stdForInitialization = np.sqrt( 2.0 / (filterShape[1] * filterShape[2] * filterShape[3] * filterShape[4]) ) #Delving Into rectifiers suggestion.
        
    W = theano.shared(
                      numpy.asarray(rng.normal(loc=0.0, scale=stdForInitialization, size=(filterShape[0],filterShape[1],filterShape[2],filterShape[3],filterShape[4])),
                                    dtype='float32'#theano.config.floatX
                                    ),
                      borrow=True
                      )
    # W shape: [#FMs of this layer, #FMs of Input, rKernFims, cKernDims, zKernDims]
    return W

def convolveWithGivenWeightMatrix(W, filterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) :
    # input weight matrix W has shape: [ Number of filters (outputFMs), number of input channels, rKernelDim, cKernelDim, zKernelDim ] == filterShape
    # filterShape is the shape of W.
    
    # Conv3d2d requires in in shape: [Number_of_output_filters, zKernelDim, Numb_of_input_Channels, rKernelDim, cKernelDim]
    wReshapedForConv = W.dimshuffle(0,4,1,2,3)
    wReshapedForConvShape = (filterShape[0], filterShape[4], filterShape[1], filterShape[2], filterShape[3])
    
    #Reshape image for what conv3d2d needs:
    inputToConvReshapedTrain = inputToConvTrain.dimshuffle(0, 4, 1, 2, 3)
    inputToConvReshapedShapeTrain = (inputToConvShapeTrain[0], inputToConvShapeTrain[4], inputToConvShapeTrain[1], inputToConvShapeTrain[2], inputToConvShapeTrain[3]) # batch_size, time, num_of_input_channels, rows, columns
    outputOfConvTrain = T.nnet.conv3d2d.conv3d(signals = inputToConvReshapedTrain, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = wReshapedForConv, # Number_of_output_filters, Z, Numb_of_input_Channels, r, c
                                  signals_shape = inputToConvReshapedShapeTrain,
                                  filters_shape = wReshapedForConvShape,
                                  border_mode = 'valid')
    #Output is in the shape of the input image (signals_shape).
    
    #Validation
    inputToConvReshapedVal = inputToConvVal.dimshuffle(0, 4, 1, 2, 3)
    inputToConvReshapedShapeVal = (inputToConvShapeVal[0], inputToConvShapeVal[4], inputToConvShapeVal[1], inputToConvShapeVal[2], inputToConvShapeVal[3])
    outputOfConvVal = T.nnet.conv3d2d.conv3d(signals = inputToConvReshapedVal, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = wReshapedForConv, # Number_of_output_filters, Time, Numb_of_input_Channels, Height/rows, Width/columns
                                  signals_shape = inputToConvReshapedShapeVal,
                                  filters_shape = wReshapedForConvShape,
                                  border_mode = 'valid')
    #Testing
    inputToConvReshapedTest = inputToConvTest.dimshuffle(0, 4, 1, 2, 3)
    inputToConvReshapedShapeTest = (inputToConvShapeTest[0], inputToConvShapeTest[4], inputToConvShapeTest[1], inputToConvShapeTest[2], inputToConvShapeTest[3])
    outputOfConvTest = T.nnet.conv3d2d.conv3d(signals = inputToConvReshapedTest, # batch_size, time, num_of_input_channels, rows, columns
                                  filters = wReshapedForConv, # Number_of_output_filters, Time, Numb_of_input_Channels, Height/rows, Width/columns
                                  signals_shape = inputToConvReshapedShapeTest,
                                  filters_shape = wReshapedForConvShape,
                                  border_mode = 'valid')
    
    outputTrain = outputOfConvTrain.dimshuffle(0, 2, 3, 4, 1) #reshape the result, to have the dimensions as the input image: [BatchSize, #FMsInThisLayer, r, c, z]
    outputVal = outputOfConvVal.dimshuffle(0, 2, 3, 4, 1)
    outputTest = outputOfConvTest.dimshuffle(0, 2, 3, 4, 1)
    
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

def checkDimsOfYpredAndYEqual(y, yPred, stringTrainOrVal) :
    if y.ndim != yPred.ndim:
        raise TypeError( "ERROR! y did not have the same shape as y_pred during " + stringTrainOrVal,
                        ('y', y.type, 'y_pred', yPred.type) )
        
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

#################################################################
#                         Layer Types                           #
#################################################################
# Inheritance:
# Block -> ConvLayer -> LowRankConvLayer
#                L-----> ConvLayerWithSoftmax

class Block(object):
    
    def __init__(self) :
        # === Input to the layer ===
        self.inputTrain = None
        self.inputVal = None
        self.inputTest = None
        self.inputShapeTrain = None
        self.inputShapeVal = None
        self.inputShapeTest = None
        
        # === Basic architecture parameters === 
        self._numberOfFeatureMaps = None
        self._maxPoolingParameters = None
        
        #=== All Trainable Parameters of the Block ===
        self._appliedBnInLayer = None # This flag is a combination of rollingAverageForBn>0 AND useBnFlag, with the latter used for the 1st layers of pathways (on image).
        
        # All trainable parameters
        self.params = [] # W, (gbn), b, (aPrelu)
        self._W = None # Careful. LowRank does not set this. Uses ._WperSubconv
        self._gBn = None # ONLY WHEN BN is applied
        self._b = None # shape: a vector with one value per FM of the input
        self._aPrelu = None # ONLY WHEN PreLu
        
        # ONLY WHEN BN! All of these are for the rolling average! If I fix this, only 2 will remain!
        self._muBnsArrayForRollingAverage = None # Array
        self._varBnsArrayForRollingAverage = None # Arrays
        self._rollingAverageForBatchNormalizationOverThatManyBatches = None
        self._indexWhereRollingAverageIs = 0 #Index in the rolling-average matrices of the layers, of the entry to update in the next batch.
        self._sharedNewMu_B = None # last value shared, to update the rolling average array.
        self._sharedNewVar_B = None
        self._newMu_B = None # last value tensor, to update the corresponding shared.
        self._newVar_B = None
        
        
        # === Output of the block ===
        self.outputTrain = None
        self.outputVal = None
        self.outputTest = None
        self.outputShapeTrain = None
        self.outputShapeVal = None
        self.outputShapeTest = None
        
    # Setters
    def _setBlocksInputAttributes(self, inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain, inputToLayerShapeVal, inputToLayerShapeTest) :
        self.inputTrain = inputToLayerTrain
        self.inputVal = inputToLayerVal
        self.inputTest = inputToLayerTest
        self.inputShapeTrain = inputToLayerShapeTrain
        self.inputShapeVal = inputToLayerShapeVal
        self.inputShapeTest = inputToLayerShapeTest
        
    def _setBlocksArchitectureAttributes(self, filterShape, maxPoolingParameters) :
        self._numberOfFeatureMaps = filterShape[0] # Of the output! Used in trainValidationVisualise.py. Not of the input!
        assert self.inputShapeTrain[1] == filterShape[1]
        self._maxPoolingParameters = maxPoolingParameters
        
    def _setBlocksOutputAttributes(self, outputTrain, outputVal, outputTest, outputShapeTrain, outputShapeVal, outputShapeTest) :
        self.outputTrain = outputTrain
        self.outputVal = outputVal
        self.outputTest = outputTest
        self.outputShapeTrain = outputShapeTrain
        self.outputShapeVal = outputShapeVal
        self.outputShapeTest = outputShapeTest
        
    # Getters
    def getNumberOfFeatureMaps(self):
        return self._numberOfFeatureMaps
    def fmsActivations(self, indices_of_fms_in_layer_to_visualise_from_to_exclusive) :
        return self.outputTest[:, indices_of_fms_in_layer_to_visualise_from_to_exclusive[0] : indices_of_fms_in_layer_to_visualise_from_to_exclusive[1], :, :, :]
    
    # Other API
    def getL1RegCost(self) : #Called for L1 weigths regularisation
        raise NotImplementedMethod() # Abstract implementation. Children classes should implement this.
    def getL2RegCost(self) : #Called for L2 weigths regularisation
        raise NotImplementedMethod()
    
    def updateTheMatricesWithTheLastMusAndVarsForTheRollingAverageOfBNInference(self):
        # This function should be erazed when I reimplement the Rolling average.
        if self._appliedBnInLayer :
            muArrayValue = self._muBnsArrayForRollingAverage.get_value()
            muArrayValue[self._indexWhereRollingAverageIs] = self._sharedNewMu_B.get_value()
            self._muBnsArrayForRollingAverage.set_value(muArrayValue, borrow=True)
            
            varArrayValue = self._varBnsArrayForRollingAverage.get_value()
            varArrayValue[self._indexWhereRollingAverageIs] = self._sharedNewVar_B.get_value()
            self._varBnsArrayForRollingAverage.set_value(varArrayValue, borrow=True)
            self._indexWhereRollingAverageIs = (self._indexWhereRollingAverageIs + 1) % self._rollingAverageForBatchNormalizationOverThatManyBatches
            
    def getUpdatesForBnRollingAverage(self) :
        # This function or something similar should stay, even if I clean the BN rolling average.
        if self._appliedBnInLayer :
            #CAREFUL: WARN, PROBLEM, THEANO BUG! If a layer has only 1FM, the .newMu_B ends up being of type (true,) instead of vector!!! Error!!!
            return [(self._sharedNewMu_B, self._newMu_B),
                    (self._sharedNewVar_B, self._newVar_B) ]
        else :
            return []
        
class ConvLayer(Block):
    
    def __init__(self) :
        Block.__init__(self)
        self._activationFunctionType = "" #linear, relu or prelu
        
    def _processInputWithBnNonLinearityDropoutPooling(self,
                rng,
                inputToLayerTrain,
                inputToLayerVal,
                inputToLayerTest,
                inputToLayerShapeTrain,
                inputToLayerShapeVal,
                inputToLayerShapeTest,
                useBnFlag, # Must be true to do BN. Used to not allow doing BN on first layers straight on image, even if rollingAvForBnOverThayManyBatches > 0.
                rollingAverageForBatchNormalizationOverThatManyBatches, #If this is <= 0, we are not using BatchNormalization, even if above is True.
                maxPoolingParameters,
                activationFunctionToUseRelu0orPrelu1orMinus1ForLinear,
                dropoutRate) :
        # ---------------- Order of what is applied -----------------
        #  Input -> [ BatchNorm OR biases applied] -> NonLinearity -> DropOut -> Pooling --> Conv ] # ala He et al "Identity Mappings in Deep Residual Networks" 2016
        # -----------------------------------------------------------
        
        #---------------------------------------------------------
        #------------------ Batch Normalization ------------------
        #---------------------------------------------------------
        if useBnFlag and rollingAverageForBatchNormalizationOverThatManyBatches > 0 :
            self._appliedBnInLayer = True
            self._rollingAverageForBatchNormalizationOverThatManyBatches = rollingAverageForBatchNormalizationOverThatManyBatches
            (inputToNonLinearityTrain,
            inputToNonLinearityVal,
            inputToNonLinearityTest,
            self._gBn,
            self._b,
            # For rolling average :
            self._muBnsArrayForRollingAverage,
            self._varBnsArrayForRollingAverage,
            self._sharedNewMu_B,
            self._sharedNewVar_B,
            self._newMu_B,
            self._newVar_B
            ) = applyBn( rollingAverageForBatchNormalizationOverThatManyBatches, inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain)
            self.params = self.params + [self._gBn, self._b]
        else : #Not using batch normalization
            self._appliedBnInLayer = False
            #make the bias terms and apply them. Like the old days before BN's own learnt bias terms.
            numberOfInputChannels = inputToLayerShapeTrain[1]
            
            (self._b,
            inputToNonLinearityTrain,
            inputToNonLinearityVal,
            inputToNonLinearityTest) = makeBiasParamsAndApplyToFms( inputToLayerTrain, inputToLayerVal, inputToLayerTest, numberOfInputChannels )
            self.params = self.params + [self._b]
            
        #--------------------------------------------------------
        #------------ Apply Activation/ non-linearity -----------
        #--------------------------------------------------------
        if activationFunctionToUseRelu0orPrelu1orMinus1ForLinear == -1 : # -1 stands for "no nonlinearity". Used for input layers of the pathway.
            self._activationFunctionType = "linear"
            ( inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = (inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest)
        elif activationFunctionToUseRelu0orPrelu1orMinus1ForLinear == 0 :
            #print "Layer: Activation function used = ReLu"
            self._activationFunctionType = "relu"
            ( inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = applyRelu(inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest)
        elif activationFunctionToUseRelu0orPrelu1orMinus1ForLinear == 1 :
            #print "Layer: Activation function used = PReLu"
            self._activationFunctionType = "prelu"
            numberOfInputChannels = inputToLayerShapeTrain[1]
            ( self._aPrelu, inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = applyPrelu(inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest, numberOfInputChannels)
            self.params = self.params + [self._aPrelu]
            
        #------------------------------------
        #------------- Dropout --------------
        #------------------------------------
        (inputToPoolTrain, inputToPoolVal, inputToPoolTest) = applyDropout(rng, dropoutRate, inputToLayerShapeTrain, inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest)
        
        #-------------------------------------------------------
        #-----------  Pooling ----------------------------------
        #-------------------------------------------------------
        if maxPoolingParameters == [] : #no max pooling before this conv
            inputToConvTrain = inputToPoolTrain
            inputToConvVal = inputToPoolVal
            inputToConvTest = inputToPoolTest
            
            inputToConvShapeTrain = inputToLayerShapeTrain
            inputToConvShapeVal = inputToLayerShapeVal
            inputToConvShapeTest = inputToLayerShapeTest
        else : #Max pooling is actually happening here...
            (inputToConvTrain, inputToConvShapeTrain) = myMaxPooling3d(inputToPoolTrain, inputToLayerShapeTrain, self._maxPoolingParameters)
            (inputToConvVal, inputToConvShapeVal) = myMaxPooling3d(inputToPoolVal, inputToLayerShapeVal, self._maxPoolingParameters)
            (inputToConvTest, inputToConvShapeTest) = myMaxPooling3d(inputToPoolTest, inputToLayerShapeTest, self._maxPoolingParameters)
            
        return (inputToConvTrain, inputToConvVal, inputToConvTest,
                inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest )
        
    def _createWeightsTensorAndConvolve(self, rng, filterShape, initializationTechniqueClassic0orDelvingInto1, 
                                        inputToConvTrain, inputToConvVal, inputToConvTest,
                                        inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) :
        #-----------------------------------------------
        #------------------ Convolution ----------------
        #-----------------------------------------------
        #----- Initialise the weights -----
        # W shape: [#FMs of this layer, #FMs of Input, rKernDim, cKernDim, zKernDim]
        self._W = createAndInitializeWeightsTensor(filterShape, initializationTechniqueClassic0orDelvingInto1, rng)
        self.params = [self._W] + self.params
        
        #---------- Convolve --------------
        tupleWithOuputAndShapeTrValTest = convolveWithGivenWeightMatrix(self._W, filterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        return tupleWithOuputAndShapeTrValTest
    
    # The main function that builds this.
    def makeLayer(self,
                rng,
                inputToLayerTrain,
                inputToLayerVal,
                inputToLayerTest,
                inputToLayerShapeTrain,
                inputToLayerShapeVal,
                inputToLayerShapeTest,
                filterShape,
                useBnFlag, # Must be true to do BN. Used to not allow doing BN on first layers straight on image, even if rollingAvForBnOverThayManyBatches > 0.
                rollingAverageForBatchNormalizationOverThatManyBatches, #If this is <= 0, we are not using BatchNormalization, even if above is True.
                maxPoolingParameters,
                initializationTechniqueClassic0orDelvingInto1,
                activationFunctionToUseRelu0orPrelu1orMinus1ForLinear=0,
                dropoutRate=0.0):
        """
        type rng: numpy.random.RandomState
        param rng: a random number generator used to initialize weights
        
        type inputToLayer:  tensor5 = theano.tensor.TensorType(dtype='float32', broadcastable=(False, False, False, False, False))
        param inputToLayer: symbolic image tensor, of shape inputToLayerShape
        
        type filterShape: tuple or list of length 5
        param filterShape: (number of filters, num input feature maps,
                            filter height, filter width, filter depth)
                            
        type inputToLayerShape: tuple or list of length 5
        param inputToLayerShape: (batch size, num input feature maps,
                            image height, image width, filter depth)
        """
        self._setBlocksInputAttributes(inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain, inputToLayerShapeVal, inputToLayerShapeTest)
        self._setBlocksArchitectureAttributes(filterShape, maxPoolingParameters)
        
        # Apply all the straightforward operations on the input, such as BN, activation function, dropout, pooling        
        (inputToConvTrain, inputToConvVal, inputToConvTest,
        inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) = self._processInputWithBnNonLinearityDropoutPooling(        rng,
                                                                                        inputToLayerTrain,
                                                                                        inputToLayerVal,
                                                                                        inputToLayerTest,
                                                                                        inputToLayerShapeTrain,
                                                                                        inputToLayerShapeVal,
                                                                                        inputToLayerShapeTest,
                                                                                        useBnFlag,
                                                                                        rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                                        maxPoolingParameters,
                                                                                        activationFunctionToUseRelu0orPrelu1orMinus1ForLinear,
                                                                                        dropoutRate)
        
        tupleWithOuputAndShapeTrValTest = self._createWeightsTensorAndConvolve( rng, filterShape, initializationTechniqueClassic0orDelvingInto1, 
                                                                                inputToConvTrain, inputToConvVal, inputToConvTest,
                                                                                inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        self._setBlocksOutputAttributes(*tupleWithOuputAndShapeTrValTest)
        
    # Override parent's abstract classes.
    def getL1RegCost(self) : #Called for L1 weigths regularisation
        return abs(self._W).sum()
    def getL2RegCost(self) : #Called for L2 weigths regularisation
        return (self._W ** 2).sum()
    
    
# Ala Yani Ioannou et al, Training CNNs with Low-Rank Filters For Efficient Image Classification, ICLR 2016. Allowed Ranks: Rank=1 or 2.
class LowRankConvLayer(ConvLayer):
    def __init__(self, rank=2) :
        ConvLayer.__init__(self)
        
        self._WperSubconv = None # List of ._W theano tensors. One per low-rank subconv. Treat carefully. 
        del(self._W) # The ._W of the Block parent is not used.
        self._rank = rank # 1 or 2 dimensions
        
    def _cropSubconvOutputsToSameDimsAndConcatenateFms( self,
                                                        rSubconvOutput, rSubconvOutputShape,
                                                        cSubconvOutput, cSubconvOutputShape,
                                                        zSubconvOutput, zSubconvOutputShape,
                                                        filterShape) :
        assert (rSubconvOutputShape[0] == cSubconvOutputShape[0]) and (cSubconvOutputShape[0] == zSubconvOutputShape[0]) # same batch size.
        
        concatOutputShape = [ rSubconvOutputShape[0],
                                rSubconvOutputShape[1] + cSubconvOutputShape[1] + zSubconvOutputShape[1],
                                rSubconvOutputShape[2],
                                cSubconvOutputShape[3],
                                zSubconvOutputShape[4]
                                ]
        rCropSlice = slice( (filterShape[2]-1)/2, (filterShape[2]-1)/2 + concatOutputShape[2] )
        cCropSlice = slice( (filterShape[3]-1)/2, (filterShape[3]-1)/2 + concatOutputShape[3] )
        zCropSlice = slice( (filterShape[4]-1)/2, (filterShape[4]-1)/2 + concatOutputShape[4] )
        rSubconvOutputCropped = rSubconvOutput[:,:, :, cCropSlice if self._rank == 1 else slice(0, MAX_INT), zCropSlice  ]
        cSubconvOutputCropped = cSubconvOutput[:,:, rCropSlice, :, zCropSlice if self._rank == 1 else slice(0, MAX_INT) ]
        zSubconvOutputCropped = zSubconvOutput[:,:, rCropSlice if self._rank == 1 else slice(0, MAX_INT), cCropSlice, : ]
        concatSubconvOutputs = T.concatenate([rSubconvOutputCropped, cSubconvOutputCropped, zSubconvOutputCropped], axis=1) #concatenate the FMs
        
        return (concatSubconvOutputs, concatOutputShape)
    
    # Overload the ConvLayer's function. Called from makeLayer. The only different behaviour, because BN, ActivationFunc, DropOut and Pooling are done on a per-FM fashion.        
    def _createWeightsTensorAndConvolve(self, rng, filterShape, initializationTechniqueClassic0orDelvingInto1, 
                                        inputToConvTrain, inputToConvVal, inputToConvTest,
                                        inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) :
        # Behaviour: Create W, set self._W, set self.params, convolve, return ouput and outputShape.
        # The created filters are either 1-dimensional (rank=1) or 2-dim (rank=2), depending  on the self._rank
        # If 1-dim: rSubconv is the input convolved with the row-1dimensional filter.
        # If 2-dim: rSubconv is the input convolved with the RC-2D filter, cSubconv with CZ-2D filter, zSubconv with ZR-2D filter. 
        
        #----- Initialise the weights and Convolve for 3 separate, low rank filters, R,C,Z. -----
        # W shape: [#FMs of this layer, #FMs of Input, rKernDim, cKernDim, zKernDim]
        
        rSubconvFilterShape = [ filterShape[0]/3, filterShape[1], filterShape[2], 1 if self._rank == 1 else filterShape[3], 1 ]
        rSubconvW = createAndInitializeWeightsTensor(rSubconvFilterShape, initializationTechniqueClassic0orDelvingInto1, rng)
        rSubconvTupleWithOuputAndShapeTrValTest = convolveWithGivenWeightMatrix(rSubconvW, rSubconvFilterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        cSubconvFilterShape = [ filterShape[0]/3, filterShape[1], 1, filterShape[3], 1 if self._rank == 1 else filterShape[4] ]
        cSubconvW = createAndInitializeWeightsTensor(cSubconvFilterShape, initializationTechniqueClassic0orDelvingInto1, rng)
        cSubconvTupleWithOuputAndShapeTrValTest = convolveWithGivenWeightMatrix(cSubconvW, cSubconvFilterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        numberOfFmsForTotalToBeExact = filterShape[0] - 2*(filterShape[0]/3) # Cause of possibly inexact integer division.
        zSubconvFilterShape = [ numberOfFmsForTotalToBeExact, filterShape[1], 1 if self._rank == 1 else filterShape[2], 1, filterShape[4] ]
        zSubconvW = createAndInitializeWeightsTensor(zSubconvFilterShape, initializationTechniqueClassic0orDelvingInto1, rng)
        zSubconvTupleWithOuputAndShapeTrValTest = convolveWithGivenWeightMatrix(zSubconvW, zSubconvFilterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        # Set the W attribute and trainable parameters.
        self._WperSubconv = [rSubconvW, cSubconvW, zSubconvW] # Bear in mind that these sub tensors have different shapes! Treat carefully.
        self.params = self._WperSubconv + self.params
        
        # concatenate together.
        (concatSubconvOutputsTrain, concatOutputShapeTrain) = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputAndShapeTrValTest[0], rSubconvTupleWithOuputAndShapeTrValTest[3],
                                                                                                        cSubconvTupleWithOuputAndShapeTrValTest[0], cSubconvTupleWithOuputAndShapeTrValTest[3],
                                                                                                        zSubconvTupleWithOuputAndShapeTrValTest[0], zSubconvTupleWithOuputAndShapeTrValTest[3],
                                                                                                        filterShape)
        (concatSubconvOutputsVal, concatOutputShapeVal) = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputAndShapeTrValTest[1], rSubconvTupleWithOuputAndShapeTrValTest[4],
                                                                                                        cSubconvTupleWithOuputAndShapeTrValTest[1], cSubconvTupleWithOuputAndShapeTrValTest[4],
                                                                                                        zSubconvTupleWithOuputAndShapeTrValTest[1], zSubconvTupleWithOuputAndShapeTrValTest[4],
                                                                                                        filterShape)
        (concatSubconvOutputsTest, concatOutputShapeTest) = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputAndShapeTrValTest[2], rSubconvTupleWithOuputAndShapeTrValTest[5],
                                                                                                        cSubconvTupleWithOuputAndShapeTrValTest[2], cSubconvTupleWithOuputAndShapeTrValTest[5],
                                                                                                        zSubconvTupleWithOuputAndShapeTrValTest[2], zSubconvTupleWithOuputAndShapeTrValTest[5],
                                                                                                        filterShape)
        
        return (concatSubconvOutputsTrain, concatSubconvOutputsVal, concatSubconvOutputsTest, concatOutputShapeTrain, concatOutputShapeVal, concatOutputShapeTest)
        
        
    # Implement parent's abstract classes.
    def getL1RegCost(self) : #Called for L1 weigths regularisation
        l1Cost = 0
        for wOfSubconv in self._WperSubconv : l1Cost += abs(wOfSubconv).sum()
        return l1Cost
    def getL2RegCost(self) : #Called for L2 weigths regularisation
        l2Cost = 0
        for wOfSubconv in self._WperSubconv : l2Cost += (wOfSubconv ** 2).sum()
        return l2Cost
    def getW(self):
        print "ERROR: For LowRankConvLayer, the ._W is not used! Use ._WperSubconv instead and treat carefully!! Exiting!"; exit(1)
        
        
class ConvLayerWithSoftmax(ConvLayer):
    """ Final Classification layer with Softmax """
    
    def __init__(self):
        ConvLayer.__init__(self)
        
        self._numberOfOutputClasses = None
        self._bClassLayer = None
        
        self._softmaxTemperature = None
        
    def makeLayer(  self,
                    rng,
                    inputToLayerTrain,
                    inputToLayerVal,
                    inputToLayerTest,
                    inputToLayerShapeTrain,
                    inputToLayerShapeVal,
                    inputToLayerShapeTest,
                    filterShape,
                    useBnFlag, # Must be true to do BN
                    rollingAverageForBatchNormalizationOverThatManyBatches, #If this is 0, it means we are not using BatchNormalization
                    maxPoolingParameters,
                    initializationTechniqueClassic0orDelvingInto1,
                    activationFunctionToUseRelu0orPrelu1orMinus1ForLinear=0,
                    dropoutRate=0.0,
                    softmaxTemperature = 1):
        
        ConvLayer.makeLayer(self,
                        rng,
                        inputToLayerTrain,
                        inputToLayerVal,
                        inputToLayerTest,
                        inputToLayerShapeTrain,
                        inputToLayerShapeVal,
                        inputToLayerShapeTest,
                        filterShape,
                        useBnFlag, # Must be true to do BN
                        rollingAverageForBatchNormalizationOverThatManyBatches, #If this is 0, it means we are not using BatchNormalization
                        maxPoolingParameters,
                        initializationTechniqueClassic0orDelvingInto1,
                        activationFunctionToUseRelu0orPrelu1orMinus1ForLinear,
                        dropoutRate)
        
        self._numberOfOutputClasses = filterShape[0]
        assert self._numberOfOutputClasses == self._numberOfFeatureMaps
        self._softmaxTemperature = softmaxTemperature
        
        outputOfConvTrain = self.outputTrain
        outputOfConvVal = self.outputVal
        outputOfConvTest = self.outputTest
        
        outputOfConvShapeTrain = self.outputShapeTrain
        outputOfConvShapeVal = self.outputShapeVal
        outputOfConvShapeTest = self.outputShapeTest
        
        # At this last classification layer, the conv output needs to have bias added before the softmax.
        # NOTE: So, two biases are associated with this layer. self.b which is added in the ouput of the previous layer's output of conv,
        # and this self._bClassLayer that is added only to this final output before the softmax.
        (self._bClassLayer,
        inputToSoftmaxTrain,
        inputToSoftmaxVal,
        inputToSoftmaxTest) = makeBiasParamsAndApplyToFms( outputOfConvTrain, outputOfConvVal, outputOfConvTest, self._numberOfFeatureMaps )
        self.params = self.params + [self._bClassLayer]
        
        # ============ Softmax ==============
        ( self.p_y_given_x_train,
        self.y_pred_train ) = applySoftmaxToFmAndReturnProbYandPredY( inputToSoftmaxTrain, outputOfConvShapeTrain, self._numberOfOutputClasses, softmaxTemperature)
        ( self.p_y_given_x_val,
        self.y_pred_val ) = applySoftmaxToFmAndReturnProbYandPredY( inputToSoftmaxVal, outputOfConvShapeVal, self._numberOfOutputClasses, softmaxTemperature)
        ( self.p_y_given_x_test,
        self.y_pred_test ) = applySoftmaxToFmAndReturnProbYandPredY( inputToSoftmaxTest, outputOfConvShapeTest, self._numberOfOutputClasses, softmaxTemperature)
        
        
    def negativeLogLikelihood(self, y, weightPerClass):
        # Used in training.
        # param y: y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        # weightPerClass is a vector with 1 element per class.
        
        #Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
        e1 = np.finfo(np.float32).tiny
        addTinyProbMatrix = T.lt(self.p_y_given_x_train, 4*e1) * e1
        
        weightPerClassBroadcasted = weightPerClass.dimshuffle('x', 0, 'x', 'x', 'x')
        log_p_y_given_x_train = T.log(self.p_y_given_x_train + addTinyProbMatrix) #added a tiny so that it does not go to zero and I have problems with nan again...
        weighted_log_p_y_given_x_train = log_p_y_given_x_train * weightPerClassBroadcasted
        # return -T.mean( weighted_log_p_y_given_x_train[T.arange(y.shape[0]), y] )
        
        # Not a very elegant way to do the indexing but oh well...
        indexDim0 = T.arange( weighted_log_p_y_given_x_train.shape[0] ).dimshuffle( 0, 'x','x','x')
        indexDim2 = T.arange( weighted_log_p_y_given_x_train.shape[2] ).dimshuffle('x', 0, 'x','x')
        indexDim3 = T.arange( weighted_log_p_y_given_x_train.shape[3] ).dimshuffle('x','x', 0, 'x')
        indexDim4 = T.arange( weighted_log_p_y_given_x_train.shape[4] ).dimshuffle('x','x','x', 0)
        return -T.mean( weighted_log_p_y_given_x_train[ indexDim0, y, indexDim2, indexDim3, indexDim4] )
    
    
    def meanErrorTraining(self, y):
        # Returns float = number of errors / number of examples of the minibatch ; [0., 1.]
        # param y: y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        # check if y has same dimension of y_pred
        checkDimsOfYpredAndYEqual(y, self.y_pred_train, "training")
        
        #Mean error of the training batch.
        tneq = T.neq(self.y_pred_train, y)
        meanError = T.mean(tneq)
        return meanError
    
    def meanErrorValidation(self, y):
        # y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        # check if y has same dimension of y_pred
        checkDimsOfYpredAndYEqual(y, self.y_pred_val, "validation")
        
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            tneq = T.neq(self.y_pred_val, y)
            meanError = T.mean(tneq)
            return meanError #The percentage of the predictions that is not the correct class.
        else:
            raise NotImplementedError()
        
    def getRpRnTpTnForTrain0OrVal1(self, y, training0OrValidation1):
        # The returned list has (numberOfClasses)x4 integers: >numberOfRealPositives, numberOfRealNegatives, numberOfTruePredictedPositives, numberOfTruePredictedNegatives< for each class (incl background).
        # Order in the list is the natural order of the classes (ie class-0 RP,RN,TPP,TPN, class-1 RP,RN,TPP,TPN, class-2 RP,RN,TPP,TPN ...)
        # param y: y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        yPredToUse = self.y_pred_train if  training0OrValidation1 == 0 else self.y_pred_val
        checkDimsOfYpredAndYEqual(y, yPredToUse, "training" if training0OrValidation1 == 0 else "validation")
        
        returnedListWithNumberOfRpRnTpTnForEachClass = []
        
        for class_i in xrange(0, self._numberOfOutputClasses) :
            #Number of Real Positive, Real Negatives, True Predicted Positives and True Predicted Negatives are reported PER CLASS (first for WHOLE).
            tensorOneAtRealPos = T.eq(y, class_i)
            tensorOneAtRealNeg = T.neq(y, class_i)

            tensorOneAtPredictedPos = T.eq(yPredToUse, class_i)
            tensorOneAtPredictedNeg = T.neq(yPredToUse, class_i)
            tensorOneAtTruePos = T.and_(tensorOneAtRealPos,tensorOneAtPredictedPos)
            tensorOneAtTrueNeg = T.and_(tensorOneAtRealNeg,tensorOneAtPredictedNeg)
                    
            returnedListWithNumberOfRpRnTpTnForEachClass.append( T.sum(tensorOneAtRealPos) )
            returnedListWithNumberOfRpRnTpTnForEachClass.append( T.sum(tensorOneAtRealNeg) )
            returnedListWithNumberOfRpRnTpTnForEachClass.append( T.sum(tensorOneAtTruePos) )
            returnedListWithNumberOfRpRnTpTnForEachClass.append( T.sum(tensorOneAtTrueNeg) )
            
        return returnedListWithNumberOfRpRnTpTnForEachClass
    
    def predictionProbabilities(self) :
        return self.p_y_given_x_test
    
    