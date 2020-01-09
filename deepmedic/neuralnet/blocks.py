# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
import random

import tensorflow as tf

import deepmedic.neuralnet.layers as dm_layers

try:
    from sys import maxint as MAX_INT
except ImportError:
    # python3 compatibility
    from sys import maxsize as MAX_INT
    
    
#################################################################
#                         Block Types                           #
#################################################################
# Blocks are sequences of layers.
# Blocks dont create new trainable parameters.
# Anything that adds new trainable parameters is a layer, added to the block/sequence.
# Inheritance:
# Block -> ConvBlock -> LowRankConvBlock
#                L-----> ConvBlockWithSoftmax
#        L-----> SoftmaxBlock (could be changed to layer)


class Block(object):
    
    def __init__(self) :
        # === Input to the layer ===
        self.input= {"train": None, "val": None, "test": None}
        
        # === Basic architecture parameters === 
        self._numberOfFeatureMaps = None
        
        #=== All layers that the block applies ===
        self._layers = []
        self._bn_l = None # Keep track to update moving avg. Only when rollingAverageForBn>0 AND useBnFlag, with the latter used for the 1st layers of pathways (on image).
        
        # === Output of the block ===
        self.output = {"train": None, "val": None, "test": None}
        # New and probably temporary, for the residual connections to be "visible".
        self.outputAfterResidualConnIfAnyAtOutp = {"train": None, "val": None, "test": None}
        
        # ==== Target Block Connected to that layer (softmax, regression, auxiliary loss etc), if any ======
        self._target_blocks = []
        
    # Setters
    def _setBlocksInputAttributes(self, inputToLayerTrain, inputToLayerVal, inputToLayerTest) :
        self.input["train"] = inputToLayerTrain
        self.input["val"] = inputToLayerVal
        self.input["test"] = inputToLayerTest
            
    def _setBlocksOutputAttributes(self, outputTrain, outputVal, outputTest) :
        self.output["train"] = outputTrain
        self.output["val"] = outputVal
        self.output["test"] = outputTest
        # New and probably temporary, for the residual connections to be "visible".
        self.outputAfterResidualConnIfAnyAtOutp["train"] = self.output["train"]
        self.outputAfterResidualConnIfAnyAtOutp["val"] = self.output["val"]
        self.outputAfterResidualConnIfAnyAtOutp["test"] = self.output["test"]
        
    def connect_target_block(self, new_target_block_instance):
        # new_target_block_instance : eg softmax layer. Future: Regression layer, or other auxiliary classifiers.
        self._target_blocks += [new_target_block_instance]
    # Getters
    def getNumberOfFeatureMaps(self):
        return self._numberOfFeatureMaps
    def fmsActivations(self, indices_of_fms_in_layer_to_visualise_from_to_exclusive) :
        return self.output["test"][:, indices_of_fms_in_layer_to_visualise_from_to_exclusive[0] : indices_of_fms_in_layer_to_visualise_from_to_exclusive[1], :, :, :]
        
    # Other API
    def trainable_params(self):
        total_params = []
        for layer in self._layers:
            total_params += layer.trainable_params() # concat lists
        for block in self._target_blocks:
            total_params += block.trainable_params() # concat lists
        return total_params
    
    def params_for_L1_L2_reg(self):
        total_params = []
        for layer in self._layers:
            total_params += layer.params_for_L1_L2_reg()
        for block in self._target_blocks:
            total_params += block.params_for_L1_L2_reg()
        return total_params
    
    def update_arrays_of_bn_moving_avg(self, sessionTf):
        # This function should be erazed when I reimplement the Rolling average.
        if self._bn_l is not None :
            self._bn_l.update_arrays_of_bn_moving_avg(sessionTf)
            
    def get_update_ops_for_bn_moving_avg(self) :
        return self._bn_l.get_update_ops_for_bn_moving_avg() if self._bn_l is not None else []



class ConvBlock(Block):
    
    def __init__(self) :
        Block.__init__(self)
        
    def _processInputWithBnNonLinearityDropoutPooling(self,
                rng,
                inputToLayerTrain,
                inputToLayerVal,
                inputToLayerTest,
                useBnFlag, # Must be true to do BN. Used to not allow doing BN on first layers straight on image, even if rollingAvForBnOverThayManyBatches > 0.
                movingAvForBnOverXBatches, #If this is <= 0, we are not using BatchNormalization, even if above is True.
                activationFunc,
                dropoutRate,
                pool_prms) :
        # ---------------- Order of what is applied -----------------
        #  Input -> [ BatchNorm OR biases applied] -> NonLinearity -> DropOut -> Pooling --> Conv ] # ala He et al "Identity Mappings in Deep Residual Networks" 2016
        # -----------------------------------------------------------
        
        #---------------------------------------------------------
        #------------------ Batch Normalization ------------------
        #---------------------------------------------------------
        if useBnFlag and movingAvForBnOverXBatches > 0 :
            self._bn_l = dm_layers.BatchNormLayer(movingAvForBnOverXBatches, n_channels=inputToLayerTrain.shape[1])            
            self._layers.append(self._bn_l)
            inputToNonLinearityTrain = self._bn_l.apply(inputToLayerTrain, mode="train")
            inputToNonLinearityVal = self._bn_l.apply(inputToLayerVal, mode="infer")
            inputToNonLinearityTest = self._bn_l.apply(inputToLayerTest, mode="infer")
            
        else : #Not using batch normalization
            #make the bias terms and apply them. Like the old days before BN's own learnt bias terms.
            bias_l = dm_layers.BiasLayer(inputToLayerTrain.shape[1])
            self._layers.append(bias_l)
            inputToNonLinearityTrain = bias_l.apply(inputToLayerTrain)
            inputToNonLinearityVal = bias_l.apply(inputToLayerVal)
            inputToNonLinearityTest = bias_l.apply(inputToLayerTest)
            
        #--------------------------------------------------------
        #------------ Apply Activation/ non-linearity -----------
        #--------------------------------------------------------
        if activationFunc == "linear" : # -1 stands for "no nonlinearity". Used for input layers of the pathway.
            act_l = dm_layers.IdentityLayer()
        elif activationFunc == "relu" :
            act_l = dm_layers.ReluLayer()
        elif activationFunc == "prelu" :
            act_l = dm_layers.PreluLayer(inputToNonLinearityTrain.shape[1])
            
        elif activationFunc == "elu" :
            act_l = dm_layers.EluLayer()
        elif activationFunc == "selu" :
            act_l = dm_layers.SeluLayer()
        self._layers.append(act_l)
        inputToDropoutTrain = act_l.apply(inputToNonLinearityTrain)
        inputToDropoutVal = act_l.apply(inputToNonLinearityVal)
        inputToDropoutTest = act_l.apply(inputToNonLinearityTest)
        
        #------------------------------------
        #------------- Dropout --------------
        #------------------------------------
        dropout_l = dm_layers.DropoutLayer(dropoutRate, rng)
        self._layers.append(dropout_l)
        inputToPoolTrain = dropout_l.apply(inputToDropoutTrain, mode="train")
        inputToPoolVal = dropout_l.apply(inputToDropoutVal, mode="infer")
        inputToPoolTest = dropout_l.apply(inputToDropoutTest, mode="infer")
        
        #-------------------------------------------------------
        #-----------  Pooling ----------------------------------
        #-------------------------------------------------------
        if pool_prms == [] : #no max pooling before this conv
            inputToConvTrain = inputToPoolTrain
            inputToConvVal = inputToPoolVal
            inputToConvTest = inputToPoolTest
        else : #Max pooling is actually happening here...
            pooling_l = PoolingLayer(pool_prms[0], pool_prms[1], pool_prms[2], pool_prms[3])
            self._layers.append(pooling_l)
            inputToConvTrain = pooling_l.apply(inputToPoolTrain)
            inputToConvVal = pooling_l.apply(inputToPoolVal)
            inputToConvTest = pooling_l.apply(inputToPoolTest)
        
        return (inputToConvTrain, inputToConvVal, inputToConvTest)
        
    def _createConvLayer(self, filter_shape, init_method, rng):
        return dm_layers.ConvolutionalLayer(filter_shape, init_method, rng)
    
    # The main function that builds this.
    def makeLayer(self,
                rng,
                inputToLayerTrain,
                inputToLayerVal,
                inputToLayerTest,
                filterShape,
                poolingParameters, # Can be []
                convWInitMethod,
                useBnFlag, # Must be true to do BN. Used to not allow doing BN on first layers straight on image, even if rollingAvForBnOverThayManyBatches > 0.
                movingAvForBnOverXBatches, #If this is <= 0, we are not using BatchNormalization, even if above is True.
                activationFunc="relu",
                dropoutRate=0.0):
        """
        type rng: numpy.random.RandomState
        param rng: a random number generator used to initialize weights
        
        type inputToLayer:  tensor5
        param inputToLayer: symbolic image tensor, of shape inputToLayerShape
        
        type filterShape: tuple or list of length 5
        param filterShape: (number of filters, num input feature maps,
                            filter height, filter width, filter depth)
                            
        type inputToLayerShape: tuple or list of length 5
        param inputToLayerShape: (batch size, num input feature maps,
                            image height, image width, filter depth)
        """
        self._setBlocksInputAttributes(inputToLayerTrain, inputToLayerVal, inputToLayerTest)
        self._numberOfFeatureMaps = filterShape[0]
        
        # Apply all the straightforward operations on the input, such as BN, activation function, dropout, pooling        
        (inputToConvTrain, inputToConvVal, inputToConvTest) = self._processInputWithBnNonLinearityDropoutPooling( rng,
                                                                                        inputToLayerTrain,
                                                                                        inputToLayerVal,
                                                                                        inputToLayerTest,
                                                                                        useBnFlag,
                                                                                        movingAvForBnOverXBatches,
                                                                                        activationFunc,
                                                                                        dropoutRate,
                                                                                        poolingParameters)
        
        conv_l = self._createConvLayer(filterShape, convWInitMethod, rng)
        self._layers.append(conv_l)
        out_train = conv_l.apply(inputToConvTrain)
        out_val = conv_l.apply(inputToConvVal)
        out_test = conv_l.apply(inputToConvTest)
        
        self._setBlocksOutputAttributes(out_train, out_val, out_test)
        
        return (out_train, out_val, out_test)
    
    
# Ala Yani Ioannou et al, Training CNNs with Low-Rank Filters For Efficient Image Classification, ICLR 2016. Allowed Ranks: Rank=1 or 2.
class LowRankConvBlock(ConvBlock):
    def __init__(self, rank=2) :
        ConvBlock.__init__(self)
        self._rank = rank # 1 or 2 dimensions
            
    # Overload the ConvBlock's function. Called from makeLayer. The only different behaviour.        
    def _createConvLayer(self, filter_shape, init_method, rng):
        return dm_layers.LowRankConvolutionalLayer(filter_shape, init_method, rng)
    
    
class SoftmaxBlock(Block):
    """ Softmax for classification. Note, this is simply the softmax function, after adding bias. Not a ConvBlock """
    def __init__(self):
        Block.__init__(self)
        self._numberOfOutputClasses = None
        self._temperature = None
        
    def makeLayer(self,
                  rng,
                  layerConnected, # the basic layer, at the output of which to connect this softmax.
                  t = 1):
        # t: temperature. Scalar
        
        self._numberOfOutputClasses = layerConnected.getNumberOfFeatureMaps()
        self._temperature = t
        
        self._setBlocksInputAttributes(layerConnected.output["train"], layerConnected.output["val"], layerConnected.output["test"])
        
        # At this last classification layer, the conv output needs to have bias added before the softmax.
        # NOTE: So, two biases are associated with this layer. self.b which is added in the ouput of the previous layer's output of conv,
        # and this self._bClassLayer that is added only to this final output before the softmax.
        
        bias_l = dm_layers.BiasLayer(self.input["train"].shape[1])
        self._layers.append(bias_l)
        logits_train = bias_l.apply(self.input["train"])
        logits_val = bias_l.apply(self.input["val"])
        logits_test = bias_l.apply(self.input["test"])
        
        # ============ Softmax ==============
        self.p_y_given_x_train = tf.nn.softmax(logits_train/t, axis=1)
        self.y_pred_train = tf.argmax(self.p_y_given_x_train, axis=1)
        self.p_y_given_x_val = tf.nn.softmax(logits_val/t, axis=1)
        self.y_pred_val = tf.argmax(self.p_y_given_x_val, axis=1)
        self.p_y_given_x_test = tf.nn.softmax(logits_test/t, axis=1)
        self.y_pred_test = tf.argmax(self.p_y_given_x_test, axis=1)
    
        self._setBlocksOutputAttributes(self.p_y_given_x_train, self.p_y_given_x_val, self.p_y_given_x_test)
        
        layerConnected.connect_target_block(self)
        
        
    def meanErrorTraining(self, y):
        # Returns float = number of errors / number of examples of the minibatch ; [0., 1.]
        # param y: y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        #Mean error of the training batch.
        tneq = tf.logical_not( tf.equal(self.y_pred_train, y) )
        meanError = tf.reduce_mean(tneq)
        return meanError
    
    def meanErrorValidation(self, y):
        # y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            tneq = tf.logical_not( tf.equal(self.y_pred_val, y) )
            meanError = tf.reduce_mean(tneq)
            return meanError #The percentage of the predictions that is not the correct class.
        else:
            raise NotImplementedError("Not implemented behaviour for y.dtype different than int.")
        
    def getRpRnTpTnForTrain0OrVal1(self, y, training0OrValidation1):
        # The returned list has (numberOfClasses)x4 integers: >numberOfRealPositives, numberOfRealNegatives, numberOfTruePredictedPositives, numberOfTruePredictedNegatives< for each class (incl background).
        # Order in the list is the natural order of the classes (ie class-0 RP,RN,TPP,TPN, class-1 RP,RN,TPP,TPN, class-2 RP,RN,TPP,TPN ...)
        # param y: y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        yPredToUse = self.y_pred_train if  training0OrValidation1 == 0 else self.y_pred_val
        
        returnedListWithNumberOfRpRnTpTnForEachClass = []
        
        for class_i in range(0, self._numberOfOutputClasses) :
            #Number of Real Positive, Real Negatives, True Predicted Positives and True Predicted Negatives are reported PER CLASS (first for WHOLE).
            tensorOneAtRealPos = tf.equal(y, class_i)
            tensorOneAtRealNeg = tf.logical_not(tensorOneAtRealPos)

            tensorOneAtPredictedPos = tf.equal(yPredToUse, class_i)
            tensorOneAtPredictedNeg = tf.logical_not(tensorOneAtPredictedPos)
            tensorOneAtTruePos = tf.logical_and(tensorOneAtRealPos,tensorOneAtPredictedPos)
            tensorOneAtTrueNeg = tf.logical_and(tensorOneAtRealNeg,tensorOneAtPredictedNeg)
                    
            returnedListWithNumberOfRpRnTpTnForEachClass.append(tf.reduce_sum(tf.cast(tensorOneAtRealPos, dtype="int32")))
            returnedListWithNumberOfRpRnTpTnForEachClass.append(tf.reduce_sum(tf.cast(tensorOneAtRealNeg, dtype="int32")))
            returnedListWithNumberOfRpRnTpTnForEachClass.append(tf.reduce_sum(tf.cast(tensorOneAtTruePos, dtype="int32")))
            returnedListWithNumberOfRpRnTpTnForEachClass.append(tf.reduce_sum(tf.cast(tensorOneAtTrueNeg, dtype="int32")))
            
        return returnedListWithNumberOfRpRnTpTnForEachClass
    
    def predictionProbabilities(self) :
        return self.p_y_given_x_test


