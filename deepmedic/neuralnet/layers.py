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

import deepmedic.neuralnet.ops as ops
from deepmedic.neuralnet.ops import applyDropout, applyRelu, applyElu, applySelu, pool3dMirrorPad
from deepmedic.neuralnet.ops import createAndInitializeWeightsTensor, convolveWithGivenWeightMatrix

try:
    from sys import maxint as MAX_INT
except ImportError:
    # python3 compatibility
    from sys import maxsize as MAX_INT

        

#################################################################
#                         Layer Types                           #
#################################################################
# Inheritance:
# Block -> ConvLayer -> LowRankConvLayer
#                L-----> ConvLayerWithSoftmax

class BiasLayer(object):
    def __init__(self, n_channels):
        self._b = tf.reshape(tf.Variable(np.zeros((n_channels), dtype = 'float32'), name="b"), shape=[1,n_channels,1,1,1])
        
    def apply(self, input):
        return input + self._b
    
    def trainable_params(self):
        return [self._b]
    
class PreluLayer(object):
    def __init__(self, n_channels, alpha=0.01):
        self._a = tf.reshape(tf.Variable(np.ones((n_channels), dtype='float32')*alpha, name="aPrelu"), shape=[1, n_channels, 1, 1, 1] )

    def apply(self, input):
        # input is a tensor of shape (batchSize, FMs, r, c, z)
        return ops.prelu(input, self._a)
    
    def trainable_params(self):
        return [self._a]
    
class BatchNormLayer(object):
    # Order of functions:
    # __init__ -> apply(train) --> get_update_ops_for_bn_moving_avg -> update_arrays_of_bn_moving_avg -> apply(infer)
    def __init__(self, moving_avg_length, n_channels):
        self._moving_avg_length = moving_avg_length # integer. The number of iterations (batches) over which to compute a moving average.
        self._g = tf.Variable( np.ones( (n_channels), dtype='float32'), name="gBn" )
        self._b = tf.Variable( np.zeros( (n_channels), dtype='float32'), name="bBn" )
        #for moving average:
        self._array_mus_for_moving_avg = tf.Variable( np.zeros( (moving_avg_length, n_channels), dtype='float32' ), name="muBnsForRollingAverage" )
        self._array_vars_for_moving_avg = tf.Variable( np.ones( (moving_avg_length, n_channels), dtype='float32' ), name="varBnsForRollingAverage" )        
        self._new_mu_batch = tf.Variable(np.zeros( (n_channels), dtype='float32'), name="sharedNewMu_B") # Value will be assigned every training-iteration.
        self._new_var_batch = tf.Variable(np.ones( (n_channels), dtype='float32'), name="sharedNewVar_B")
        # Create ops for updating the matrices with the bn inference stats.
        self._tf_plchld_int32 = tf.compat.v1.placeholder( dtype="int32", name="tf_plchld_int32") # convenience for tf.assign
        self._op_update_mtrx_bn_inf_mu = tf.compat.v1.assign( self._array_mus_for_moving_avg[self._tf_plchld_int32], self._new_mu_batch ) # Cant it just take tensor? self._latest_mu_batch?
        self._op_update_mtrx_bn_inf_var = tf.compat.v1.assign( self._array_vars_for_moving_avg[self._tf_plchld_int32], self._new_var_batch )
        
        self._latest_mu_batch = None # I think this is useless
        self._latest_var_batch = None # I think this is useless
        
        self._idx_where_moving_avg_is = 0 #Index in the rolling-average matrices of the layers, of the entry to update in the next batch.

    def trainable_params(self):
        return [self._g, self._b]
    
    def apply(self, input, mode, e1 = np.finfo(np.float32).tiny):
        # mode: String in ["train", "infer"]
        n_channs = input.shape[1]
        
        if mode == "train":
            self._new_mu_batch_t, self._new_var_batch_t = tf.nn.moments(input, axes=[0,2,3,4])
            mu = self._new_mu_batch_t
            var = self._new_var_batch_t
        elif mode == "infer":
            mu = tf.reduce_mean(self._array_mus_for_moving_avg, axis=0)
            var = tf.reduce_mean(self._array_vars_for_moving_avg, axis=0)
        else:
            raise NotImplementedError()
        
        # Reshape for broadcast.
        g_resh = tf.reshape(self._g, shape=[1,n_channs,1,1,1])
        b_resh = tf.reshape(self._b, shape=[1,n_channs,1,1,1])
        mu     = tf.reshape(mu, shape=[1,n_channs,1,1,1])
        var    = tf.reshape(var, shape=[1,n_channs,1,1,1])
        # Normalize
        norm_inp = (input - mu ) /  tf.sqrt(var + e1) # e1 should come OUT of the sqrt! 
        norm_inp = g_resh * norm_inp + b_resh
        
        # Returns mu_batch, var_batch to update the moving average afterwards (during training)
        return norm_inp
        
    def get_update_ops_for_bn_moving_avg(self) : # I think this is utterly useless.
        # This function or something similar should stay, even if I clean the BN rolling average.
        return [ tf.compat.v1.assign( ref=self._new_mu_batch, value=self._new_mu_batch_t, validate_shape=True ),
                tf.compat.v1.assign( ref=self._new_var_batch, value=self._new_var_batch_t, validate_shape=True ) ]
        
    def update_arrays_of_bn_moving_avg(self, sessionTf):
            sessionTf.run( fetches=self._op_update_mtrx_bn_inf_mu, feed_dict={self._tf_plchld_int32: self._idx_where_moving_avg_is} )
            sessionTf.run( fetches=self._op_update_mtrx_bn_inf_var, feed_dict={self._tf_plchld_int32: self._idx_where_moving_avg_is} )
            self._idx_where_moving_avg_is = (self._idx_where_moving_avg_is + 1) % self._moving_avg_length
            
class Block(object):
    
    def __init__(self) :
        # === Input to the layer ===
        self.input= {"train": None, "val": None, "test": None}
        
        # === Basic architecture parameters === 
        self._numberOfFeatureMaps = None
        self._poolingParameters = None
        
        #=== All Trainable Parameters of the Block ===
        self._bn_layer = None # Keep track to update moving avg. Only when rollingAverageForBn>0 AND useBnFlag, with the latter used for the 1st layers of pathways (on image).
        
        # All trainable parameters
        # NOTE: VIOLATED _HIDDEN ENCAPSULATION BY THE FUNCTION THAT TRANSFERS PRETRAINED WEIGHTS deepmed.neuralnet.transferParameters.transferParametersBetweenLayers.
        # TEMPORARY TILL THE API GETS FIXED (AFTER DA)!
        self._params = [] # W, (gbn), b, (aPrelu)
        self._W = None # Careful. LowRank does not set this. Uses ._WperSubconv
        self._aPrelu = None # ONLY WHEN PreLu
        
        # === Output of the block ===
        self.output = {"train": None, "val": None, "test": None}
        # New and probably temporary, for the residual connections to be "visible".
        self.outputAfterResidualConnIfAnyAtOutp = {"train": None, "val": None, "test": None}
        
        # ==== Target Block Connected to that layer (softmax, regression, auxiliary loss etc), if any ======
        self.targetBlock = None
        
    # Setters
    def _setBlocksInputAttributes(self, inputToLayerTrain, inputToLayerVal, inputToLayerTest) :
        self.input["train"] = inputToLayerTrain
        self.input["val"] = inputToLayerVal
        self.input["test"] = inputToLayerTest
        
    def _setBlocksArchitectureAttributes(self, filterShape, poolingParameters) :
        self._numberOfFeatureMaps = filterShape[0] # Of the output! Used in trainValidationVisualise.py. Not of the input!
        assert self.input["train"].shape[1] == filterShape[1]
        self._poolingParameters = poolingParameters
        
    def _setBlocksOutputAttributes(self, outputTrain, outputVal, outputTest) :
        self.output["train"] = outputTrain
        self.output["val"] = outputVal
        self.output["test"] = outputTest
        # New and probably temporary, for the residual connections to be "visible".
        self.outputAfterResidualConnIfAnyAtOutp["train"] = self.output["train"]
        self.outputAfterResidualConnIfAnyAtOutp["val"] = self.output["val"]
        self.outputAfterResidualConnIfAnyAtOutp["test"] = self.output["test"]
        
    def setTargetBlock(self, targetBlockInstance):
        # targetBlockInstance : eg softmax layer. Future: Regression layer, or other auxiliary classifiers.
        self.targetBlock = targetBlockInstance
    # Getters
    def getNumberOfFeatureMaps(self):
        return self._numberOfFeatureMaps
    def fmsActivations(self, indices_of_fms_in_layer_to_visualise_from_to_exclusive) :
        return self.output["test"][:, indices_of_fms_in_layer_to_visualise_from_to_exclusive[0] : indices_of_fms_in_layer_to_visualise_from_to_exclusive[1], :, :, :]
    
    # Other API
    def _get_L1_cost(self) : #Called for L1 weigths regularisation
        raise NotImplementedMethod() # Abstract implementation. Children classes should implement this.
    def _get_L2_cost(self) : #Called for L2 weigths regularisation
        raise NotImplementedMethod()
    def trainable_params(self):
        if self.targetBlock is None :
            return self._params
        else :
            return self._params + self.targetBlock.trainable_params()
        
    def update_arrays_of_bn_moving_avg(self, sessionTf):
        # This function should be erazed when I reimplement the Rolling average.
        if self._bn_layer is not None :
            self._bn_layer.update_arrays_of_bn_moving_avg(sessionTf)
            
    def get_update_ops_for_bn_moving_avg(self) :
        return self._bn_layer.get_update_ops_for_bn_moving_avg() if self._bn_layer is not None else []
        
class ConvLayer(Block):
    
    def __init__(self) :
        Block.__init__(self)
        self._activationFunctionType = "" #linear, relu or prelu
        
    def _processInputWithBnNonLinearityDropoutPooling(self,
                rng,
                inputToLayerTrain,
                inputToLayerVal,
                inputToLayerTest,
                useBnFlag, # Must be true to do BN. Used to not allow doing BN on first layers straight on image, even if rollingAvForBnOverThayManyBatches > 0.
                movingAvForBnOverXBatches, #If this is <= 0, we are not using BatchNormalization, even if above is True.
                activationFunc,
                dropoutRate) :
        # ---------------- Order of what is applied -----------------
        #  Input -> [ BatchNorm OR biases applied] -> NonLinearity -> DropOut -> Pooling --> Conv ] # ala He et al "Identity Mappings in Deep Residual Networks" 2016
        # -----------------------------------------------------------
        
        #---------------------------------------------------------
        #------------------ Batch Normalization ------------------
        #---------------------------------------------------------
        if useBnFlag and movingAvForBnOverXBatches > 0 :
            self._bn_layer = BatchNormLayer(movingAvForBnOverXBatches, n_channels=inputToLayerTrain.shape[1])            
            self._params = self._params + self._bn_layer.trainable_params()
            
            inputToNonLinearityTrain = self._bn_layer.apply(inputToLayerTrain, mode="train")
            inputToNonLinearityVal = self._bn_layer.apply(inputToLayerVal, mode="infer")
            inputToNonLinearityTest = self._bn_layer.apply(inputToLayerTest, mode="infer")
            
        else : #Not using batch normalization
            #make the bias terms and apply them. Like the old days before BN's own learnt bias terms.
            bias_layer = BiasLayer(inputToLayerTrain.shape[1])
            inputToNonLinearityTrain = bias_layer.apply(inputToLayerTrain)
            inputToNonLinearityVal = bias_layer.apply(inputToLayerVal)
            inputToNonLinearityTest = bias_layer.apply(inputToLayerTest)
            self._params = self._params + bias_layer.trainable_params()
            
            
        #--------------------------------------------------------
        #------------ Apply Activation/ non-linearity -----------
        #--------------------------------------------------------
        self._activationFunctionType = activationFunc
        if self._activationFunctionType == "linear" : # -1 stands for "no nonlinearity". Used for input layers of the pathway.
            inputToDropoutTrain = inputToNonLinearityTrain
            inputToDropoutVal = inputToNonLinearityVal
            inputToDropoutTest = inputToNonLinearityTest
        elif self._activationFunctionType == "relu" :
            inputToDropoutTrain = applyRelu(inputToNonLinearityTrain)
            inputToDropoutVal = inapplyRelu(putToNonLinearityVal)
            inputToDropoutTest = applyRelu(inputToNonLinearityTest)
        elif self._activationFunctionType == "prelu" :
            prelu_layer = PreluLayer(inputToNonLinearityTrain.shape[1])
            inputToDropoutTrain = prelu_layer.apply(inputToNonLinearityTrain)
            inputToDropoutVal = prelu_layer.apply(inputToNonLinearityVal)
            inputToDropoutTest = prelu_layer.apply(inputToNonLinearityTest)
            self._params = self._params + prelu_layer.trainable_params()
        elif self._activationFunctionType == "elu" :
            ( inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = applyElu(inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest)
        elif self._activationFunctionType == "selu" :
            ( inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = applySelu(inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest)
            
        #------------------------------------
        #------------- Dropout --------------
        #------------------------------------
        (inputToPoolTrain, inputToPoolVal, inputToPoolTest) = applyDropout(rng, dropoutRate, inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest)
        
        #-------------------------------------------------------
        #-----------  Pooling ----------------------------------
        #-------------------------------------------------------
        if self._poolingParameters == [] : #no max pooling before this conv
            inputToConvTrain = inputToPoolTrain
            inputToConvVal = inputToPoolVal
            inputToConvTest = inputToPoolTest
        else : #Max pooling is actually happening here...
            inputToConvTrain = pool3dMirrorPad(inputToPoolTrain, self._poolingParameters)
            inputToConvVal = pool3dMirrorPad(inputToPoolVal, self._poolingParameters)
            inputToConvTest = pool3dMirrorPad(inputToPoolTest, self._poolingParameters)
            
        return (inputToConvTrain, inputToConvVal, inputToConvTest)
        
    def _createWeightsTensorAndConvolve(self, rng, filterShape, convWInitMethod, 
                                        inputToConvTrain, inputToConvVal, inputToConvTest):
        #-----------------------------------------------
        #------------------ Convolution ----------------
        #-----------------------------------------------
        #----- Initialise the weights -----
        # W shape: [#FMs of this layer, #FMs of Input, rKernDim, cKernDim, zKernDim]
        self._W = createAndInitializeWeightsTensor(filterShape, convWInitMethod, rng)
        self._params = [self._W] + self._params
        
        #---------- Convolve --------------
        (out_train, out_val, out_test) = convolveWithGivenWeightMatrix(self._W, inputToConvTrain, inputToConvVal, inputToConvTest)
        
        return (out_train, out_val, out_test)
    
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
        self._setBlocksArchitectureAttributes(filterShape, poolingParameters)
        
        # Apply all the straightforward operations on the input, such as BN, activation function, dropout, pooling        
        (inputToConvTrain, inputToConvVal, inputToConvTest) = self._processInputWithBnNonLinearityDropoutPooling( rng,
                                                                                        inputToLayerTrain,
                                                                                        inputToLayerVal,
                                                                                        inputToLayerTest,
                                                                                        useBnFlag,
                                                                                        movingAvForBnOverXBatches,
                                                                                        activationFunc,
                                                                                        dropoutRate)
        
        tupleWithOuputsTrValTest = self._createWeightsTensorAndConvolve(rng, filterShape, convWInitMethod, 
                                                                        inputToConvTrain, inputToConvVal, inputToConvTest)
        
        self._setBlocksOutputAttributes(*tupleWithOuputsTrValTest)
        
    # Override parent's abstract classes.
    def _get_L1_cost(self) : #Called for L1 weigths regularisation
        return tf.reduce_sum(tf.abs(self._W))
    def _get_L2_cost(self) : #Called for L2 weigths regularisation
        return tf.reduce_sum(self._W ** 2)
    
    
# Ala Yani Ioannou et al, Training CNNs with Low-Rank Filters For Efficient Image Classification, ICLR 2016. Allowed Ranks: Rank=1 or 2.
class LowRankConvLayer(ConvLayer):
    def __init__(self, rank=2) :
        ConvLayer.__init__(self)
        
        self._WperSubconv = None # List of ._W tensors. One per low-rank subconv. Treat carefully. 
        del(self._W) # The ._W of the Block parent is not used.
        self._rank = rank # 1 or 2 dimensions
        
    def _cropSubconvOutputsToSameDimsAndConcatenateFms( self,
                                                        rSubconvOutput,
                                                        cSubconvOutput,
                                                        zSubconvOutput,
                                                        filterShape) :
        assert (rSubconvOutput.shape[0] == cSubconvOutput.shape[0]) and (cSubconvOutput.shape[0] == zSubconvOutput.shape[0]) # same batch size.
        
        concatOutputShape = [rSubconvOutput.shape[0],
                             rSubconvOutput.shape[1] + cSubconvOutput.shape[1] + zSubconvOutput.shape[1],
                             rSubconvOutput.shape[2],
                             cSubconvOutput.shape[3],
                             zSubconvOutput.shape[4]
                            ]
        rCropSlice = slice( (filterShape[2]-1)//2, (filterShape[2]-1)//2 + concatOutputShape[2] )
        cCropSlice = slice( (filterShape[3]-1)//2, (filterShape[3]-1)//2 + concatOutputShape[3] )
        zCropSlice = slice( (filterShape[4]-1)//2, (filterShape[4]-1)//2 + concatOutputShape[4] )
        rSubconvOutputCropped = rSubconvOutput[:,:, :, cCropSlice if self._rank == 1 else slice(0, MAX_INT), zCropSlice  ]
        cSubconvOutputCropped = cSubconvOutput[:,:, rCropSlice, :, zCropSlice if self._rank == 1 else slice(0, MAX_INT) ]
        zSubconvOutputCropped = zSubconvOutput[:,:, rCropSlice if self._rank == 1 else slice(0, MAX_INT), cCropSlice, : ]
        concatSubconvOutputs = tf.concat([rSubconvOutputCropped, cSubconvOutputCropped, zSubconvOutputCropped], axis=1) #concatenate the FMs
        
        return concatSubconvOutputs
    
    # Overload the ConvLayer's function. Called from makeLayer. The only different behaviour, because BN, ActivationFunc, DropOut and Pooling are done on a per-FM fashion.        
    def _createWeightsTensorAndConvolve(self, rng, filterShape, convWInitMethod, 
                                        inputToConvTrain, inputToConvVal, inputToConvTest) :
        # Behaviour: Create W, set self._W, set self._params, convolve, return ouput and outputShape.
        # The created filters are either 1-dimensional (rank=1) or 2-dim (rank=2), depending  on the self._rank
        # If 1-dim: rSubconv is the input convolved with the row-1dimensional filter.
        # If 2-dim: rSubconv is the input convolved with the RC-2D filter, cSubconv with CZ-2D filter, zSubconv with ZR-2D filter. 
        
        #----- Initialise the weights and Convolve for 3 separate, low rank filters, R,C,Z. -----
        # W shape: [#FMs of this layer, #FMs of Input, rKernDim, cKernDim, zKernDim]
        
        rSubconvFilterShape = [ filterShape[0]//3, filterShape[1], filterShape[2], 1 if self._rank == 1 else filterShape[3], 1 ]
        rSubconvW = createAndInitializeWeightsTensor(rSubconvFilterShape, convWInitMethod, rng)
        rSubconvTupleWithOuputs = convolveWithGivenWeightMatrix(rSubconvW, inputToConvTrain, inputToConvVal, inputToConvTest)
        
        cSubconvFilterShape = [ filterShape[0]//3, filterShape[1], 1, filterShape[3], 1 if self._rank == 1 else filterShape[4] ]
        cSubconvW = createAndInitializeWeightsTensor(cSubconvFilterShape, convWInitMethod, rng)
        cSubconvTupleWithOuputs = convolveWithGivenWeightMatrix(cSubconvW, inputToConvTrain, inputToConvVal, inputToConvTest)
        
        numberOfFmsForTotalToBeExact = filterShape[0] - 2*(filterShape[0]//3) # Cause of possibly inexact integer division.
        zSubconvFilterShape = [ numberOfFmsForTotalToBeExact, filterShape[1], 1 if self._rank == 1 else filterShape[2], 1, filterShape[4] ]
        zSubconvW = createAndInitializeWeightsTensor(zSubconvFilterShape, convWInitMethod, rng)
        zSubconvTupleWithOuputs = convolveWithGivenWeightMatrix(zSubconvW, inputToConvTrain, inputToConvVal, inputToConvTest)
        
        # Set the W attribute and trainable parameters.
        self._WperSubconv = [rSubconvW, cSubconvW, zSubconvW] # Bear in mind that these sub tensors have different shapes! Treat carefully.
        self._params = self._WperSubconv + self._params
        
        # concatenate together.
        concatSubconvOutputsTrain = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputs[0],
                                                                                        cSubconvTupleWithOuputs[0],
                                                                                        zSubconvTupleWithOuputs[0],
                                                                                        filterShape)
        concatSubconvOutputsVal = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputs[1],
                                                                                      cSubconvTupleWithOuputs[1],
                                                                                      zSubconvTupleWithOuputs[1],
                                                                                      filterShape)
        concatSubconvOutputsTest = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputs[2],
                                                                                       cSubconvTupleWithOuputs[2],
                                                                                       zSubconvTupleWithOuputs[2],
                                                                                       filterShape)
        
        return (concatSubconvOutputsTrain, concatSubconvOutputsVal, concatSubconvOutputsTest)
        
        
    # Implement parent's abstract classes.
    def _get_L1_cost(self) : #Called for L1 weigths regularisation
        l1Cost = 0
        for wOfSubconv in self._WperSubconv : l1Cost += tf.reduce_sum(tf.abs(wOfSubconv))
        return l1Cost
    def _get_L2_cost(self) : #Called for L2 weigths regularisation
        l2Cost = 0
        for wOfSubconv in self._WperSubconv : l2Cost += tf.reduce_sum(wOfSubconv ** 2)
        return l2Cost
    def getW(self):
        print("ERROR: For LowRankConvLayer, the ._W is not used! Use ._WperSubconv instead and treat carefully!! Exiting!"); exit(1)
        
        
class TargetLayer(Block):
    # Mother class of all layers the output of which will be "trained".
    # i.e. requires a y_gt feed, which is specified for each child in get_output_gt_tensor_feed()
    def __init__(self):
        Block.__init__(self)
        
    def get_output_gt_tensor_feed(self):
        raise NotImplementedError("Not implemented virtual function.")
    
    
    
class SoftmaxLayer(TargetLayer):
    """ Softmax for classification. Note, this is simply the softmax function, after adding bias. Not a ConvLayer """
    
    def __init__(self):
        TargetLayer.__init__(self)
        self._numberOfOutputClasses = None
        self._temperature = None
        
    def makeLayer(  self,
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
        
        bias_layer = BiasLayer(self.input["train"].shape[1])
        logits_train = bias_layer.apply(self.input["train"])
        logits_val = bias_layer.apply(self.input["val"])
        logits_test = bias_layer.apply(self.input["test"])
        self._params = self._params + bias_layer.trainable_params()
        
        
        # ============ Softmax ==============
        self.p_y_given_x_train = tf.nn.softmax(logits_train/t, axis=1)
        self.y_pred_train = tf.argmax(self.p_y_given_x_train, axis=1)
        self.p_y_given_x_val = tf.nn.softmax(logits_val/t, axis=1)
        self.y_pred_val = tf.argmax(self.p_y_given_x_val, axis=1)
        self.p_y_given_x_test = tf.nn.softmax(logits_test/t, axis=1)
        self.y_pred_test = tf.argmax(self.p_y_given_x_test, axis=1)
    
        self._setBlocksOutputAttributes(self.p_y_given_x_train, self.p_y_given_x_val, self.p_y_given_x_test)
        
        layerConnected.setTargetBlock(self)
        
    def get_output_gt_tensor_feed(self):
        # Input. Dimensions of y labels: [batchSize, r, c, z]
        y_gt_train = tf.compat.v1.placeholder(dtype="int32", shape=[None, None, None, None], name="y_train")
        y_gt_val = tf.compat.v1.placeholder(dtype="int32", shape=[None, None, None, None], name="y_val")
        return (y_gt_train, y_gt_val)
        
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
    
    
