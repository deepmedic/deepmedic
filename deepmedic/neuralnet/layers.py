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

class Layer(object):
    def apply(self, input):
        # mode: "train" or "infer"
        raise NotImplementedError()
    def trainable_params(self):
        raise NotImplementedError()
    
class ConvolutionalLayer(Layer):
    def __init__(self, filter_shape, init_method, rng) :
        # filter_shape of dimensions: list/np.array: [#FMs in this layer, #FMs in input, kern-dim-x, kern-dim-y, kern-dim-z]
        if init_method[0] == "normal" :
            std_init = init_method[1] # commonly 0.01 from Krizhevski
        elif init_method[0] == "fanIn" :
            var_scale = init_method[1] # 2 for init ala Delving into Rectifier, 1 for SNN.
            std_init = np.sqrt( var_scale / np.prod(filter_shape[1:]))
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=filter_shape), dtype='float32')
        self._w = tf.Variable(w_init, dtype="float32", name="W") # w shape: [#FMs of this layer, #FMs of Input, x, y, z]
        
    def apply(self, input):
        # input weight matrix W has shape: [ #ChannelsOut, #ChannelsIn, R, C, Z ]
        # Input signal given in shape [BatchSize, Channels, R, C, Z]
        
        # Tensorflow's Conv3d requires filter shape: [ D/Z, H/C, W/R, C_in, C_out ] #ChannelsOut, #ChannelsIn, Z, R, C ]
        w_resh = tf.transpose(self._w, perm=[4,3,2,1,0])
        # Conv3d requires signal in shape: [BatchSize, Channels, Z, R, C]
        input_resh = tf.transpose(input, perm=[0,4,3,2,1])
        output = tf.nn.conv3d(input = input_resh, # batch_size, time, num_of_input_channels, rows, columns
                              filters = w_resh, # TF: Depth, Height, Wight, Chans_in, Chans_out
                              strides = [1,1,1,1,1],
                              padding = "VALID",
                              data_format = "NDHWC"
                              )
        #Output is in the shape of the input image (signals_shape).
        output = tf.transpose(output, perm=[0,4,3,2,1]) #reshape the result, back to the shape of the input image.
        return output

    def trainable_params(self):
        return [self._w]
    
class DropoutLayer(Layer):
    def __init__(self, dropout_rate, rng):
        self._keep_prob = 1 - dropout_rate
        self._rng = rng
        
    def apply(self, input, mode):
        if self._keep_prob > 0.999: #Dropout below 0.001 I take it as if there is no dropout. To avoid float problems with drop == 0.0
            return input
        
        if mode == "train":
            random_tensor = self._keep_prob
            random_tensor += tf.random.uniform(shape=tf.shape(input), minval=0., maxval=1., seed=self._rng.randint(999999), dtype="float32")
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            dropout_mask = tf.floor(random_tensor)
            output = input * dropout_mask
        elif mode == "infer":
            output = input * self._keep_prob
        else:
            raise NotImplementedError()
        
        return output
    
    def trainable_params(self):
        return []
    
class BiasLayer(Layer):
    def __init__(self, n_channels):
        self._b = tf.Variable(np.zeros((n_channels), dtype = 'float32'), name="b")
        
    def apply(self, input):
        # self._b.shape[0] should already be input.shape[1] number of input channels.
        return input + tf.reshape(self._b, shape=[1,input.shape[1],1,1,1])
    
    def trainable_params(self):
        return [self._b]
    
class BatchNormLayer(Layer):
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
            
class PreluLayer(Layer):
    def __init__(self, n_channels, alpha=0.01):
        self._a = tf.Variable(np.ones((n_channels), dtype='float32')*alpha, name="aPrelu")

    def apply(self, input):
        # input is a tensor of shape (batchSize, FMs, r, c, z)
        return ops.prelu(input, tf.reshape(self._a, shape=[1,input.shape[1],1,1,1]) )
    
    def trainable_params(self):
        return [self._a]
    
class IdentityLayer(Layer):
    def apply(self, input): return input
    def trainable_params(self): return []
    
class ReluLayer(Layer):
    def apply(self, input): return ops.relu(input)
    def trainable_params(self): return []

class EluLayer(Layer):
    def apply(self, input): return ops.elu(input)
    def trainable_params(self): return []
    
class SeluLayer(Layer):
    def apply(self, input): return ops.selu(input)
    def trainable_params(self): return []
    
class Block(object):
    
    def __init__(self) :
        # === Input to the layer ===
        self.input= {"train": None, "val": None, "test": None}
        
        # === Basic architecture parameters === 
        self._numberOfFeatureMaps = None
        self._poolingParameters = None
        
        #=== All Trainable Parameters of the Block ===
        self._bn_l = None # Keep track to update moving avg. Only when rollingAverageForBn>0 AND useBnFlag, with the latter used for the 1st layers of pathways (on image).
        
        # All trainable parameters
        # NOTE: VIOLATED _HIDDEN ENCAPSULATION BY THE FUNCTION THAT TRANSFERS PRETRAINED WEIGHTS deepmed.neuralnet.transferParameters.transferParametersBetweenLayers.
        # TEMPORARY TILL THE API GETS FIXED (AFTER DA)!
        self._params = [] # W, (gbn), b, (aPrelu)
        self._params_for_L1_L2_reg = []
        # self._W = None # Careful. LowRank does not set this. Uses ._WperSubconv
        
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
        if self._bn_l is not None :
            self._bn_l.update_arrays_of_bn_moving_avg(sessionTf)
            
    def get_update_ops_for_bn_moving_avg(self) :
        return self._bn_l.get_update_ops_for_bn_moving_avg() if self._bn_l is not None else []
        
class ConvLayer(Block):
    
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
                dropoutRate) :
        # ---------------- Order of what is applied -----------------
        #  Input -> [ BatchNorm OR biases applied] -> NonLinearity -> DropOut -> Pooling --> Conv ] # ala He et al "Identity Mappings in Deep Residual Networks" 2016
        # -----------------------------------------------------------
        
        #---------------------------------------------------------
        #------------------ Batch Normalization ------------------
        #---------------------------------------------------------
        if useBnFlag and movingAvForBnOverXBatches > 0 :
            self._bn_l = BatchNormLayer(movingAvForBnOverXBatches, n_channels=inputToLayerTrain.shape[1])            
            self._params = self._params + self._bn_l.trainable_params()
            
            inputToNonLinearityTrain = self._bn_l.apply(inputToLayerTrain, mode="train")
            inputToNonLinearityVal = self._bn_l.apply(inputToLayerVal, mode="infer")
            inputToNonLinearityTest = self._bn_l.apply(inputToLayerTest, mode="infer")
            
        else : #Not using batch normalization
            #make the bias terms and apply them. Like the old days before BN's own learnt bias terms.
            bias_l = BiasLayer(inputToLayerTrain.shape[1])
            inputToNonLinearityTrain = bias_l.apply(inputToLayerTrain)
            inputToNonLinearityVal = bias_l.apply(inputToLayerVal)
            inputToNonLinearityTest = bias_l.apply(inputToLayerTest)
            self._params = self._params + bias_l.trainable_params()
            
            
        #--------------------------------------------------------
        #------------ Apply Activation/ non-linearity -----------
        #--------------------------------------------------------
        if activationFunc == "linear" : # -1 stands for "no nonlinearity". Used for input layers of the pathway.
            act_l = IdentityLayer()
        elif activationFunc == "relu" :
            act_l = ReluLayer()
        elif activationFunc == "prelu" :
            act_l = PreluLayer(inputToNonLinearityTrain.shape[1])
            
        elif activationFunc == "elu" :
            act_l = EluLayer()
        elif activationFunc == "selu" :
            act_l = SeluLayer()
        inputToDropoutTrain = act_l.apply(inputToNonLinearityTrain)
        inputToDropoutVal = act_l.apply(inputToNonLinearityVal)
        inputToDropoutTest = act_l.apply(inputToNonLinearityTest)
        self._params = self._params + act_l.trainable_params()
        
        #------------------------------------
        #------------- Dropout --------------
        #------------------------------------
        dropout_l = DropoutLayer(dropoutRate, rng)
        inputToPoolTrain = dropout_l.apply(inputToDropoutTrain, mode="train")
        inputToPoolVal = dropout_l.apply(inputToDropoutVal, mode="infer")
        inputToPoolTest = dropout_l.apply(inputToDropoutTest, mode="infer")
        self._params = self._params + dropout_l.trainable_params()
        
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
        conv_l = ConvolutionalLayer(filterShape, convWInitMethod, rng)
        out_train = conv_l.apply(inputToConvTrain)
        out_val = conv_l.apply(inputToConvVal)
        out_test = conv_l.apply(inputToConvTest)
        self._params = conv_l.trainable_params() + self._params
        self._params_for_L1_L2_reg = self._params_for_L1_L2_reg + conv_l.trainable_params()
        
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
        cost = 0
        for prm in self._params_for_L1_L2_reg:
            cost += tf.reduce_sum(tf.abs(prm))
        return cost
    
    def _get_L2_cost(self) : #Called for L2 weigths regularisation
        cost = 0
        for prm in self._params_for_L1_L2_reg:
            cost += tf.reduce_sum(prm ** 2)
        return cost
    
    
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
        
        bias_l = BiasLayer(self.input["train"].shape[1])
        logits_train = bias_l.apply(self.input["train"])
        logits_val = bias_l.apply(self.input["val"])
        logits_test = bias_l.apply(self.input["test"])
        self._params = self._params + bias_l.trainable_params()
        
        
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
    
    
