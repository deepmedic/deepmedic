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
# Block -> ConvBlock -> LowRankConvBlock
#                L-----> ConvBlockWithSoftmax

class Layer(object):
    def apply(self, input):
        # mode: "train" or "infer"
        raise NotImplementedError()
    def trainable_params(self):
        raise NotImplementedError()
    
class ConvolutionalLayer(Layer):
    def __init__(self, filter_shape, init_method, rng) :
        # filter_shape of dimensions: list/np.array: [#FMs in this layer, #FMs in input, kern-dim-x, kern-dim-y, kern-dim-z]
        std_init = self._get_std_init(init_method, filter_shape)
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=filter_shape), dtype='float32')
        self._w = tf.Variable(w_init, dtype="float32", name="W") # w shape: [#FMs of this layer, #FMs of Input, x, y, z]
        
    def _get_std_init(self, init_method, filter_shape):
        if init_method[0] == "normal" :
            std_init = init_method[1] # commonly 0.01 from Krizhevski
        elif init_method[0] == "fanIn" :
            var_scale = init_method[1] # 2 for init ala Delving into Rectifier, 1 for SNN.
            std_init = np.sqrt( var_scale / np.prod(filter_shape[1:]))
        return std_init
    
    def apply(self, input):
        return ops.conv_3d(input, self._w)

    def trainable_params(self):
        return [self._w]
    

class LowRankConvolutionalLayer(ConvolutionalLayer):
        # Behaviour: Create W, set self._W, set self._params, convolve, return ouput and outputShape.
        # The created filters are either 1-dimensional (rank=1) or 2-dim (rank=2), depending  on the self._rank
        # If 1-dim: rSubconv is the input convolved with the row-1dimensional filter.
        # If 2-dim: rSubconv is the input convolved with the RC-2D filter, cSubconv with CZ-2D filter, zSubconv with ZR-2D filter. 

    def __init__(self, filter_shape, init_method, rng) :
        self._filter_shape = filter_shape # For _crop_sub_outputs_same_dims_and_concat(). Could be done differently?
        std_init = self._get_std_init(init_method, filter_shape)
                
        x_subfilter_shape = [filter_shape[0]//3, filter_shape[1], filter_shape[2], 1 if self._rank == 1 else filter_shape[3], 1]
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=x_subfilter_shape), dtype='float32')
        self._w_x = tf.Variable(w_init, dtype="float32", name="w_x")
        
        y_subfilter_shape = [filter_shape[0]//3, filter_shape[1], 1, filter_shape[3], 1 if self._rank == 1 else filter_shape[4]]
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=y_subfilter_shape), dtype='float32')
        self._w_y = tf.Variable(w_init, dtype="float32", name="w_y")
        
        n_fms_left = filter_shape[0] - 2*(filter_shape[0]//3) # Cause of possibly inexact integer division.
        z_subfilter_shape = [n_fms_left, filter_shape[1], 1 if self._rank == 1 else filter_shape[2], 1, filter_shape[4]]
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=z_subfilter_shape), dtype='float32')
        self._w_z = tf.Variable(w_init, dtype="float32", name="w_z")

    def trainable_params(self):
        return [self._w_x, self._w_y, self._w_z] # Note: these tensors have different shapes! Treat carefully.
    
    def apply(self, input):
        out_x = ops.conv_3d(input, self._w_x)
        out_y = ops.conv_3d(input, self._w_y)
        out_z = ops.conv_3d(input, self._w_z)
        # concatenate together.
        out = self._crop_sub_outputs_same_dims_and_concat(out_x, out_y, out_z)
        return out
        
    def _crop_sub_outputs_same_dims_and_concat(self, tens_x, tens_y, tens_z):
        assert (tens_x.shape[0] == tens_y.shape[0]) and (tens_y.shape[0] == tens_z.shape[0]) # batch-size
        conv_tens_shape = [tens_x.shape[0],
                           tens_x.shape[1] + tens_y.shape[1] + tens_z.shape[1],
                           tens_x.shape[2],
                           tens_y.shape[3],
                           tens_z.shape[4]
                           ]
        x_crop_slice = slice( (self._filter_shape[2]-1)//2, (self._filter_shape[2]-1)//2 + conv_tens_shape[2] )
        y_crop_slice = slice( (self._filter_shape[3]-1)//2, (self._filter_shape[3]-1)//2 + conv_tens_shape[3] )
        z_crop_slice = slice( (self._filter_shape[4]-1)//2, (self._filter_shape[4]-1)//2 + conv_tens_shape[4] )
        tens_x_crop = tens_x[:,:, :, y_crop_slice if self._rank == 1 else slice(0, MAX_INT), z_crop_slice  ]
        tens_y_crop = tens_y[:,:, x_crop_slice, :, z_crop_slice if self._rank == 1 else slice(0, MAX_INT) ]
        tens_z_crop = tens_z[:,:, x_crop_slice if self._rank == 1 else slice(0, MAX_INT), y_crop_slice, : ]
        conc_tens = tf.concat([tens_x_crop, tens_y_crop, tens_z_crop], axis=1) #concatenate the FMs
        return conc_tens
    

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
    

