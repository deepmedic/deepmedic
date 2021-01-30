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
# >> Any operation that has "trainable" parameters is a layer. 
#

class Layer(object):
    def apply(self, input, mode):
        # mode: "train" or "infer"
        raise NotImplementedError()
    
    def trainable_params(self):
        raise NotImplementedError()
    
    def params_for_L1_L2_reg(self):
        return []
    
    def rec_field(self, rf_at_inp, stride_rf_at_inp):
        # stride_rf_at_inp: [str-x, str-y, str-z
        #                stride of the rec field at prev layer wrt cnn's inp
        # Returns: receptive field [x,y,z] of neurons in this input, and ...
        #          stride of rf wrt input-image when shifting between 2 consequtive neurons in this layer
        return rf_at_inp, stride_rf_at_inp
    
    def calc_outp_dims_given_inp(self, inp_dims):
        return inp_dims
    
    def calc_inp_dims_given_outp(self, outp_dims):
        return outp_dims
    
class PoolingLayer(Layer):
    def __init__(self, window_size, strides, pad_mode, pool_mode):
        # window_size: [wx, wy, wz]
        # strides: [sx, sy, sz]
        # mode: 'MAX' or 'AVG'
        # mirror_pad: [mirrorPad-x,-y,-z]
        self._window_size = window_size
        self._strides = strides
        self._pad_mode = pad_mode
        self._pool_mode = pool_mode
        
    def apply(self, input, _):
        # input dimensions: (batch, fms, r, c, z)
        return ops.pool_3d(input, self._window_size, self._strides, self._pad_mode, self._pool_mode)
        
    def trainable_params(self):
        return []
    
    def _n_padding(self):
        # Returns [padx,pady,padz], how much pad would have been added to preserve dimensions ('SAME' or 'MIRROR').
        return [0,0,0] if self._pad_mode == 'VALID' else [self._window_size[d] - 1 for d in range(3)]
    
    def rec_field(self):
        raise NotImplementedError()
    
    def calc_outp_dims_given_inp(self, inp_dims): # Same as conv.
        padding = self._n_padding()
        if np.any([inp_dims[d] + padding[d] < self._window_size[d] for d in range(3)]):
            return [0,0,0]
        else:
            return [1 + (inp_dims[d] + padding[d] - self._window_size[d]) // self._strides[d] for d in range(3)]
    
    def calc_inp_dims_given_outp(self, outp_dims):
        assert np.all([outp_dims[d] > 0 for d in range(3)])
        padding = self._n_padding()
        return [(outp_dims[d]-1)*self._strides[d] + self._window_size[d] - padding[d] for d in range(3)]
    
class ConvolutionalLayer(Layer):
    def __init__(self, fms_in, fms_out, conv_kernel_dims, init_method, pad_mode, rng):
        # filter_shape of dimensions: list/np.array: [#FMs in this layer, #FMs in input, kern-dim-x, kern-dim-y, kern-dim-z]
        std_init = self._get_std_init(init_method, fms_in, fms_out, conv_kernel_dims)
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=[fms_out, fms_in] + conv_kernel_dims), dtype='float32')
        self._w = tf.Variable(w_init, dtype="float32", name="W") # w shape: [#FMs of this layer, #FMs of Input, x, y, z]
        self._strides = [1,1,1]
        self._pad_mode = pad_mode
        
    def _get_std_init(self, init_method, fms_in, fms_out, conv_kernel_dims):
        if init_method[0] == "normal" :
            std_init = init_method[1] # commonly 0.01 from Krizhevski
        elif init_method[0] == "fanIn" :
            var_scale = init_method[1] # 2 for init ala Delving into Rectifier, 1 for SNN.
            std_init = np.sqrt(var_scale/np.prod([fms_in] + conv_kernel_dims))
        return std_init
    
    def apply(self, input, mode):
        return ops.conv_3d(input, self._w, self._pad_mode)

    def trainable_params(self):
        return [self._w]
    
    def params_for_L1_L2_reg(self):
        return self.trainable_params()
    
    def _n_padding(self):
        # Returns [padx,pady,padz], how much pad would have been added to preserve dimensions ('SAME' or 'MIRROR').
        return [0,0,0] if self._pad_mode == 'VALID' else [self._w.shape.as_list()[2+d] - 1 for d in range(3)]
    
    def rec_field(self, rf_at_inp, stride_rf_at_inp):
        rf_out = [rf_at_inp[d] + (self._w.shape.as_list()[2+d]-1)*stride_rf_at_inp[d] for d in range(3)]
        stride_rf = [stride_rf_at_inp[d]*self._strides[d] for d in range(3)]
        return rf_out, stride_rf
    
    def calc_outp_dims_given_inp(self, inp_dims):
        padding = self._n_padding()
        if np.any([inp_dims[d] + padding[d] < self._w.shape.as_list()[2+d] for d in range(3)]):
            return [0,0,0]
        else:
            return [1 + (inp_dims[d] + padding[d] - self._w.shape.as_list()[2+d]) // self._strides[d] for d in range(3)]
    
    def calc_inp_dims_given_outp(self, outp_dims):
        assert np.all([outp_dims[d] > 0 for d in range(3)])
        padding = self._n_padding()
        return [(outp_dims[d]-1)*self._strides[d] + self._w.shape.as_list()[2+d] - padding[d] for d in range(3)]
    
    
class LowRankConvolutionalLayer(ConvolutionalLayer):
        # Behaviour: Create W, set self._W, set self._params, convolve, return ouput and outputShape.
        # The created filters are either 1-dimensional (rank=1) or 2-dim (rank=2), depending  on the self._rank
        # If 1-dim: rSubconv is the input convolved with the row-1dimensional filter.
        # If 2-dim: rSubconv is the input convolved with the RC-2D filter, cSubconv with CZ-2D filter, zSubconv with ZR-2D filter. 

    def __init__(self, fms_in, fms_out, conv_kernel_dims, init_method, pad_mode, rng) :
        self._conv_kernel_dims = conv_kernel_dims # For _crop_sub_outputs_same_dims_and_concat(). Could be done differently?
        std_init = self._get_std_init(init_method, fms_in, fms_out, conv_kernel_dims)
                
        x_subfilter_shape = [fms_out//3, fms_in, conv_kernel_dims[0], 1 if self._rank == 1 else conv_kernel_dims[1], 1]
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=x_subfilter_shape), dtype='float32')
        self._w_x = tf.Variable(w_init, dtype="float32", name="w_x")
        
        y_subfilter_shape = [fms_out//3, fms_in, 1, conv_kernel_dims[1], 1 if self._rank == 1 else conv_kernel_dims[2]]
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=y_subfilter_shape), dtype='float32')
        self._w_y = tf.Variable(w_init, dtype="float32", name="w_y")
        
        n_fms_left = fms_out - 2*(fms_out//3) # Cause of possibly inexact integer division.
        z_subfilter_shape = [n_fms_left, fms_in, 1 if self._rank == 1 else conv_kernel_dims[0], 1, conv_kernel_dims[2]]
        w_init = np.asarray(rng.normal(loc=0.0, scale=std_init, size=z_subfilter_shape), dtype='float32')
        self._w_z = tf.Variable(w_init, dtype="float32", name="w_z")
        
        self._strides = [1,1,1]
        self._pad_mode = pad_mode
        
    def trainable_params(self):
        return [self._w_x, self._w_y, self._w_z] # Note: these tensors have different shapes! Treat carefully.
    
    def params_for_L1_L2_reg(self):
        return self.trainable_params()
    
    def apply(self, input, mode):
        out_x = ops.conv_3d(input, self._w_x, self._pad_mode)
        out_y = ops.conv_3d(input, self._w_y, self._pad_mode)
        out_z = ops.conv_3d(input, self._w_z, self._pad_mode)
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
        x_crop_slice = slice( (self._conv_kernel_dims[0]-1)//2, (self._conv_kernel_dims[0]-1)//2 + conv_tens_shape[2] )
        y_crop_slice = slice( (self._conv_kernel_dims[1]-1)//2, (self._conv_kernel_dims[1]-1)//2 + conv_tens_shape[3] )
        z_crop_slice = slice( (self._conv_kernel_dims[2]-1)//2, (self._conv_kernel_dims[2]-1)//2 + conv_tens_shape[4] )
        tens_x_crop = tens_x[:,:, :, y_crop_slice if self._rank == 1 else slice(0, MAX_INT), z_crop_slice  ]
        tens_y_crop = tens_y[:,:, x_crop_slice, :, z_crop_slice if self._rank == 1 else slice(0, MAX_INT) ]
        tens_z_crop = tens_z[:,:, x_crop_slice if self._rank == 1 else slice(0, MAX_INT), y_crop_slice, : ]
        conc_tens = tf.concat([tens_x_crop, tens_y_crop, tens_z_crop], axis=1) #concatenate the FMs
        return conc_tens
    
    def _n_padding(self):
        # Returns [padx,pady,padz], how much pad would have been added to preserve dimensions ('SAME' or 'MIRROR').
        if self._pad_mode == 'VALID':
            padding = [0,0,0]
        else:
            padding = [self._w_x.shape[2] - 1, self._w_y.shape[3] - 1, self._w_z.shape[4] - 1]
        return padding
    
    def rec_field(self, rf_at_inp, stride_rf_at_inp):
        rf_out = [rf_at_inp[0] + (self._w_x.shape[2]-1)*stride_rf_at_inp[0],
                  rf_at_inp[1] + (self._w_y.shape[3]-1)*stride_rf_at_inp[1],
                  rf_at_inp[2] + (self._w_z.shape[4]-1)*stride_rf_at_inp[2]]
        stride_rf = [stride_rf_at_inp[d]*self._strides[d] for d in range(3)]
        return rf_out, stride_rf
    
    def calc_outp_dims_given_inp(self, inp_dims):
        padding = self._n_padding()
        if np.any([inp_dims[d] + padding[d] < self._w.shape.as_list()[2+d] for d in range(3)]):
            return [0,0,0]
        else:
            return [1 + (inp_dims[0] + padding[0] - self._w_x.shape.as_list()[2]) // self._strides[0],
                    1 + (inp_dims[1] + padding[1] - self._w_y.shape.as_list()[3]) // self._strides[1],
                    1 + (inp_dims[2] + padding[2] - self._w_z.shape.as_list()[4]) // self._strides[2]]
               
    def calc_inp_dims_given_outp(self, outp_dims):
        assert np.all([outp_dims[d] > 0 for d in range(3)])
        padding = self._n_padding()
        return [(outp_dims[0]-1)*self._strides[0] + self._w_x.shape.as_list()[2] - padding[0],
                (outp_dims[1]-1)*self._strides[1] + self._w_y.shape.as_list()[3] - padding[1],
                (outp_dims[2]-1)*self._strides[2] + self._w_z.shape.as_list()[4] - padding[2]]
    
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
        

    def apply(self, input, _):
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
        
        self._idx_where_moving_avg_is = 0 #Index in the rolling-average matrices of the layers, of the entry to update in the next batch. Could be tf.Var.

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
            

def get_act_layer(act_str, n_fms_in):
    if act_str == "linear" : # -1 stands for "no nonlinearity". Used for input layers of the pathway.
        return IdentityLayer()
    elif act_str == "relu" :
        return ReluLayer()
    elif act_str == "prelu" :
        return PreluLayer(n_fms_in)
    elif act_str == "elu" :
        return EluLayer()
    elif act_str == "selu" :
        return SeluLayer()

class PreluLayer(Layer):
    def __init__(self, n_channels, alpha=0.01):
        self._a = tf.Variable(np.ones((n_channels), dtype='float32')*alpha, name="aPrelu")

    def apply(self, input, _):
        # input is a tensor of shape (batchSize, FMs, r, c, z)
        return ops.prelu(input, tf.reshape(self._a, shape=[1,input.shape[1],1,1,1]) )
    
    def trainable_params(self):
        return [self._a]
    
class IdentityLayer(Layer):
    def apply(self, input, _): return input
    def trainable_params(self): return []
    
class ReluLayer(Layer):
    def apply(self, input, _): return ops.relu(input)
    def trainable_params(self): return []

class EluLayer(Layer):
    def apply(self, input, _): return ops.elu(input)
    def trainable_params(self): return []
    
class SeluLayer(Layer):
    def apply(self, input, _): return ops.selu(input)
    def trainable_params(self): return []
    

