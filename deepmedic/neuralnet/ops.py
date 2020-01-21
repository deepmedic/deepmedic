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

def conv_3d(input, w, padding="VALID"):
    # input weight matrix W has shape: [ #ChannelsOut, #ChannelsIn, R, C, Z ]
    # Input signal given in shape [BatchSize, Channels, R, C, Z]
    # padding: 'VALID', 'SAME' or 'MIRROR'
    # Tensorflow's Conv3d requires filter shape: [ D/Z, H/C, W/R, C_in, C_out ] #ChannelsOut, #ChannelsIn, Z, R, C ]
    if padding in ['MIRROR', 'mirror']: # If mirror, do it here and perform conv as if not pad ('SAME')
        input = pad_by_mirroring(input, n_vox_pad_per_dim=[w.shape[2+d] - 1 for d in range(3)])
        padding = 'VALID'
    elif padding in ['ZERO', 'zero']:
        padding = 'SAME'
    elif padding is None or padding in ['none', 'VALID', 'valid']:
        padding = 'VALID'
        
    w_resh = tf.transpose(w, perm=[4,3,2,1,0])
    # Conv3d requires signal in shape: [BatchSize, Channels, Z, R, C]
    input_resh = tf.transpose(input, perm=[0,4,3,2,1])
    output = tf.nn.conv3d(input = input_resh, # batch_size, time, num_of_input_channels, rows, columns
                          filters = w_resh, # TF: Depth, Height, Wight, Chans_in, Chans_out
                          strides = [1,1,1,1,1],
                          padding = padding,
                          data_format = "NDHWC"
                          )
    #Output is in the shape of the input image (signals_shape).
    output = tf.transpose(output, perm=[0,4,3,2,1]) #reshape the result, back to the shape of the input image.
    return output

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


def pool_3d(input, window_size, strides, pad_mode, pool_mode) :
    # input dimensions: (batch, fms, r, c, z)
    # poolParams: [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode]
    # pool_mode: 'MAX' or 'AVG'
    if pad_mode in ['MIRROR', 'mirror']:
        input = pad_by_mirroring(input, n_vox_pad_per_dim=[window_size.shape[2+d] - 1 for d in range(3)])
        pad_mode = 'VALID'
    elif padding in ['ZERO', 'zero']:
        padding = 'SAME'
    elif padding is None or padding in ['none']:
        padding = 'VALID'
        
    inp_resh = tf.transpose(input, perm=[0,4,3,2,1]) # Channels last.
    pooled_out = tf.nn.pool(input = inp_resh,
                            window_shape=window_size,
                            strides=strides,
                            padding=pad_mode, # SAME or VALID
                            pooling_type=pool_mode,
                            data_format="NDHWC") # AVG or MAX
    pooled_out = tf.transpose(pooled_out, perm=[0,4,3,2,1])
    
    return pooled_out


def crop_center(fms, listOfNumberOfCentralVoxelsToGetPerDimension) :
    # fms: a 5D tensor, [batch, fms, r, c, z]
    # listOfNumberOfCentralVoxelsToGetPerDimension: list of 3 scalars or Tensorflow 1D tensor (eg from tf.shape(x)). [r, c, z]
    # NOTE: Because of the indexing in the end, the shape returned is commonyl (None, Fms, None, None, None). Should be reshape to preserve shape.
    fmsShape = tf.shape(fms) #fms.shape works too.
    # if part is of even width, one voxel to the left is the centre.
    rCentreOfPartIndex = (fmsShape[2] - 1) // 2
    rIndexToStartGettingCentralVoxels = rCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[0] - 1) // 2
    rIndexToStopGettingCentralVoxels = rIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[0]  # Excluding
    cCentreOfPartIndex = (fmsShape[3] - 1) // 2
    cIndexToStartGettingCentralVoxels = cCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[1] - 1) // 2
    cIndexToStopGettingCentralVoxels = cIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[1]  # Excluding
    zCentreOfPartIndex = (fmsShape[4] - 1) // 2
    zIndexToStartGettingCentralVoxels = zCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[2] - 1) // 2
    zIndexToStopGettingCentralVoxels = zIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[2]  # Excluding
    return fms[    :, :,
                rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels,
                zIndexToStartGettingCentralVoxels : zIndexToStopGettingCentralVoxels]


def crop_to_match_dims(input, dims_to_match):
    # dims_to_match : [ batch size, num of fms, r, c, z] 
    output = input[:,
                   :,
                   :dims_to_match[2],
                   :dims_to_match[3],
                   :dims_to_match[4]]
    return output


def make_residual_connection(tensor_1, tensor_2) :
    # tensor_1: earlier tensor
    # tensor_2: deeper tensor
    # Add the outputs of the two layers and return the output, as well as its dimensions.
    # tensor_2 & tensor_1: 5D tensors [batchsize, chans, x, y, z], outputs of deepest and earliest layer of the Res.Conn.
    # Result: Shape of result should be exactly the same as the output of Deeper layer.
    tens_1_shape = tf.shape(tensor_1)
    tens_2_shape = tf.shape(tensor_2)
    # Get part of the earlier layer that is of the same dimensions as the FMs of the deeper:
    tens_1_center_crop = crop_center(tensor_1, tens_2_shape[2:])
    # Add the FMs, after taking care of zero padding if the deeper layer has more FMs.
    if tensor_2.get_shape()[1] >= tensor_1.get_shape()[1] : # ifs not allowed via tensor (from tf.shape(...))
        blank_channels = tf.zeros(shape=[tens_2_shape[0],
                                         tens_2_shape[1] - tens_1_shape[1],
                                         tens_2_shape[2], tens_2_shape[3], tens_2_shape[4]], dtype="float32")
        res_out = tensor_2 + tf.concat( [tens_1_center_crop, blank_channels], axis=1)

    else : # Deeper FMs are fewer than earlier. This should not happen in most architectures. But oh well...
        res_out = tensor_2 + tens_1_center_crop[:, :tens_2_shape[1], :,:,:]
    # The following reshape is to enforce the 4 dimensions to be "visible" to TF (cause the indexing in crop_center makes them dynamic/None)
    res_out = tf.reshape(res_out, shape=[-1, res_out.shape[1], tensor_2.shape[2], tensor_2.shape[3], tensor_2.shape[4]])
    return res_out


def upsample_by_repeat(input, up_factors):
    # input: [batch size, num of FMs, r, c, z]. Ala input/output of conv layers.
    # Repeat FM in the three last dimensions, to upsample back to the normal resolution space.
    # In numpy below: (but tf has no repeat, only tile, so, implementation is funny.
    # up_factors: list of upsampling factors per (3d) dimension. [up-x, up-y, up-z]
    res = input
    res_shape = tf.shape(input) # Dynamic. For batch and r,c,z dimensions. (unknown prior to runtime)
    n_fms = input.get_shape()[1] # Static via get_shape(). Known. For reshape to return tensor with *known* shape[1].
    # If tf.shape()[1] is used, reshape changes res.get_shape()[1] to (?).
    
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1]*res_shape[2], 1, res_shape[3], res_shape[4]] ),
                               multiples=[1, 1, up_factors[0], 1, 1] ),
                    shape=[res_shape[0], n_fms, res_shape[2]*up_factors[0], res_shape[3], res_shape[4]] )
    res_shape = tf.shape(res)
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1], res_shape[2]*res_shape[3], 1, res_shape[4]] ),
                               multiples=[1, 1, 1, up_factors[1], 1] ),
                    shape=[res_shape[0], n_fms, res_shape[2], res_shape[3]*up_factors[1], res_shape[4]] )
    res_shape = tf.shape(res)
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1], res_shape[2], res_shape[3]*res_shape[4], 1] ),
                               multiples=[1, 1, 1, 1, up_factors[2]] ),
                    shape=[res_shape[0], n_fms, res_shape[2], res_shape[3], res_shape[4]*up_factors[2]] )
    return res
    
def upsample_5D_tens_and_crop(input, up_factors, upsampl_type="repeat", dims_to_match=None) :
    # input: [batch_size, numberOfFms, r, c, z].
    # up_factors: list of upsampling factors per (3d) dimension. [up-x, up-y, up-z]
    if upsampl_type == "repeat" :
        out_hr = upsample_by_repeat(input, up_factors)
    else :
        print("ERROR: in upsample_5D_tens_and_crop(...). Not implemented type of upsampling! Exiting!"); exit(1)
        
    if dims_to_match is not None:
        # If the central-voxels are eg 10, the susampled-part will have 4 central voxels. Which above will be repeated to 3*4 = 12.
        # I need to clip the last ones, to have the same dimension as the input from 1st pathway, which will have dimensions equal to the centrally predicted voxels (10)
        out_hr_crop = crop_to_match_dims(out_hr, dims_to_match)
    else :
        out_hr_crop = out_hr
        
    return out_hr_crop


def pad_by_mirroring(input, n_vox_pad_per_dim):
    # input shape: [batchSize, #channels#, r, c, z]
    # inputImageDimensions : [ batchSize, #channels, dim r, dim c, dim z ] of input
    # n_vox_pad_per_dim shape: [ num o voxels in r-dim to add, ...c-dim, ...z-dim ]
    # If n_vox_pad_per_dim is odd, 1 more voxel is added to the right side.
    # r-axis
    assert np.all(n_vox_pad_per_dim) >= 0
    padLeft = int(n_vox_pad_per_dim[0] // 2); padRight = int((n_vox_pad_per_dim[0] + 1) // 2);
    paddedImage = tf.concat([input[:, :, int(n_vox_pad_per_dim[0] // 2) - 1::-1 , :, :], input], axis=2) if padLeft > 0 else input
    paddedImage = tf.concat([paddedImage, paddedImage[ :, :, -1:-1 - int((n_vox_pad_per_dim[0] + 1) // 2):-1, :, :]], axis=2) if padRight > 0 else paddedImage
    # c-axis
    padLeft = int(n_vox_pad_per_dim[1] // 2); padRight = int((n_vox_pad_per_dim[1] + 1) // 2);
    paddedImage = tf.concat([paddedImage[:, :, :, padLeft - 1::-1 , :], paddedImage], axis=3) if padLeft > 0 else paddedImage
    paddedImage = tf.concat([paddedImage, paddedImage[:, :, :, -1:-1 - padRight:-1, :]], axis=3) if padRight > 0 else paddedImage
    # z-axis
    padLeft = int(n_vox_pad_per_dim[2] // 2); padRight = int((n_vox_pad_per_dim[2] + 1) // 2)
    paddedImage = tf.concat([paddedImage[:, :, :, :, padLeft - 1::-1 ], paddedImage], axis=4) if padLeft > 0 else paddedImage
    paddedImage = tf.concat([paddedImage, paddedImage[:, :, :, :, -1:-1 - padRight:-1]], axis=4) if padRight > 0 else paddedImage
    
    return paddedImage

