# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
import copy
from math import ceil

import tensorflow as tf

from deepmedic.neuralnet.pathwayTypes import PathwayTypes
from deepmedic.neuralnet.utils import calcRecFieldFromKernDimListPerLayerWhenStrides1
from deepmedic.neuralnet.blocks import ConvBlock, LowRankConvBlock


#################################################################
#                         Pathway Types                         #
#################################################################

def cropRczOf5DimArrayToMatchOther(array5DimToCrop, dimensionsOf5DimArrayToMatchInRcz):
    # dimensionsOf5DimArrayToMatchInRcz : [ batch size, num of fms, r, c, z] 
    output = array5DimToCrop[:,
                            :,
                            :dimensionsOf5DimArrayToMatchInRcz[2],
                            :dimensionsOf5DimArrayToMatchInRcz[3],
                            :dimensionsOf5DimArrayToMatchInRcz[4]]
    return output
    
def repeatRcz5DimArrayByFactor(array5Dim, factor3Dim):
    # array5Dim: [batch size, num of FMs, r, c, z]. Ala input/output of conv layers.
    # Repeat FM in the three last dimensions, to upsample back to the normal resolution space.
    # In numpy below: (but tf has no repeat, only tile, so, implementation is funny.
    #expandedR = array5Dim.repeat(factor3Dim[0], axis=2)
    #expandedRC = expandedR.repeat(factor3Dim[1], axis=3)
    #expandedRCZ = expandedRC.repeat(factor3Dim[2], axis=4)
    res = array5Dim
    res_shape = tf.shape(array5Dim) # Dynamic. For batch and r,c,z dimensions. (unknown prior to runtime)
    n_fms = array5Dim.get_shape()[1] # Static via get_shape(). Known. For reshape to return tensor with *known* shape[1].
    # If tf.shape()[1] is used, reshape changes res.get_shape()[1] to (?).
    
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1]*res_shape[2], 1, res_shape[3], res_shape[4]] ),
                               multiples=[1, 1, factor3Dim[0], 1, 1] ),
                    shape=[res_shape[0], n_fms, res_shape[2]*factor3Dim[0], res_shape[3], res_shape[4]] )
    res_shape = tf.shape(res)
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1], res_shape[2]*res_shape[3], 1, res_shape[4]] ),
                               multiples=[1, 1, 1, factor3Dim[1], 1] ),
                    shape=[res_shape[0], n_fms, res_shape[2], res_shape[3]*factor3Dim[1], res_shape[4]] )
    res_shape = tf.shape(res)
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1], res_shape[2], res_shape[3]*res_shape[4], 1] ),
                               multiples=[1, 1, 1, 1, factor3Dim[2]] ),
                    shape=[res_shape[0], n_fms, res_shape[2], res_shape[3], res_shape[4]*factor3Dim[2]] )
    return res
    
def upsampleRcz5DimArrayAndOptionalCrop(array5dimToUpsample,
                                        upsamplingFactor,
                                        upsamplingScheme="repeat",
                                        dimensionsOf5DimArrayToMatchInRcz=None) :
    # array5dimToUpsample : [batch_size, numberOfFms, r, c, z].
    if upsamplingScheme == "repeat" :
        upsampledOutput = repeatRcz5DimArrayByFactor(array5dimToUpsample, upsamplingFactor)
    else :
        print("ERROR: in upsampleRcz5DimArrayAndOptionalCrop(...). Not implemented type of upsampling! Exiting!"); exit(1)
        
    if dimensionsOf5DimArrayToMatchInRcz is not None:
        # If the central-voxels are eg 10, the susampled-part will have 4 central voxels. Which above will be repeated to 3*4 = 12.
        # I need to clip the last ones, to have the same dimension as the input from 1st pathway, which will have dimensions equal to the centrally predicted voxels (10)
        output = cropRczOf5DimArrayToMatchOther(upsampledOutput, dimensionsOf5DimArrayToMatchInRcz)
    else :
        output = upsampledOutput
        
    return output
    
def getMiddlePartOfFms(fms, listOfNumberOfCentralVoxelsToGetPerDimension) :
    # fms: a 5D tensor, [batch, fms, r, c, z]
    # listOfNumberOfCentralVoxelsToGetPerDimension: list of 3 scalars or Tensorflow 1D tensor (eg from tf.shape(x)). [r, c, z]
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
    return fms[	:, :,
                rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels,
                zIndexToStartGettingCentralVoxels : zIndexToStopGettingCentralVoxels]

        
def make_residual_connection(log, deeperLOut, earlierLOut) :
    # Add the outputs of the two layers and return the output, as well as its dimensions.
    # deeperLOut & earlierLOut: 5D tensors [batchsize, chans, x, y, z], outputs of deepest and earliest layer of the Res.Conn.
    # Result: Shape of result should be exactly the same as the output of Deeper layer.
    deeperLOutShape = tf.shape(deeperLOut)
    earlierLOutShape = tf.shape(earlierLOut)
    # Get part of the earlier layer that is of the same dimensions as the FMs of the deeper:
    partOfEarlierFmsToAddTrain = getMiddlePartOfFms(earlierLOut, deeperLOutShape[2:])
    # Add the FMs, after taking care of zero padding if the deeper layer has more FMs.
    if deeperLOut.get_shape()[1] >= earlierLOut.get_shape()[1] : # ifs not allowed via tensor (from tf.shape(...))
        zeroFmsToConcatTrain = tf.zeros(shape=[deeperLOutShape[0],
                                               deeperLOutShape[1] - earlierLOutShape[1],
                                               deeperLOutShape[2], deeperLOutShape[3], deeperLOutShape[4]], dtype="float32")
        outputOfResConnTrain = deeperLOut + tf.concat( [partOfEarlierFmsToAddTrain, zeroFmsToConcatTrain], axis=1)

    else : # Deeper FMs are fewer than earlier. This should not happen in most architectures. But oh well...
        outputOfResConnTrain = deeperLOut + partOfEarlierFmsToAddTrain[:, :deeperLOutShape[1], :,:,:]
        
    # Dimensions of output are the same as those of the deeperLayer
    return outputOfResConnTrain
    
    
#################################################################
#                        Classes of Pathways                    #
#################################################################
        
class Pathway(object):
    # This is a virtual class.
    
    def __init__(self, pName=None) :
        self._pName = pName
        self._pType = None # Pathway Type.
        
        # === Input to the pathway ===
        self._input_shape = {"train": None, "val": None, "test": None}
        self._n_fms_in = None
        # === Basic architecture parameters === 
        self._blocks = []
        self._subs_factor = [1,1,1]
        self._recField = None # At the end of pathway
        self._inds_of_blocks_for_res_conns_at_out = None
        
        # === Output of the block ===
        self._n_fms_out = None
        
    # Getters
    def get_number_fms_in(self):
        return self._n_fms_in
    def get_number_fms_out(self):
        return self._n_fms_out
    
    def apply(self, input, mode, train_val_test, verbose=False, log=None):
        # mode: 'train' / 'infer'
        if verbose:
            log.print3("\tPathway ["+str(self.getStringType())+"], Mode: [" + mode + "], Input's Shape: " + str(input.shape))
            
        input_to_prev_layer = None
        input_to_next_layer = input
        
        for idx, block in enumerate(self._blocks):
            if verbose:
                log.print3("\tBlock [" + str(idx) + "], Mode: [" + mode + "], Input's Shape: " + str(input_to_next_layer.shape))
            
            out = block.apply(input_to_next_layer, mode)
            block.output[train_val_test] = out # HACK TEMPORARY
            
            if idx not in self._inds_of_blocks_for_res_conns_at_out: #not a residual connecting here
                input_to_prev_layer = input_to_next_layer
                input_to_next_layer = out
            else : #make residual connection
                assert layer_i > 0 # The very first block (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                out_res = make_residual_connection(log, out, input_to_prev_layer)
                input_to_prev_layer = input_to_next_layer
                input_to_next_layer = out_res
                
        if verbose:
            log.print3("\tPathway ["+str(self.getStringType())+"], Mode: [" + mode + "], Output's Shape: " + str(input_to_next_layer.shape))
            log.print3("Pathway [" + str(self.getStringType()) + "] done.")
        
        return input_to_next_layer
    


    def build(self,
              log,
              rng,
              n_input_channels,
              num_kerns_per_layer,
              conv_kernel_dims_per_layer,
              conv_w_init_method,
              use_bn_per_layer, # As a flag for case that I want to apply BN on input image. I want to apply to input of FC.
              moving_avg_length,
              activ_func_per_layer,
              dropout_rate_per_layer=[],
              pool_prms_for_path = [],
              inds_of_lower_rank_convs=[],
              ranks_of_lower_rank_convs = [],
              inds_of_blocks_for_res_conns_at_out=[]
              ):
        log.print3("[Pathway_" + str(self.getStringType()) + "] is being built...")

        self._n_fms_in = n_input_channels
        self._recField = self.calcRecFieldOfPathway(conv_kernel_dims_per_layer)
        self._inds_of_blocks_for_res_conns_at_out = inds_of_blocks_for_res_conns_at_out
        
        n_fms_input_to_prev_layer = None
        n_fms_input_to_next_layer = n_input_channels
        n_blocks = len(num_kerns_per_layer)
        for layer_i in range(0, n_blocks) :

            if layer_i in inds_of_lower_rank_convs :
                block = LowRankConvBlock(ranks_of_lower_rank_convs[ inds_of_lower_rank_convs.index(layer_i) ])
            else : # normal conv block
                block = ConvBlock()
                
            log.print3("\tBlock [" + str(layer_i) + "], FMs-In: " + str(n_fms_input_to_next_layer) +\
                                               ", FMs-Out: " + str(num_kerns_per_layer[layer_i]) +\
                                               ", Conv Filter dimensions: " + str(conv_kernel_dims_per_layer[layer_i]))
            block.build(rng,
                        n_fms_in=n_fms_input_to_next_layer,
                        n_fms_out=num_kerns_per_layer[layer_i],
                        conv_kernel_dims=conv_kernel_dims_per_layer[layer_i],
                        pool_prms=pool_prms_for_path[layer_i],
                        conv_w_init_method=conv_w_init_method,
                        use_bn = use_bn_per_layer[layer_i],
                        moving_avg_length=moving_avg_length,
                        activ_func=activ_func_per_layer[layer_i],
                        dropout_rate=dropout_rate_per_layer[layer_i] if len(dropout_rate_per_layer) > 0 else 0
                        )
            self._blocks.append(block)
            
            if layer_i not in inds_of_blocks_for_res_conns_at_out: #not a residual connecting here
                n_fms_input_to_prev_layer = n_fms_input_to_next_layer
                n_fms_input_to_next_layer = num_kerns_per_layer[layer_i]
            else : #make residual connection
                log.print3("\t[Pathway_"+str(self.getStringType())+"]: making Residual Connection between output of [Layer_"+str(layer_i)+"] to input of previous block.")
                assert layer_i > 0 # The very first block (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                n_fms_res_out = max(num_kerns_per_layer[layer_i], n_fms_input_to_prev_layer)
                n_fms_input_to_prev_layer = n_fms_input_to_next_layer
                n_fms_input_to_next_layer = n_fms_res_out
                
        self._n_fms_out = n_fms_input_to_next_layer
                
        
    # The below should be updated, and calculated in here properly with private function and per block.
    def calcRecFieldOfPathway(self, conv_kernel_dims_per_layer) :
        return calcRecFieldFromKernDimListPerLayerWhenStrides1(conv_kernel_dims_per_layer)
        
    def calcInputRczDimsToProduceOutputFmsOfCompatibleDims(self, thisPathWayKernelDims, dimsOfOutputFromPrimaryPathway):
        recFieldAtEndOfPathway = self.calcRecFieldOfPathway(thisPathWayKernelDims)
        rczDimsOfInputToPathwayShouldBe = [-1,-1,-1]
        rczDimsOfOutputOfPathwayShouldBe = [-1,-1,-1]
        
        rczDimsOfOutputFromPrimaryPathway = dimsOfOutputFromPrimaryPathway[2:]
        for rcz_i in range(3) :
            rczDimsOfOutputOfPathwayShouldBe[rcz_i] = int(ceil(rczDimsOfOutputFromPrimaryPathway[rcz_i]/(1.0*self.subsFactor()[rcz_i])))
            rczDimsOfInputToPathwayShouldBe[rcz_i] = recFieldAtEndOfPathway[rcz_i] + rczDimsOfOutputOfPathwayShouldBe[rcz_i] - 1
        return rczDimsOfInputToPathwayShouldBe
        
    # Setters
    def set_input_shape(self, in_shape_tr, in_shape_val, in_shape_test):
        self._input_shape["train"] = in_shape_tr; self._input_shape["val"] = in_shape_val; self._input_shape["test"] = in_shape_test
            
    # Getters
    def pName(self):
        return self._pName
    def pType(self):
        return self._pType
    def get_blocks(self):
        return self._blocks
    def get_block(self, index):
        return self._blocks[index]
    def subsFactor(self):
        return self._subs_factor
    def getShapeOfInput(self, train_val_test_str):
        assert train_val_test_str in ["train", "val", "test"]
        return self._input_shape[train_val_test_str]
    
    # Other API :
    def getStringType(self) : raise NotImplementedMethod() # Abstract implementation. Children classes should implement this.
    
class NormalPathway(Pathway):
    def __init__(self, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.NORM
    # Override parent's abstract classes.
    def getStringType(self) :
        return "NORMAL"
        
class SubsampledPathway(Pathway):
    def __init__(self, subsamplingFactor, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.SUBS
        self._subs_factor = subsamplingFactor
        
    def upsampleOutputToNormalRes(self, input, shapeToMatchInRcz, upsamplingScheme="repeat"):
        output = upsampleRcz5DimArrayAndOptionalCrop(input, self.subsFactor(), upsamplingScheme, shapeToMatchInRcz)
        return output
    
    # OVERRIDING parent's classes.
    def getStringType(self) :
        return "SUBSAMPLED" + str(self.subsFactor())
                    
             
class FcPathway(Pathway):
    def __init__(self, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.FC
    # Override parent's abstract classes.
    def getStringType(self) :
        return "FC"



