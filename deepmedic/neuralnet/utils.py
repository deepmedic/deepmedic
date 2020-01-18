# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
from math import ceil

def calc_rec_field_of_path_given_kern_dims_w_stride_1(kern_dims): # Used by modelParams.py to find default input-shape. TODO: Remove
    if not kern_dims : #list is []
        return 0
    
    n_dims = len(kern_dims[0])
    receptive_field = [1]*n_dims
    for dim_idx in range(n_dims) :
        for layer_idx in range(len(kern_dims)) :
            receptive_field[dim_idx] += kern_dims[layer_idx][dim_idx] - 1
    return receptive_field
    
def checkRecFieldVsSegmSize(receptiveFieldDim, segmentDim) :
    numberOfRFDim = len(receptiveFieldDim)
    numberOfSegmDim = len(segmentDim)
    if numberOfRFDim != numberOfSegmDim :
        print("ERROR: [in function checkRecFieldVsSegmSize()] : Receptive field and image segment have different number of dimensions! (should be 3 for both! Exiting!)")
        exit(1)
    for dim_i in range(numberOfRFDim) :
        if receptiveFieldDim[dim_i] > segmentDim[dim_i] :
            print("ERROR: [in function checkRecFieldVsSegmSize()] : The segment-size (input) should be at least as big as the receptive field of the model! This was not found to hold! Dimensions of Receptive Field:", receptiveFieldDim, ". Dimensions of Segment: ", segmentDim)
            return False
    return True

def checkKernDimPerLayerCorrect3dAndNumLayers(kernDimensionsPerLayer, numOfLayers) :
    #kernDimensionsPerLayer : a list with sublists. One sublist per layer. Each sublist should have 3 integers, specifying the dimensions of the kernel at the corresponding layer of the pathway. eg: kernDimensionsPerLayer = [ [3,3,3], [3,3,3], [5,5,5] ] 
    if kernDimensionsPerLayer == None or len(kernDimensionsPerLayer) != numOfLayers :
        return False
    for kernDimInLayer in kernDimensionsPerLayer :
        if len(kernDimInLayer) != 3 :
            return False
    return True

def checkSubsampleFactorEven(subFactor) :
    for dim_i in range(len(subFactor)) :
        if subFactor[dim_i]%2 != 1 :
            return False
    return True

