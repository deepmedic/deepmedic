# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
from math import ceil

def calculateSubsampledImagePartDimensionsFromImagePartSizeAndSubsampleFactor(imagePartDimensions, rec_field_hr, subsampleFactor) :
    """
    This function gives you how big your subsampled-image-part should be, so that it corresponds to the correct number of central-voxels in the normal-part. Currently, it's coupled with the patch-size of the normal-scale. I.e. the subsampled-patch HAS TO BE THE SAME SIZE as the normal-scale, and corresponds to subFactor*patchsize in context.
    When the central voxels are not a multiple of the subFactor, you get ceil(), so +1 sub-patch. When the CNN repeats the pattern, it is giving dimension higher than the central-voxels of the normal-part, but then they are sliced-down to the correct number (in the cnn_make_model function, right after the repeat).        
    This function works like this because of getImagePartFromSubsampledImageForTraining(), which gets a subsampled-image-part by going 1 normal-patch back from the top-left voxel of a normal-scale-part, and then 3 ahead. If I change it to start from the top-left-CENTRAL-voxel back and front, I will be able to decouple the normal-patch size and the subsampled-patch-size. 
    """
    # imagePartDimensions: [dim-x, dim-y, dim-z] of input to normal res pathway.
    # rec_field_hr: [rf_x, rf_y, rf_z] at end of normal-res pathway.
    #if patch is 17x17, a 17x17 subPart is cool for 3 voxels with a subsampleFactor. +2 to be ok for the 9x9 centrally classified voxels, so 19x19 sub-part.
    subsampledImagePartDimensions = []
    for rcz_i in range(len(imagePartDimensions)) :
        centralVoxelsInThisDimension = imagePartDimensions[rcz_i] - rec_field_hr[rcz_i] + 1  # ASSUMES SYMMETRIC PATHS
        centralVoxelsInThisDimensionForSubsampledPart = int(ceil(centralVoxelsInThisDimension*1.0/subsampleFactor[rcz_i]))
        sizeOfSubsampledImagePartInThisDimension = rec_field_hr[rcz_i] + centralVoxelsInThisDimensionForSubsampledPart - 1 # ASSUMES SYMMETRIC PATHS
        subsampledImagePartDimensions.append(sizeOfSubsampledImagePartInThisDimension)
    return subsampledImagePartDimensions

# The below should be updated, and calculated in here properly with private function and per block.
def calc_inp_dims_lr_path_to_match_outp_dims(kernel_dims_lr, subs_factor, out_shape_of_hr_path):
    # subs_factor: list [subsampling-factor-x, subfactor-y, subfactor-z]. From pathway.subsFactor()
    # out_shape_of_hr_path: [dim-x, dim-y, dim-z]
    rec_of_pathway = calc_rec_field_of_path_given_kern_dims_w_stride_1(kernel_dims_lr)
    required_inp_dims_to_path = [-1,-1,-1]
    required_outp_dims_of_path = [-1,-1,-1]
    
    for rcz_i in range(3):
        required_outp_dims_of_path[rcz_i] = int(ceil(out_shape_of_hr_path[rcz_i]/(1.0*subs_factor[rcz_i])))
        required_inp_dims_to_path[rcz_i] = rec_of_pathway[rcz_i] + required_outp_dims_of_path[rcz_i] - 1
    return required_inp_dims_to_path

def calc_rec_field_of_path_given_kern_dims_w_stride_1(kern_dims) :
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

