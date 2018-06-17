# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, division

import numpy as np


def reflectImageArrayIfNeeded(reflectFlags, imageArray) :
    stepsForReflectionPerDimension = [-1 if reflectFlags[0] else 1, -1 if reflectFlags[1] else 1, -1 if reflectFlags[2] else 1]
    
    reflImageArray = imageArray[::stepsForReflectionPerDimension[0], ::stepsForReflectionPerDimension[1], ::stepsForReflectionPerDimension[2]]
    return reflImageArray
    

def calculateTheZeroIntensityOf3dImage(image3d) :
    intensityZeroOfChannel = np.mean([image3d[0,0,0],
                                      image3d[-1,0,0],
                                      image3d[0,-1,0],
                                      image3d[-1,-1,0],
                                      image3d[0,0,-1],
                                      image3d[-1,0,-1],
                                      image3d[0,-1,-1],
                                      image3d[-1,-1,-1]
                                      ])
    return intensityZeroOfChannel


# The padding / unpadding could probably be done more generically.
#These two pad/unpad should have their own class, and an instance should be created per subject. So that unpad gets how much to unpad from the pad.
def padCnnInputs(array1, cnnReceptiveField, imagePartDimensions) : #Works for 2D as well I think.
    # array1: the loaded volume. Not segments.
    # imagePartDimensions: The size of image segments that the cnn gets. So that we calculate the pad that will go to the side of the volume.
    cnnReceptiveFieldArray = np.asarray(cnnReceptiveField, dtype="int16")
    array1Dimensions = np.asarray(array1.shape,dtype="int16")
    if len(array1.shape) != 3 :
        print("ERROR! Given array in padCnnInputs() was expected of 3-dimensions, but was passed an array of dimensions: ", array1.shape,", Exiting!")
        exit(1)
    #paddingValue = (array1[0,0,0] + array1[-1,0,0] + array1[0,-1,0] + array1[-1,-1,0] + array1[0,0,-1] + array1[-1,0,-1] + array1[0,-1,-1] + array1[-1,-1,-1]) / 8.0
    #Calculate how much padding needed to fully infer the original array1, taking only the receptive field in account.
    paddingAtLeftPerAxis = (cnnReceptiveFieldArray - 1) // 2
    paddingAtRightPerAxis = cnnReceptiveFieldArray - 1 - paddingAtLeftPerAxis
    #Now, to cover the case that the specified image-segment of the CNN is larger than the image (eg full-image inference and current image is smaller), pad further to right.
    paddingFurtherToTheRightNeededForSegment = np.maximum(0, np.asarray(imagePartDimensions,dtype="int16")-(array1Dimensions+paddingAtLeftPerAxis+paddingAtRightPerAxis))
    paddingAtRightPerAxis += paddingFurtherToTheRightNeededForSegment
    
    tupleOfPaddingPerAxes = ( (paddingAtLeftPerAxis[0],paddingAtRightPerAxis[0]), (paddingAtLeftPerAxis[1],paddingAtRightPerAxis[1]), (paddingAtLeftPerAxis[2],paddingAtRightPerAxis[2]))
    #Very poor design because channels/gt/bmask etc are all getting back a different padding? tupleOfPaddingPerAxes is returned in order for unpad to know.
    return [np.lib.pad(array1, tupleOfPaddingPerAxes, 'reflect' ), tupleOfPaddingPerAxes]


#In the 3 first axes. Which means it can take a 4-dim image.
def unpadCnnOutputs(array1, tupleOfPaddingPerAxesLeftRight) :
    #tupleOfPaddingPerAxesLeftRight : ( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)).
    unpaddedArray1 = array1[tupleOfPaddingPerAxesLeftRight[0][0]:, tupleOfPaddingPerAxesLeftRight[1][0]:, tupleOfPaddingPerAxesLeftRight[2][0]:]
    #The checks below are to make it work if padding == 0, which may happen for 2D on the 3rd axis.
    unpaddedArray1 = unpaddedArray1[:-tupleOfPaddingPerAxesLeftRight[0][1],:,:] if tupleOfPaddingPerAxesLeftRight[0][1] > 0 else unpaddedArray1 
    unpaddedArray1 = unpaddedArray1[:,:-tupleOfPaddingPerAxesLeftRight[1][1],:] if tupleOfPaddingPerAxesLeftRight[1][1] > 0 else unpaddedArray1
    unpaddedArray1 = unpaddedArray1[:,:,:-tupleOfPaddingPerAxesLeftRight[2][1]] if tupleOfPaddingPerAxesLeftRight[2][1] > 0 else unpaddedArray1
    return unpaddedArray1

    
# Deprecated and currently unused.
def getSuggestedStdForSubsampledImage(subsampleFactor) :
    arraySubsampledFactor = np.asarray(subsampleFactor)
    suggestedStdsForSubsampledChannels = arraySubsampledFactor/2.0
    #if subsampledFactor == 1 for a certain axis (eg z axis), it means I am actually doing 2D processing. In this case, use std=0 on this axis, and I dont smooth at all. I do clean slice-by-slice.
    suggestedStdsForSubsampledChannels = suggestedStdsForSubsampledChannels * (arraySubsampledFactor!=1)
    return suggestedStdsForSubsampledChannels

