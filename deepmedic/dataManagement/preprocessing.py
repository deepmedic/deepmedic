# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, division

import numpy as np


def reflectImageArrayIfNeeded(reflectFlags, imageArray):
    stepsForReflectionPerDimension = [-1 if reflectFlags[0] else 1,
                                      -1 if reflectFlags[1] else 1,
                                      -1 if reflectFlags[2] else 1]
    
    reflImageArray = imageArray[::stepsForReflectionPerDimension[0],
                                ::stepsForReflectionPerDimension[1],
                                ::stepsForReflectionPerDimension[2]]
    return reflImageArray
    

def calculateTheZeroIntensityOf3dImage(image3d):
    intensityZeroOfChannel = np.mean([image3d[0, 0, 0],
                                      image3d[-1, 0, 0],
                                      image3d[0, -1, 0],
                                      image3d[-1, -1, 0],
                                      image3d[0, 0, -1],
                                      image3d[-1, 0, -1],
                                      image3d[0, -1, -1],
                                      image3d[-1, -1, -1]
                                      ])
    return intensityZeroOfChannel


# The padding / unpadding could probably be done more generically.
# These two pad/unpad should have their own class, and an instance should be created per subject.
# So that unpad gets how much to unpad from the pad.
def padCnnInputs(array1, cnnReceptiveField, imagePartDimensions):  # Works for 2D as well I think.
    # array1: the loaded volume. Not segments.
    # imagePartDimensions: The size of image segments that the cnn gets.
    # So that we calculate the pad that will go to the side of the volume.
    cnnReceptiveFieldArray = np.asarray(cnnReceptiveField, dtype="int16")
    array1Dimensions = np.asarray(array1.shape,dtype="int16")
    if len(array1.shape) != 3 :
        print("ERROR! Given array in padCnnInputs() was expected of 3-dimensions, "
              "but was passed an array of dimensions: ", array1.shape, ", Exiting!")
        exit(1)

    # paddingValue = (array1[0, 0, 0] + array1[-1, 0, 0] + array1[0, -1, 0] + array1[-1, -1, 0] + array1[0, 0, -1]
    #                 + array1[-1, 0, -1] + array1[0, -1, -1] + array1[-1, -1, -1]) / 8.0
    # Calculate how much padding needed to fully infer the original array1, taking only the receptive field in account.
    paddingAtLeftPerAxis = (cnnReceptiveFieldArray - 1) // 2
    paddingAtRightPerAxis = cnnReceptiveFieldArray - 1 - paddingAtLeftPerAxis
    # Now, to cover the case that the specified image-segment of the CNN is larger than the image
    # (eg full-image inference and current image is smaller), pad further to right.
    paddingFurtherToTheRightNeededForSegment = np.maximum(0,
                                                          np.asarray(imagePartDimensions, dtype="int16")
                                                          - (array1Dimensions
                                                             + paddingAtLeftPerAxis+paddingAtRightPerAxis))
    paddingAtRightPerAxis += paddingFurtherToTheRightNeededForSegment
    
    tupleOfPaddingPerAxes = ((paddingAtLeftPerAxis[0], paddingAtRightPerAxis[0]),
                             (paddingAtLeftPerAxis[1], paddingAtRightPerAxis[1]),
                             (paddingAtLeftPerAxis[2], paddingAtRightPerAxis[2]))
    # Very poor design because channels/gt/bmask etc are all getting back a different padding?
    # tupleOfPaddingPerAxes is returned in order for unpad to know.
    return [np.lib.pad(array1, tupleOfPaddingPerAxes, 'reflect'), tupleOfPaddingPerAxes]


# In the 3 first axes. Which means it can take a 4-dim image.
def unpad3dArray(array1, tupleOfPaddingPerAxesLeftRight):
    # tupleOfPaddingPerAxesLeftRight : ( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)).
    unpaddedArray1 = array1[tupleOfPaddingPerAxesLeftRight[0][0]:,
                     tupleOfPaddingPerAxesLeftRight[1][0]:,
                     tupleOfPaddingPerAxesLeftRight[2][0]:]
    # The checks below are to make it work if padding == 0, which may happen for 2D on the 3rd axis.
    unpaddedArray1 = unpaddedArray1[:-tupleOfPaddingPerAxesLeftRight[0][1], :, :] \
        if tupleOfPaddingPerAxesLeftRight[0][1] > 0 else unpaddedArray1
    unpaddedArray1 = unpaddedArray1[:, :-tupleOfPaddingPerAxesLeftRight[1][1], :] \
        if tupleOfPaddingPerAxesLeftRight[1][1] > 0 else unpaddedArray1
    unpaddedArray1 = unpaddedArray1[:, :, :-tupleOfPaddingPerAxesLeftRight[2][1]] \
        if tupleOfPaddingPerAxesLeftRight[2][1] > 0 else unpaddedArray1
        
    return unpaddedArray1


# ====================== (below) Z-Score Intensity Normalization. ==================
# Could make classes? class Normalizer and children? (zscore)

def neg_val_check(img, log):
    is_neg_int = img < 0
    num_voxels_neg_int = np.sum(is_neg_int)
    if num_voxels_neg_int > 0:
        log.print3("WARN: One image has " + str(num_voxels_neg_int) +
                   " voxels with intensity below zero! Unexpected!\n" +
                   "The voxels with negative ints have (min, max, mean) = (" + str(np.min(img[is_neg_int])) + ", " +
                   str(np.max(img[is_neg_int])) + ", " + str(np.mean(img[is_neg_int])) + ").")


def init_norm_params():
    return {'int_norm': False,
            'cutoff_percent': None,
            'cutoff_std': None,
            'cutoff_mean': False}


def get_img_stats(img):
    return np.mean(img), np.std(img), np.max(img)


def get_cutoff_mask(src, low, high):
    low_mask = src > low
    high_mask = src < high

    return low_mask * high_mask


def get_norm_stats(log, src, roi_mask_bool,
                   cutoff_percent=None, cutoff_std=None, cutoff_mean=False,
                   verbose=False, id_str=''):

    neg_val_check(src, log)
    src_mean, src_std, src_max = get_img_stats(src)

    src_roi = src[roi_mask_bool]  # This gets flattened automatically. It's a vector array.
    src_roi_mean, src_roi_std, src_roi_max = get_img_stats(src_roi)

    # Init auxiliary variables
    mask_bool_norm = roi_mask_bool.copy()
    if cutoff_percent:
        cutoff_low = np.percentile(src_roi, cutoff_percent[0])
        cutoff_high = np.percentile(src_roi, cutoff_percent[1])
        mask_bool_norm *= get_cutoff_mask(src, cutoff_low, cutoff_high)
        if verbose:
            log.print3(id_str + "Cutting off intensities with [percentiles] (within Roi). "
                                "Cutoffs: Min=" + str(cutoff_low) + ", High=" + str(cutoff_high))

    if cutoff_std:
        cutoff_low = src_roi_mean - cutoff_std[0] * src_roi_std
        cutoff_high = src_roi_mean + cutoff_std[1] * src_roi_std
        cutoff_mask = get_cutoff_mask(src, cutoff_low, cutoff_high)
        mask_bool_norm *= cutoff_mask
        if verbose:
            log.print3(id_str + "Cutting off intensities with [std] (within Roi). "
                                "Cutoffs: Min=" + str(cutoff_low) + ", High=" + str(cutoff_high))

    if cutoff_mean:
        cutoff_low = src_mean
        mask_bool_norm *= get_cutoff_mask(src, cutoff_low, src_max)  # no high cutoff
        if verbose:
            log.print3(id_str + "Cutting off intensities with [below wholeImageMean for air]. "
                                "Cutoff: Min=" + str(cutoff_low))

    norm_mean, norm_std, _ = get_img_stats(src[mask_bool_norm])

    return norm_mean, norm_std


def print_norm_log(log, norm_params, num_channels, id_str=''):

    cutoff_types = []

    if norm_params['cutoff_percent']:
        cutoff_types += ['Percentile']
    if norm_params['cutoff_std']:
        cutoff_types += ['Standard Deviation']
    if norm_params['cutoff_mean']:
        cutoff_types += ['Whole Image Mean']

    log.print3(id_str + "Normalising " + str(num_channels) + " channel(s) with the following cutoff type(s): " +
               ', '.join(list(cutoff_types)) if cutoff_types else 'None')


def normalise_zscore(log, channels, roi_mask, norm_params=None, id_str='', verbose=False):

    channels_norm = np.zeros(channels.shape)
    roi_mask_bool = roi_mask > 0
    if norm_params is None:
        norm_params = init_norm_params()

    if id_str:
        id_str += ' '

    print_norm_log(log, norm_params, len(channels), id_str=id_str)

    for idx, channel in enumerate(channels):
        norm_mean, norm_std = get_norm_stats(log, channel, roi_mask_bool,
                                             cutoff_percent=norm_params['cutoff_percent'],
                                             cutoff_std=norm_params['cutoff_std'],
                                             cutoff_mean=norm_params['cutoff_mean'],
                                             verbose=verbose,
                                             id_str=id_str)
        # Apply the normalization
        channels_norm[idx] = (channel - norm_mean) / (1.0 * norm_std)

        if verbose:
            old_mean, old_std, _ = get_img_stats(channel)
            log.print3(id_str + "Original image stats(channel " + str(idx) +
                       "): Mean=" + str(old_mean) + ", Std=" + str(old_std))
            log.print3(id_str + "Image was normalized using: Mean=" + str(norm_mean) + ", Std=" + str(norm_std))
            new_mean, new_std, _ = get_img_stats(channels_norm[idx])
            log.print3(id_str + "Normalised image stats(channel " + str(idx) +
                       "): Mean=" + str(new_mean) + ", Std=" + str(new_std))

    return channels_norm

# ====================== (above) Z-Score Intensity Normalization. ==================

