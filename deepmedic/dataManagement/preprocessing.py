# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, division

import numpy as np
    

def calc_border_int_of_3d_img(img_3d):
    border_int = np.mean([img_3d[0, 0, 0],
                          img_3d[-1, 0, 0],
                          img_3d[0, -1, 0],
                          img_3d[-1, -1, 0],
                          img_3d[0, 0, -1],
                          img_3d[-1, 0, -1],
                          img_3d[0, -1, -1],
                          img_3d[-1, -1, -1]
                          ])
    return border_int

# ============= Padding =======================

def calc_pad_per_axis(pad_input_imgs, dims_img, rec_field, dims_highres_segment):
    # rec_field: size of CNN's receptive field. [x,y,z]
    # dims_highres_segment: The size of image segments that the cnn gets.
    #     So that we calculate the pad that will go to the side of the volume.
    if not pad_input_imgs:
        return ((0, 0), (0, 0), (0, 0))
    
    rec_field_array = np.asarray(rec_field, dtype="int16")
    dims_img_arr = np.asarray(dims_img,dtype="int16")
    dims_segm_arr = np.asarray(dims_highres_segment, dtype="int16")
    # paddingValue = (img[0, 0, 0] + img[-1, 0, 0] + img[0, -1, 0] + img[-1, -1, 0] + img[0, 0, -1]
    #                 + img[-1, 0, -1] + img[0, -1, -1] + img[-1, -1, -1]) / 8.0
    # Calculate how much padding needed to fully infer the original img, taking only the receptive field in account.
    pad_left = (rec_field_array - 1) // 2
    pad_right = rec_field_array - 1 - pad_left
    # Now, to cover the case that the specified size for sampled image-segment is larger than the image
    # (eg full-image inference and current image is smaller), pad further to right.
    extra_pad_right = np.maximum(0, dims_segm_arr - (dims_img_arr + pad_left + pad_right))
    pad_right += extra_pad_right
    
    pad_left_right_per_axis = ((pad_left[0], pad_right[0]),
                               (pad_left[1], pad_right[1]),
                               (pad_left[2], pad_right[2]))
    
    return pad_left_right_per_axis

# The padding / unpadding could probably be done more generically.
# These pad/unpad should have their own class, and an instance should be created per subject.
# So that unpad gets how much to unpad from the pad.
def pad_imgs_of_case(channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat,
                     pad_input_imgs, rec_field, dims_highres_segment):
    # channels: np.array of dimensions [n_channels, x-dim, y-dim, z-dim]
    # gt_lbl_img: np.array
    # roi_mask: np.array
    # wmaps_to_sample_per_cat: np.array of dimensions [num_categories, x-dim, y-dim, z-dim]
    # dims_highres_segment: list [x,y,z] of dimensions of the normal-resolution samples for cnn.
    # Returns:
    # pad_left_right_axes: Padding added before and after each axis. All 0s if no padding.
    
    # Padding added before and after each axis. ((0, 0), (0, 0), (0, 0)) if no pad.
    pad_left_right_per_axis = calc_pad_per_axis(pad_input_imgs, channels[0].shape, rec_field, dims_highres_segment)
    if not pad_input_imgs:
        return channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat, pad_left_right_axes
    
    channels = pad_4d_arr(channels, pad_left_right_per_axis)

    if gt_lbl_img is not None:
        gt_lbl_img = pad_3d_img(gt_lbl_img, pad_left_right_per_axis)
    
    if roi_mask is not None:
        roi_mask = pad_3d_img(roi_mask, pad_left_right_per_axis)
    
    if wmaps_to_sample_per_cat is not None:
        wmaps_to_sample_per_cat = pad_4d_arr(wmaps_to_sample_per_cat, pad_left_right_per_axis)
    
    return channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat, pad_left_right_per_axis

def pad_4d_arr(arr_4d, pad_left_right_per_axis_3d):
    # Do not pad first dimension. E.g. for channels or weightmaps, [n_chans,x,y,z]
    pad_left_right_per_axis_4d = ((0,0),) + pad_left_right_per_axis_3d
    return np.lib.pad(arr_4d, pad_left_right_per_axis_4d, 'reflect')

def pad_3d_img(img, pad_left_right_per_axis):
    # img: 3D array.
    # pad_left_right_per_axis is returned in order for unpadding to know how much to remove.
    return np.lib.pad(img, pad_left_right_per_axis, 'reflect')

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


# ============================ (below) Intensity Normalization. ==================================
# Could make classes? class Normalizer and children? (zscore)

def normalize_int_of_imgs(log, channels, roi_mask, prms, id_str):
    if prms is not None:
        channels = normalize_int_zscore(log, channels, roi_mask, prms['zscore'], id_str)
    return channels

# ===== (below) Z-Score Intensity Normalization. =====
    
def neg_val_check(img, log):
    is_neg_int = img < 0
    num_voxels_neg_int = np.sum(is_neg_int)
    if num_voxels_neg_int > 0:
        log.print3("WARN: One image has " + str(num_voxels_neg_int) +
                   " voxels with intensity below zero! Unexpected!\n" +
                   "The voxels with negative ints have (min, max, mean) = (" + str(np.min(img[is_neg_int])) + ", " +
                   str(np.max(img[is_neg_int])) + ", " + str(np.mean(img[is_neg_int])) + ").")


def default_zscore_prms():
    # For BRATS: cutoff_perc: [5., 95], cutoff_times_std: [2., 2.], cutoff_below_mean: True
    return {'apply': False, # True/False
            'cutoff_percents': None, # None or [low, high] with each 0.0 to 1.0
            'cutoff_times_std': None, # None or [low, high] with each positive Float
            'cutoff_below_mean': False} # True/False


def get_img_stats(img):
    return np.mean(img), np.std(img), np.max(img)


def get_cutoff_mask(src, low, high):
    low_mask = src > low
    high_mask = src < high

    return low_mask * high_mask


def get_norm_stats(log, src, roi_mask_bool,
                   cutoff_percents=None, cutoff_times_std=None, cutoff_below_mean=False,
                   verbose=False, id_str=''):

    neg_val_check(src, log)
    src_mean, src_std, src_max = get_img_stats(src)

    src_roi = src[roi_mask_bool]  # This gets flattened automatically. It's a vector array.
    src_roi_mean, src_roi_std, src_roi_max = get_img_stats(src_roi)

    # Init auxiliary variables
    mask_bool_norm = roi_mask_bool.copy()
    if cutoff_percents:
        cutoff_low = np.percentile(src_roi, cutoff_percents[0])
        cutoff_high = np.percentile(src_roi, cutoff_percents[1])
        mask_bool_norm *= get_cutoff_mask(src, cutoff_low, cutoff_high)
        if verbose:
            log.print3(id_str + "Cutting off intensities with [percentiles] (within Roi). "
                                "Cutoffs: Min=" + str(cutoff_low) + ", High=" + str(cutoff_high))

    if cutoff_times_std:
        cutoff_low = src_roi_mean - cutoff_times_std[0] * src_roi_std
        cutoff_high = src_roi_mean + cutoff_times_std[1] * src_roi_std
        cutoff_mask = get_cutoff_mask(src, cutoff_low, cutoff_high)
        mask_bool_norm *= cutoff_mask
        if verbose:
            log.print3(id_str + "Cutting off intensities with [std] (within Roi). "
                                "Cutoffs: Min=" + str(cutoff_low) + ", High=" + str(cutoff_high))

    if cutoff_below_mean:
        cutoff_low = src_mean
        mask_bool_norm *= get_cutoff_mask(src, cutoff_low, src_max)  # no high cutoff
        if verbose:
            log.print3(id_str + "Cutting off intensities with [below wholeImageMean for air]. "
                                "Cutoff: Min=" + str(cutoff_low))

    norm_mean, norm_std, _ = get_img_stats(src[mask_bool_norm])

    return norm_mean, norm_std


def print_norm_log(log, norm_prms, num_channels, id_str=''):

    cutoff_types = []

    if norm_prms['cutoff_percents']:
        cutoff_types += ['Percentile']
    if norm_prms['cutoff_times_std']:
        cutoff_types += ['Standard Deviation']
    if norm_prms['cutoff_below_mean']:
        cutoff_types += ['Whole Image Mean']

    log.print3(id_str + "Normalizing " + str(num_channels) + " channel(s) with the following cutoff type(s): " +
               ', '.join(list(cutoff_types)) if cutoff_types else 'None')


def normalize_int_zscore(log, channels, roi_mask, norm_prms, id_str='', verbose=False):

    channels_norm = np.zeros(channels.shape)
    roi_mask_bool = roi_mask > 0
    if norm_prms is None:
        norm_prms = default_zscore_prms()
        
    if id_str:
        id_str += ' '

    print_norm_log(log, norm_prms, len(channels), id_str=id_str)

    for idx, channel in enumerate(channels):
        norm_mean, norm_std = get_norm_stats(log, channel, roi_mask_bool,
                                             cutoff_percents=norm_prms['cutoff_percents'],
                                             cutoff_times_std=norm_prms['cutoff_times_std'],
                                             cutoff_below_mean=norm_prms['cutoff_below_mean'],
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
            log.print3(id_str + "Normalized image stats(channel " + str(idx) +
                       "): Mean=" + str(new_mean) + ", Std=" + str(new_std))

    return channels_norm

# ====================== (above) Z-Score Intensity Normalization. ==================

# ================= Others ========================
# Deprecated
def reflect_array_if_needed(reflect_flags, arr):
    strides_for_refl_per_dim = [-1 if reflect_flags[0] else 1,
                              -1 if reflect_flags[1] else 1,
                              -1 if reflect_flags[2] else 1]
    
    refl_arr = arr[::strides_for_refl_per_dim[0],
                   ::strides_for_refl_per_dim[1],
                   ::strides_for_refl_per_dim[2]]
    return refl_arr

