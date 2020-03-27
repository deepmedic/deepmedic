# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, division

import numpy as np
import time


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

def calc_pad_per_axis(pad_input_imgs, dims_img, dims_rec_field, dims_highres_segment):
    # dims_rec_field: size of CNN's receptive field. [x,y,z]
    # dims_highres_segment: The size of image segments that the cnn gets.
    #     So that we calculate the pad that will go to the side of the volume.
    if not pad_input_imgs:
        return ((0, 0), (0, 0), (0, 0))
    
    rec_field_array = np.asarray(dims_rec_field, dtype="int16")
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
                     pad_input_imgs, unpred_margin):
    # channels: np.array of dimensions [n_channels, x-dim, y-dim, z-dim]
    # gt_lbl_img: np.array
    # roi_mask: np.array
    # wmaps_to_sample_per_cat: np.array of dimensions [num_categories, x-dim, y-dim, z-dim]
    # pad_input_imgs: Boolean, do padding or not.
    # unpred_margin: [[pre-x, post-x], [pre-y, post-y], [pre-z, post-z]], number voxels not predicted
    # Returns:
    # pad_left_right_axes: Padding added before and after each axis. All 0s if no padding.
    if not pad_input_imgs:
        pad_left_right_axis = [[0, 0], [0, 0], [0, 0]]
        return channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat, pad_left_right_axis
                         
    # Padding added before and after each axis.
    pad_left_right_per_axis = unpred_margin
    
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
    pad_left_right_per_axis_4d = [[0,0],] + pad_left_right_per_axis_3d
    return np.lib.pad(arr_4d, pad_left_right_per_axis_4d, 'reflect')

def pad_3d_img(img, pad_left_right_per_axis):
    # img: 3D array.
    # pad_left_right_per_axis is returned in order for unpadding to know how much to remove.
    return np.lib.pad(img, pad_left_right_per_axis, 'reflect')

# In the 3 first axes. Which means it can take a 4-dim image.
def unpad_3d_img(img, padding_left_right_per_axis):
    # img: 3d array
    # padding_left_right_per_axis: ((pad-l-x,pad-r-x), (pad-l-y,pad-r-y), (pad-l-z,pad-r-z))
    unpadded_img = img[padding_left_right_per_axis[0][0]:,
                       padding_left_right_per_axis[1][0]:,
                       padding_left_right_per_axis[2][0]:]
    # The checks below are to make it work if padding == 0, which may happen for 2D on the 3rd axis.
    unpadded_img = unpadded_img[:-padding_left_right_per_axis[0][1], :, :] \
        if padding_left_right_per_axis[0][1] > 0 else unpadded_img
    unpadded_img = unpadded_img[:, :-padding_left_right_per_axis[1][1], :] \
        if padding_left_right_per_axis[1][1] > 0 else unpadded_img
    unpadded_img = unpadded_img[:, :, :-padding_left_right_per_axis[2][1]] \
        if padding_left_right_per_axis[2][1] > 0 else unpadded_img
        
    return unpadded_img


# ============================ (below) Intensity Normalization. ==================================
# Could make classes? class Normalizer and children? (zscore)

# Main normalization method. This calls each different type of normalizer.
def normalize_int_of_subj(log, channels, roi_mask, prms, job_id):
    # prms = {'verbose_lvl': 0/1/2
    #         'zscore': None or dictionary with parameters
    #        }
    
    if prms is None:
        return channels
    
    verbose_lvl = prms['verbose_lvl'] if 'verbose_lvl' in prms else 0
    norms_applied = []

    time_0 = time.time()
    # TODO: window_ints(min, max)
    # TODO: linear_rescale_to(new_min, new_max)
    if 'zscore' in prms:
        channels, applied = normalize_zscore_subj(log, channels, roi_mask, prms['zscore'], verbose_lvl, job_id)
        if applied:
            norms_applied.append('zscore')

    if verbose_lvl >= 1:
        log.print3(job_id + " Normalized subject's images with " +str(norms_applied) + ". " +\
                   "Took [{0:.1f}".format(time.time() - time_0) + "] secs")
        
    return channels

# ===== (below) Z-Score Intensity Normalization. =====

def get_img_stats(img, calc_mean=True, calc_std=True, calc_max=True):
    mean = np.mean(img) if calc_mean else None
    std = np.std(img) if calc_std else None
    max = np.max(img) if calc_max else None
    return mean, std, max


def get_cutoff_mask(img, low, high):
    low_mask = img > low
    high_mask = img < high
    return low_mask * high_mask


def normalize_zscore_img(img, roi_mask_bool,
                         cutoff_percents, cutoff_times_std, cutoff_below_mean,
                         get_stats_info=False):
    #     cutoff_percents  : Percentile cutoff (floats: [low_percentile, high_percentile], values in [0-100])
    #     cutoff_times_std : Cutoff in terms of standard deviation (floats: [low_multiple, high_multiple])
    #     cutoff_below_mean: Low cutoff of whole image mean (True or False)
    # Returns: Normalized image and a string that can be logged/printed with info on cutoffs used.
    # get_stats_info: If True, also computes and returns info on statistics. Extra compute.
    
    old_mean = None
    old_std = None
    log_info = "For computing mean/std for normalizing, disregarded voxels according to following rules:"
    img_roi = img[roi_mask_bool]  # This gets flattened automatically. It's a vector array.
    mask_bool_norm = roi_mask_bool.copy() # Needed, so that below do not change it for other channels
    log_info += "\n\t Cutoff outside ROI."
    
    if cutoff_percents is not None:
        cutoff_low = np.percentile(img_roi, cutoff_percents[0])
        cutoff_high = np.percentile(img_roi, cutoff_percents[1])
        mask_bool_norm *= get_cutoff_mask(img, cutoff_low, cutoff_high)
        log_info += "\n\t Cutoff ints outside " + str(cutoff_percents) + " 'percentiles' (within ROI)." +\
                    " Cutoffs: Low={0:.2f}".format(cutoff_low) + ", Max={0:.2f}".format(cutoff_high)

    if cutoff_times_std is not None:
        img_roi_mean, img_roi_std, _ = get_img_stats(img_roi, calc_max=False)
        old_mean = img_roi_mean
        old_std = img_roi_std
        cutoff_low = img_roi_mean - cutoff_times_std[0] * img_roi_std
        cutoff_high = img_roi_mean + cutoff_times_std[1] * img_roi_std
        mask_bool_norm *= get_cutoff_mask(img, cutoff_low, cutoff_high)
        log_info += "\n\t Cutoff ints below/above " + str(cutoff_times_std) +\
                    " times the 'std' from the 'mean' (within ROI)." +\
                    " Cutoffs: Low={0:.2f}".format(cutoff_low) + ", High={0:.2f}".format(cutoff_high)

    if cutoff_below_mean: # Avoid if not asked, to save compute.
        img_mean, _, img_max = get_img_stats(img, calc_std=False)
        cutoff_low = img_mean
        cutoff_high = img_max
        mask_bool_norm *= get_cutoff_mask(img, cutoff_low, cutoff_high)  # no high cutoff
        log_info += "\n\t Cutoff ints below mean of *original* img (cuts air in brain MRI)." +\
                    " Cutoff: Low={0:.2f}".format(cutoff_low)

    norm_mean, norm_std, _ = get_img_stats(img[mask_bool_norm], calc_max=False)

    # Normalize
    normalized_img = (img - norm_mean) / (1.0 * norm_std)
    
    # Report
    if get_stats_info:
        if old_mean is None:# May have been computed as part of above calc.
            old_mean = np.mean(img_roi)
        if old_std is None:
            old_std = np.std(img_roi)
        new_mean, new_std, _ = get_img_stats(normalized_img[roi_mask_bool], calc_max=False)
        
        log_info += "\n\t Stats (mean/std within ROI): " +\
                    "Original [{0:.2f}".format(old_mean) + "/{0:.2f}".format(old_std) + "], " +\
                    "Normalized using [{0:.2f}".format(norm_mean) + "/{0:.2f}".format(norm_std) + "], " +\
                    "Final [{0:.2f}".format(new_mean) + "/{0:.2f}".format(new_std) +"]"
        
    return normalized_img, log_info


# Main z-score method.
def normalize_zscore_subj(log, channels, roi_mask, prms, verbose_lvl=0, job_id='', in_place=True):
    # channels: array [n_channels, x, y, z]
    # roi_mask: array [x,y,z]
    # norm_params: dictionary with following key:value entries
    #     'apply_to_all_channels': True/False -> Whether to perform z-score normalization.
    #     'apply_per_channel': [Booleans] -> List of len(channels) booleans, whether to normalize each channel.
    #     'cutoff_percents', 'cutoff_times_std', 'cutoff_below_mean': see called func normalize_zscore_img()
    #     NOTE: If apply_to_all_channels: True, then REQUIRES that apply_per_channel: None
    #     E.g. BRATS: cutoff_perc: [5., 95.], cutoff_times_std: [2., 2.], cutoff_below_mean: True
    # verbose_lvl: 0: no logging, 1: Timing, 2: Stats per channel
    # job_id: string for logging, specifying job number and pid. In testing, "".
    assert not (prms['apply_to_all_channels'] and prms['apply_per_channel'] is not None)
    assert (prms['apply_per_channel'] is None or isinstance(prms['apply_per_channel'], list))
    
    list_bools_apply_per_c = None

    if prms is None:
        return channels, False
    elif prms['apply_to_all_channels']:
        list_bools_apply_per_c = [True] * len(channels)
    elif prms['apply_per_channel'] is None:
        return channels, False
    elif isinstance(prms['apply_per_channel'], list):
        assert len(prms['apply_per_channel']) == len(channels)
        list_bools_apply_per_c = prms['apply_per_channel']
    else:
        raise ValueError("Unexpected value for parameter in normalize_zscore_subj()")
    
    channels_norm = channels if in_place else np.zeros(channels.shape)
    roi_mask_bool = roi_mask > 0 if roi_mask is not None else np.ones(channels[0].shape) > 0
    applied = False
    
    for idx, channel in enumerate(channels):
        if not list_bools_apply_per_c[idx]:
            continue

        channels_norm[idx], log_info = normalize_zscore_img(channel, roi_mask_bool,
                                                            prms['cutoff_percents'],
                                                            prms['cutoff_times_std'],
                                                            prms['cutoff_below_mean'],
                                                            verbose_lvl>=2)
        applied = True
        
        if verbose_lvl >=2:
            log.print3(job_id + " Z-Score Normalization of Channel-" + str(idx) + ":\n\t" + log_info)
    
    return channels_norm, applied

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

