# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import os
import sys
import time
import numpy as np
import math
import random
import traceback
import multiprocessing
import signal
import collections

from deepmedic.dataManagement.io import load_volume
from deepmedic.neuralnet.pathwayTypes import PathwayTypes as pt
from deepmedic.dataManagement.preprocessing import pad_imgs_of_case, normalize_int_of_subj, calc_border_int_of_3d_img
from deepmedic.dataManagement.augmentSample import augment_sample
from deepmedic.dataManagement.augmentImage import augment_imgs_of_case


# Order of calls:
# get_samples_for_subepoch
#    choose_random_subjects
#    get_n_samples_per_subj
#    load_subj_and_sample
#        load_imgs_of_subject
#        sample_idxs_of_segments
#        extractSegmentGivenSliceCoords
#            get_subsampl_segment
#    shuffle_samples


# Main sampling process during training. Executed in parallel while training on a batch on GPU.
# Called from training.do_training()
# TODO: I think this should be a "sampler" class and moved to training.py. To keep this file generic-sampling.
def get_samples_for_subepoch(log,
                             train_val_or_test,
                             num_parallel_proc,
                             run_input_checks,
                             cnn3d,
                             max_n_cases_per_subep,
                             n_samples_per_subep,
                             sampling_type,
                             inp_shapes_per_path,
                             outp_pred_dims,
                             unpred_margin,
                             # Paths to input files
                             paths_per_chan_per_subj,
                             paths_to_lbls_per_subj,
                             paths_to_masks_per_subj,
                             paths_to_wmaps_per_sampl_cat_per_subj,
                             # Preprocessing & Augmentation
                             pad_input_imgs,
                             norm_prms,
                             augm_img_prms,
                             augm_sample_prms):
    # train_val_or_test: 'train', 'val' or 'test'
    # Returns: channs_of_samples_arr_per_path - List of arrays [N_samples, Channs, R,C,Z], one per pathway.
    #          lbls_predicted_part_of_samples_arr - Array of shape: [N_samples, R_out, C_out, Z_out)
    
    sampler_id = "[TRA|SAMPLER|PID:" + str(os.getpid()) + "]" if train_val_or_test == "train" \
            else "[VAL|SAMPLER|PID:" + str(os.getpid()) + "]"
    start_time_sampling = time.time()
    tr_or_val_str_log = "Training" if train_val_or_test == "train" else "Validation"

    log.print3(sampler_id +
               " :=:=:=:=:=:=: Starting to sample for next [" + tr_or_val_str_log + "]... :=:=:=:=:=:=:")

    n_total_subjects = len(paths_per_chan_per_subj)
    idxs_of_subjs_for_subep = choose_random_subjects(n_total_subjects, max_n_cases_per_subep)

    log.print3(sampler_id + " Out of [" + str(n_total_subjects) + "] subjects given for [" +
               tr_or_val_str_log + "], we will sample from maximum [" + str(max_n_cases_per_subep) +
               "] per subepoch.")
    log.print3(sampler_id + " Shuffled indices of subjects that were randomly chosen: " + str(idxs_of_subjs_for_subep))

    # List, with [numberOfPathwaysThatTakeInput] sublists.
    # Each sublist is list of [partImagesLoadedPerSubepoch] arrays [channels, R,C,Z].
    channs_of_samples_per_path = [[] for i in range(cnn3d.getNumPathwaysThatRequireInput())]
    lbls_predicted_part_of_samples = []  # Labels only for the central/predicted part of segments.
    # Can be different than max_n_cases_per_subep, because of available images number.
    n_subjs_for_subep = len(idxs_of_subjs_for_subep)

    # Get how many samples I should get from each subject.
    n_samples_per_subj = get_n_samples_per_subj(n_samples_per_subep, n_subjs_for_subep)

    args_sampling_job = [log,
                         train_val_or_test,
                         run_input_checks,
                         cnn3d,
                         sampling_type,
                         paths_per_chan_per_subj,
                         paths_to_lbls_per_subj,
                         paths_to_masks_per_subj,
                         paths_to_wmaps_per_sampl_cat_per_subj,
                         # Pre-processing:
                         pad_input_imgs,
                         norm_prms,
                         augm_img_prms,
                         augm_sample_prms,

                         n_subjs_for_subep,
                         idxs_of_subjs_for_subep,
                         n_samples_per_subj,
                         inp_shapes_per_path,
                         outp_pred_dims,
                         unpred_margin
                         ]

    log.print3(sampler_id + " Will sample from [" + str(n_subjs_for_subep) +
               "] subjects for next " + tr_or_val_str_log + "...")

    jobs_idxs_to_do = list(range(n_subjs_for_subep))  # One job per subject.

    if num_parallel_proc <= 0:  # Sequentially
        for job_idx in jobs_idxs_to_do:
            (channs_samples_from_job_per_path,
             lbls_predicted_part_samples_from_job) = load_subj_and_sample(*([job_idx] + args_sampling_job))
            for pathway_i in range(cnn3d.getNumPathwaysThatRequireInput()):
                # concat does not copy.
                channs_of_samples_per_path[pathway_i] += channs_samples_from_job_per_path[pathway_i]
            lbls_predicted_part_of_samples += lbls_predicted_part_samples_from_job  # concat does not copy.

    else:  # Parallelize sampling from each subject
        while len(jobs_idxs_to_do) > 0:  # While jobs remain.
            jobs = collections.OrderedDict()

            log.print3(sampler_id + " ******* Spawning children processes to sample from [" +
                       str(len(jobs_idxs_to_do)) + "] subjects*******")
            log.print3(sampler_id + " MULTIPR: Number of CPUs detected: " + str(multiprocessing.cpu_count()) +
                       ". Requested to use max: [" + str(num_parallel_proc) + "]")
            n_workers = min(num_parallel_proc, multiprocessing.cpu_count())
            log.print3(sampler_id + " MULTIPR: Spawning [" + str(n_workers) + "] processes to load and sample.")
            mp_pool = multiprocessing.Pool(processes=n_workers, initializer=init_sampling_proc)

            try:  # Stacktrace in MULTIPR: https://jichu4n.com/posts/python-multiprocessing-and-exceptions/
                for job_idx in jobs_idxs_to_do:  # submit jobs
                    jobs[job_idx] = mp_pool.apply_async(load_subj_and_sample, ([job_idx] + args_sampling_job))

                # copy with list(...), so that this loops normally even if something is removed from list.
                for job_idx in list(jobs_idxs_to_do):
                    try:
                        # timeout in case process for some reason never started (happens in py3)
                        (channs_samples_from_job_per_path,
                         lbls_predicted_part_samples_from_job) = jobs[job_idx].get(timeout=60)
                        for pathway_i in range(cnn3d.getNumPathwaysThatRequireInput()):
                            # concat does not copy.
                            channs_of_samples_per_path[pathway_i] += channs_samples_from_job_per_path[pathway_i]
                        # concat does not copy.
                        lbls_predicted_part_of_samples += lbls_predicted_part_samples_from_job
                        jobs_idxs_to_do.remove(job_idx)
                    except multiprocessing.TimeoutError:
                        log.print3(sampler_id +\
                              "\n\n WARN: MULTIPR: Caught TimeoutError when getting results of job [" +
                              str(job_idx) + "].\n WARN: MULTIPR: Will resubmit job [" + str(job_idx) + "].\n")
                        if n_workers == 1:
                            break  # If this worker got stuck, every job will wait timeout. Slow. Recreate pool.
                    except Exception as e:
                        log.print3(sampler_id + "\n\n ERROR: Caught exception from job [" + str(job_idx) + "].")
                        raise e

            except (Exception, KeyboardInterrupt) as e:
                log.print3(
                    sampler_id + "\n\n ERROR: Caught exception in get_samples_for_subepoch(): " + str(e) + "\n")
                log.print3(traceback.format_exc())
                mp_pool.terminate()
                mp_pool.join()  # Will wait. A KeybInt will kill this (py3)
                raise e
            except:  # Catches everything, even a sys.exit(1) exception.
                log.print3(sampler_id + "\n\n ERROR: Unexpected error in get_samples_for_subepoch(). " +\
                           "System info: ", sys.exc_info()[0])
                mp_pool.terminate()
                mp_pool.join()
                raise Exception("Unexpected error.")
            else:  # Nothing went wrong
                # Needed in case any processes are hanging. mp_pool.close() does not solve this.
                mp_pool.terminate()
                mp_pool.join()

    # Got all samples for subepoch. Now shuffle them, together segments and their labels.
    (channs_of_samples_per_path,
     lbls_predicted_part_of_samples) = shuffle_samples(channs_of_samples_per_path, lbls_predicted_part_of_samples)
    log.print3(sampler_id + " TIMING: Sampling for next [" + tr_or_val_str_log +
               "] lasted: {0:.1f}".format(time.time() - start_time_sampling) + " secs.")

    log.print3(sampler_id + " :=:=:=:=:=:= Finished sampling for next [" + tr_or_val_str_log + "] =:=:=:=:=:=:")

    channs_of_samples_arr_per_path = [np.asarray(channs_of_samples_for_path, dtype="float32") for
                                      channs_of_samples_for_path in channs_of_samples_per_path]

    lbls_predicted_part_of_samples_arr = np.asarray(lbls_predicted_part_of_samples, dtype="int32")

    return channs_of_samples_arr_per_path, lbls_predicted_part_of_samples_arr


def init_sampling_proc():
    # This will make child-processes ignore the KeyboardInterupt (sigInt). Parent will handle it.
    # See: http://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python/35134329#35134329
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def choose_random_subjects(n_total_subjects,
                           max_subjects_on_gpu_for_subepoch,
                           get_max_subjects_for_gpu_even_if_total_less=False):
    # Returns: list of indices
    subjects_indices = list(range(n_total_subjects))  # list() for python3, cause shuffle() cant get range
    random_order_chosen_subjects = []
    random.shuffle(subjects_indices)  # does it in place. Now they are shuffled

    if max_subjects_on_gpu_for_subepoch >= n_total_subjects:
        random_order_chosen_subjects += subjects_indices

        # This is if I want to have a certain amount on GPU, even if total subjects are less.
        if get_max_subjects_for_gpu_even_if_total_less:
            while len(random_order_chosen_subjects) < max_subjects_on_gpu_for_subepoch:
                random.shuffle(subjects_indices)
                number_of_extra_subjects_to_get_to_fill_gpu = min(
                    max_subjects_on_gpu_for_subepoch - len(random_order_chosen_subjects), n_total_subjects)
                random_order_chosen_subjects += (subjects_indices[:number_of_extra_subjects_to_get_to_fill_gpu])
            assert len(random_order_chosen_subjects) == max_subjects_on_gpu_for_subepoch
            
    else:
        random_order_chosen_subjects += subjects_indices[:max_subjects_on_gpu_for_subepoch]

    return random_order_chosen_subjects


def get_n_samples_per_subj(n_samples, n_subjects):
    # Distribute samples of each cat to subjects.
    n_samples_per_subj = np.ones([n_subjects], dtype='int32') * (n_samples // n_subjects)
    n_undistributed_samples = n_samples % n_subjects
    # Distribute samples that were left by inexact division.
    for idx in range(n_undistributed_samples):
        n_samples_per_subj[random.randint(0, n_subjects - 1)] += 1
    return n_samples_per_subj


def constrain_sampling_maps_near_edges(sample_maps_per_cat, dims_sample):
    # We wish to sample pixels around which the samples we will extract will be centered.
    # But we need to be CAREFUL and get only pixels that are NOT closer to the image boundaries than the dimensions of the
    # samples we wish to extract permit.
    constrained_maps = []
    mask_excl_edges = comp_valid_sampling_mask_excluding_edges(dims_sample, sample_maps_per_cat[0].shape)
    for cat_i in range(len(sample_maps_per_cat)):
        sampling_map = sample_maps_per_cat[cat_i]
        sampling_map_excl_near_edges = np.multiply(sampling_map, mask_excl_edges, dtype=sampling_map.dtype)
        constrained_maps.append(sampling_map_excl_near_edges)
    return constrained_maps


def load_subj_and_sample(job_idx,
                         log,
                         train_val_or_test,
                         run_input_checks,
                         cnn3d,
                         sampling_type,
                         paths_per_chan_per_subj,
                         paths_to_lbls_per_subj,
                         paths_to_masks_per_subj,
                         paths_to_wmaps_per_sampl_cat_per_subj,
                         # Pre-processing:
                         pad_input_imgs,
                         norm_prms,
                         augm_img_prms,
                         augm_sample_prms,
                         n_subjs_for_subep,
                         idxs_of_subjs_for_subep,
                         n_samples_per_subj,
                         inp_shapes_per_path,
                         outp_pred_dims,
                         unpred_margin):
    # train_val_or_test: 'train', 'val' or 'test'
    # paths_per_chan_per_subj: [[ for chan-0 [ one path per subj ]], ..., [for chan-n  [ one path per subj ] ]]
    # n_samples_per_cat_per_subj: np arr, shape [num sampling categories, num subjects in subepoch]
    # returns: ( channs_of_samples_per_path, lbls_predicted_part_of_samples )
    job_id = "[TRA|JOB:" + str(job_idx) + "|PID:" + str(os.getpid()) + "]" if train_val_or_test == 'train' \
        else "[VAL|JOB:" + str(job_idx) + "|PID:" + str(os.getpid()) + "]"
    
    log.print3(job_id + " Started. (#" + str(job_idx) + "/" + str(n_subjs_for_subep) + ") sampling job. " +
               "Load & sample from subject of index (in user's list): " + str(idxs_of_subjs_for_subep[job_idx]) )

    # List, with [numberOfPathwaysThatTakeInput] sublists.
    # Each sublist is list of [partImagesLoadedPerSubepoch] arrays [channels, R,C,Z].
    channs_of_samples_per_path = [[] for i in range(cnn3d.getNumPathwaysThatRequireInput())]
    lbls_predicted_part_of_samples = []  # Labels only for the central/predicted part of segments.

    dims_hres_segment = inp_shapes_per_path[0]
    
    # Load images of subject
    time_load_0 = time.time()
    (channels,  # nparray [channels,dim0,dim1,dim2]
     gt_lbl_img,
     roi_mask,
     wmaps_to_sample_per_cat) = load_imgs_of_subject(log, job_id,
                                                     idxs_of_subjs_for_subep[job_idx],
                                                     paths_per_chan_per_subj,
                                                     paths_to_lbls_per_subj,
                                                     paths_to_wmaps_per_sampl_cat_per_subj,
                                                     paths_to_masks_per_subj)
     
    # Pre-process images of subject
    time_load = time.time() - time_load_0
    time_prep_0 = time.time()
    (channels,
    gt_lbl_img,
    roi_mask,
    wmaps_to_sample_per_cat,
    pad_left_right_per_axis) = preproc_imgs_of_subj(log, job_id,
                                                    channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat,
                                                    run_input_checks, cnn3d.num_classes, # checks
                                                    pad_input_imgs, unpred_margin,
                                                    norm_prms)
    time_prep = time.time() - time_prep_0
    
    # Augment at image level:
    time_augm_0 = time.time()
    (channels,
     gt_lbl_img,
     roi_mask,
     wmaps_to_sample_per_cat) = augment_imgs_of_case(channels,
                                                     gt_lbl_img,
                                                     roi_mask,
                                                     wmaps_to_sample_per_cat,
                                                     augm_img_prms)
    time_augm_img = time.time() - time_augm_0

    # Sampling of segments (sub-volumes) from an image.
    dims_of_scan = channels[0].shape
    sampling_maps_per_cat = sampling_type.derive_sampling_maps_per_cat(wmaps_to_sample_per_cat,
                                                                       gt_lbl_img,
                                                                       roi_mask,
                                                                       dims_of_scan)
    sampling_maps_per_cat = constrain_sampling_maps_near_edges(sampling_maps_per_cat, dims_hres_segment)
    
    
    # Get number of samples per sampling-category for the specific subject (class, foregr/backgr, etc)
    (n_samples_per_cat, valid_cats) = sampling_type.distribute_n_samples_to_categs(n_samples_per_subj[job_idx],
                                                                                   sampling_maps_per_cat)
    time_sample_idxs = 0
    time_extr_samples = 0
    time_augm_samples = 0
    str_samples_per_cat = " Done. Samples per category: "
    for cat_i in range(sampling_type.get_n_sampling_cats()):
        cat_str = sampling_type.get_sampling_cats_as_str()[cat_i]
        n_samples_for_cat = n_samples_per_cat[cat_i]
        sampling_map = sampling_maps_per_cat[cat_i]
        # Check if the class is valid for sampling.
        # Invalid if eg there is no such class in the subject's manual segmentation.
        if not valid_cats[cat_i]:
            log.print3( job_id + " WARN: Invalid sampling category! Sampling map just zeros! No [" + cat_str +
                        "] samples from this subject!")
            assert n_samples_for_cat == 0
            continue # This should not be needed, the next func should also handle it. But whatever.
            
        time_sample_idx0 = time.time()
        idxs_sampl_centers = sample_idxs_of_segments(log,
                                                     job_id,
                                                     n_samples_for_cat,
                                                     sampling_map)
        time_sample_idxs += time.time() - time_sample_idx0
        str_samples_per_cat += "[" + cat_str + ": " + str(len(idxs_sampl_centers[0])) + "/" + str(n_samples_for_cat) + "] "
        
        # Use the just sampled coordinates of slices to actually extract the segments (data) from the subject's images.
        for image_part_i in range(len(idxs_sampl_centers[0])):
            coord_center = idxs_sampl_centers[:, image_part_i]
            
            time_extr_sample_0 = time.time()
            (channs_of_sample_per_path,
             lbls_predicted_part_of_sample) = extractSegmentGivenSliceCoords(train_val_or_test,
                                                                             cnn3d,
                                                                             coord_center,
                                                                             channels,
                                                                             gt_lbl_img,
                                                                             inp_shapes_per_path,
                                                                             outp_pred_dims)
            time_extr_samples += time.time() - time_extr_sample_0
            
            # Augmentation of segments
            time_augm_sample_0 = time.time()
            (channs_of_sample_per_path,
             lbls_predicted_part_of_sample) = augment_sample(channs_of_sample_per_path,
                                                             lbls_predicted_part_of_sample,
                                                             augm_sample_prms)
            time_augm_samples += time.time() - time_augm_sample_0
            
            for pathway_i in range(cnn3d.getNumPathwaysThatRequireInput()):
                channs_of_samples_per_path[pathway_i].append(channs_of_sample_per_path[pathway_i])
            lbls_predicted_part_of_samples.append(lbls_predicted_part_of_sample)
        
    log.print3(job_id + str_samples_per_cat)
    log.print3(job_id + " TIMING: " +
               "[Load: {0:.1f}".format(time_load) + "] "
               "[Preproc: {0:.1f}".format(time_prep) + "] " +
               "[Augm-Img: {0:.1f}".format(time_augm_img) + "] " +
               "[Sample Coords: {0:.1f}".format(time_sample_idxs) + "] " +
               "[Extract Sampl: {0:.1f}".format(time_extr_samples) + "] " +
               "[Augm-Samples: {0:.1f}".format(time_augm_samples) + "] secs")
    return (channs_of_samples_per_path, lbls_predicted_part_of_samples)


# roi_mask_filename and roiMinusLesion_mask_filename can be passed "no".
# In this case, the corresponding return result is nothing.
# This is so because: the do_training() function only needs the roiMinusLesion_mask,
# whereas the do_testing() only needs the roi_mask.
def load_imgs_of_subject(log,
                         job_id,
                         subj_i,
                         paths_per_chan_per_subj,
                         paths_to_lbls_per_subj,
                         paths_to_wmaps_per_sampl_cat_per_subj,
                         paths_to_masks_per_subj
                         ):
    # paths_per_chan_per_subj: None or List of lists. One sublist per case. Each should contain...
    # ... as many elements(strings-filenamePaths) as numberOfChannels, pointing to (nii) channels of this case.
    
    log.print3(job_id + " Loading subject with 1st channel at: " + str(paths_per_chan_per_subj[subj_i][0]))
    
    n_channels = len(paths_per_chan_per_subj[0])
        
    # Load the channels of the patient.
    inp_chan_dims = None  # Dimensions of the (padded) input channels.
    channels = None
    for channel_i in range(n_channels):
        path_to_chan = paths_per_chan_per_subj[subj_i][channel_i]
        if path_to_chan != "-":  # normal case, filepath was given.
            channel = load_volume(path_to_chan)
            
            if channels is None:
                # Initialize the array in which all the channels for the patient will be placed.
                inp_chan_dims = list(channel.shape)
                channels = np.zeros((n_channels, inp_chan_dims[0], inp_chan_dims[1], inp_chan_dims[2]))

            channels[channel_i] = channel
        else:  # "-" was given in the config-listing file. Do Min-fill!
            log.print3(job_id + " WARN: No modality #" + str(channel_i) + " given. Will make zero-filled channel.")
            channels[channel_i] = 0.0
    
    # Load the class labels.
    if paths_to_lbls_per_subj is not None:
        path_to_lbl = paths_to_lbls_per_subj[subj_i]
        gt_lbl_img = load_volume(path_to_lbl)

        if gt_lbl_img.dtype.kind not in ['i', 'u']:
            dtype_gt_lbls = 'int16'
            log.print3(job_id + " WARN: Loaded labels are dtype [" + str(gt_lbl_img.dtype) + "]."
                       " Rounding and casting to [" + dtype_gt_lbls + "]!")
            gt_lbl_img = np.rint(gt_lbl_img).astype(dtype_gt_lbls)
    else:
        gt_lbl_img = None  # For validation and testing

    if paths_to_masks_per_subj is not None:
        path_to_roi_mask = paths_to_masks_per_subj[subj_i]
        roi_mask = load_volume(path_to_roi_mask)
        
        if roi_mask.dtype.kind not in ['i','u']:
            dtype_roi_mask = 'int16'
            log.print3(job_id + " WARN: Loaded ROI-mask is dtype [" + str(roi_mask.dtype) + "]."
                       " Rounding and casting to [" + dtype_roi_mask + "]!")
            roi_mask = np.rint(roi_mask).astype(dtype_roi_mask)    
    else:
        roi_mask = None
        
    # May be provided only for training.
    if paths_to_wmaps_per_sampl_cat_per_subj is not None:
        n_sampl_categs = len(paths_to_wmaps_per_sampl_cat_per_subj)
        wmaps_to_sample_per_cat = np.zeros([n_sampl_categs] + list(channels[0].shape), dtype="float32")
        for cat_i in range(n_sampl_categs):
            path_to_wmaps_for_this_cat_per_subj = paths_to_wmaps_per_sampl_cat_per_subj[cat_i]
            path_to_wmap_for_this_cat = path_to_wmaps_for_this_cat_per_subj[subj_i]
            wmap_for_this_cat = load_volume(path_to_wmap_for_this_cat)
            if not np.all(wmap_for_this_cat >= 0):
                raise ValueError("Negative values found in weightmap. Unexpected. Zero or positives allowed.")
            wmaps_to_sample_per_cat[cat_i] = wmap_for_this_cat
    else:
        wmaps_to_sample_per_cat = None

    return channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat


def preproc_imgs_of_subj(log, job_id, channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat,
                         run_input_checks, n_classes, pad_input_imgs, unpred_margin, norm_prms):
    # job_id: Should be "" in testing.
    
    if run_input_checks:
        check_gt_vs_num_classes(log, job_id, gt_lbl_img, n_classes)
    
    (channels,
     gt_lbl_img,
     roi_mask,
     wmaps_to_sample_per_cat,
     pad_left_right_per_axis) = pad_imgs_of_case(channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat,
                                                 pad_input_imgs, unpred_margin)
    
    channels = normalize_int_of_subj(log, channels, roi_mask, norm_prms, job_id)
    
    return channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat, pad_left_right_per_axis


def comp_valid_sampling_mask_excluding_edges(dims_of_segment, shape):
    # I look for lesions that are not closer to the image boundaries than the ImagePart dimensions allow.
    # KernelDim is always odd. BUT ImagePart dimensions can be odd or even.
    # If odd, ok, floor(dim/2) from central.
    # If even, dim/2-1 voxels towards the begining of the axis and dim/2 towards the end.
    # Ie, "central" imagePart voxel is 1 closer to begining.
    # BTW imagePartDim takes kernel into account (ie if I want 9^3 voxels classified per imagePart with kernel 5x5,
    # I want 13 dim ImagePart)
    
    # number of voxels to exclude from edges of the image, left and right in each axis, when sampling...
    # ...the center of a segment. So that the segment will be fully contained in the image. (half segm left & right)
    # dim1: 1 row per r,c,z. Dim2: left/right width not to sample from (=half segment).
    n_vox_excl_left_right = np.zeros((len(dims_of_segment), 2), dtype='int16')

    # The below starts all zero. Will be Multiplied by other true-false arrays expressing if the relevant
    # voxels are within boundaries.
    # In the end, the final vector will be true only for the indices of lesions that are within all boundaries.
    mask_excl_near_edges = np.zeros(shape, dtype='int8')

    # This loop leads to mask_excl_near_edges to be true for the indices ...
    # ...that allow getting an imagePart CENTERED on them, and be safely within image boundaries. Note that ...
    # ... if the imagePart is of even dimension, the "central" voxel is one voxel to the left.
    for rcz_i in range(len(dims_of_segment)):
        if dims_of_segment[rcz_i] % 2 == 0:  # even
            dims_div_2 = dims_of_segment[rcz_i] // 2
            # central of ImagePart is 1 vox closer to begining of axes.
            n_vox_excl_left_right[rcz_i] = [dims_div_2 - 1, dims_div_2]
        else:  # odd
            # If odd, middle voxel is the "central". Eg 5/2 = 2, with 3rd voxel being the central.
            dims_div_2_floor = math.floor(dims_of_segment[rcz_i] // 2)
            n_vox_excl_left_right[rcz_i] = [dims_div_2_floor, dims_div_2_floor]
            # used to be [n_vox_excl_left_right[0][0]: -n_vox_excl_left_right[0][1]],
            # but in 2D case n_vox_excl_left_right might be ==0, causes problem and you get a null slice.
    mask_excl_near_edges[
        n_vox_excl_left_right[0][0]: shape[0] - n_vox_excl_left_right[0][1],
        n_vox_excl_left_right[1][0]: shape[1] - n_vox_excl_left_right[1][1],
        n_vox_excl_left_right[2][0]: shape[2] - n_vox_excl_left_right[2][1]] = 1
        
    return mask_excl_near_edges
        
def sampling_cumsum(p_1darr, n_samples):
    p_cumsum = p_1darr.cumsum(dtype='float64') # This is dangerous, final elements go below or beyond 1.
    p_cumsum = np.clip(p_cumsum, a_min=None, a_max=1., out=p_cumsum)
    #p_cumsum[p_cumsum>1.] = 1.
    p_cumsum[-1] = 1.
    random_ps = np.random.random(size=n_samples)
    idxs_sampled = np.searchsorted(p_cumsum, random_ps)
    return idxs_sampled

def sample_with_appropriate_algorithm(n_samples, sampling_map, sum_sampl_map):
    
    p_sampling_flat = sampling_map.ravel() # Unnormalized
    is_0 = p_sampling_flat == 0. # np.isclose(p_sampling_flat, 0.)
    is_1 = p_sampling_flat == 1. # np.isclose(p_sampling_flat, 1.)
    if np.all(np.logical_or(is_0, is_1)): # Whole sampling map is either 0 or 1. Do faster sampling only under valid idxs.
        idxs = np.arange(p_sampling_flat.size, dtype='int32') # Unlikely large image size will need int64
        valid_idxs = idxs[is_1>0]
        idxs_sampled = np.random.choice(valid_idxs, size=n_samples, replace=True)
    else: #if np.any( np.logical_and(sampling_map != 0., sampling_map != 1.)):
        # Normalize, because sampling method np.random.choice requires it.
        p_sampling_normed = np.multiply(p_sampling_flat, 1./sum_sampl_map, dtype='float64') # as before.
        #idxs_sampled = sampling_cumsum(p_sampling_normed, n_samples)
        idxs_sampled = np.random.choice(p_sampling_normed.size, size=n_samples, replace=True, p=p_sampling_normed)
    
    # np.unravel_index([listOfIndicesInFlattened], dims) returns a tuple of arrays (eg 3 of them if 3 dimImage), 
    # where each of the array in the tuple has the same shape as the listOfIndices. 
    # They have the r/c/z coords that correspond to the index of the flattened version.
    # So, idxs_sampled will be array of shape: 3(rcz) x n_samples.
    idxs_sampled = np.asarray(np.unravel_index(idxs_sampled, sampling_map.shape))
    
    return idxs_sampled

# made for 3d
def sample_idxs_of_segments(log,
                            job_id,
                            n_samples,
                            sampling_map):
    """
    sampling_map: np.array of shape (H,W,D), dtype="int16" or potentially float if weightmaps given by user.
    Returns: idxs_of_sampled_centers
             Coordinates (xyz indices) of the "central" voxel of sampled segments (1 voxel to the left if dimension is even).
    
    > idxs_of_sampled_centers: array with shape: 3(xyz) x n_samples.
        Example: [ xCoordsForCentralVoxelOfEachPart, yCoordsForCentralVoxelOfEachPart, zCoordsForCentralVoxelOfEachPart ]
        >> x/y/z-CoordsForCentralVoxelOfEachPart: 1-dim array with n_samples, holding the x-indices of samples in image.
    """
    # Check if the weight map is fully-zeros. In this case, return no element.
    # Note: Currently, the caller function is checking this case already and does not let this being called.
    # Which is still fine.
    sum_sampl_map = np.sum(sampling_map, dtype='float64')
    if np.isclose(sum_sampl_map, 0.) : # is zero
        log.print3(job_id + " WARN: Sampling map for category (after excluding near edges) is just zeros! " +\
                   " No samples for category from subject!")
        return [ [[],[],[]], [[],[],[]] ]
    
    # Sample indexes of pixels around which we should extract sample segments.
    idxs_of_sampled_centers = sample_with_appropriate_algorithm(n_samples, sampling_map, sum_sampl_map)
    
    return idxs_of_sampled_centers


def get_subsampl_segment(rec_field_hr_path, channels, segment_hr_slice_coords, subs_factor, dims_lr_segm):
    """
    Given the segment_hr_slice_coords, which has the coordinates where the high-resolution image segment starts and
    ends (inclusive), this returns the corresponding image segment of down-sampled context for the parallel path(s).

    (Actually, in this implementation, the right (end) part of segment_hr_slice_coords is not used.)
    The way it works is NOT optimal. It ASSUMES that receptive field or high res and low-res paths are the same size.
    From the beginning of the high-resolution segment,
    it goes further to the left 1 receptive-field and then forward subs_factor * receptive-fields.
    This stops it from being used with arbitrary size of subsampled segment (decoupled by the high-res segment).
    Now, the subsampled receptive-field has to be of the same size as the normal-scale.
    To change this, I should find where THE FIRST TOP LEFT CENTRAL (predicted) VOXEL is, <=================
    and do the back-one-(sub)rec_field + front-3-(sub)ref_fields from there, not from the beginning of the patch.

    Current way it works (correct):
    If I have eg subsample factor=3 and 9 central-pred-voxels, I get 3 "central" voxels for the
    subsampled-segment. Straightforward. If I have a number of central voxels that is not an exact multiple of
    the subfactor, eg 10 central-voxels, I get 3+1 central voxels in the subsampled-segment.
    When the cnn is convolving them, they will get repeated to 4(last-layer-neurons)*3(factor) = 12,
    and will get sliced down to 10, in order to have same dimension with the 1st pathway.
    """
    dims_hr_segm = [segment_hr_slice_coords[d][1] - segment_hr_slice_coords[d][0] + 1 for d in range(3)]
    dims_outp_hr_path = [dims_hr_segm[d] - rec_field_hr_path[d] + 1 for d in range(3)] # Assumes no stride.
    dims_img = channels.shape[1:] # Channels: [batch, X, Y, Z]

    segment_lr = np.ones((channels.shape[0], dims_lr_segm[0], dims_lr_segm[1], dims_lr_segm[2]), dtype='float32')

    n_vox_before = [None, None, None]
    centre_of_downsampl_kernel = [None, None, None]
    for d in range(3):
        if subs_factor[d] % 2 == 1: # Odd
            n_vox_before[d] = ((subs_factor[d] - 1) // 2) * rec_field_hr_path[d] # TODO: Should be rec_field_lr_path
            centre_of_downsampl_kernel[d] = subs_factor[d] // 2 # 3 ==> 1
        else:
            n_vox_before[d] = (subs_factor[d] - 2) // 2 * rec_field_hr_path[d] + rec_field_hr_path[d] // 2
            centre_of_downsampl_kernel[d] = subs_factor[d] // 2 - 1 # One pixel closer to the beginning of dim.

    # This is where to start taking voxels from the subsampled image: From the beginning of the hr_segment...
    # ... go forward a few steps to the voxel that is like the "central" in this subsampled (eg 3x3) area.
    # ...Then go backwards -Patchsize to find the first voxel of the subsampled.

    # These indices can run out of image boundaries. I ll correct them afterwards.
    low = [segment_hr_slice_coords[d][0] + centre_of_downsampl_kernel[d] - n_vox_before[d] for d in range(3)]
    # If the rec_field is 17x17, I want a 17x17 subsampled Patch. BUT if the segment is 25x25 (9voxClass),
    # I want 3 voxels in my subsampled-segment to cover this area!
    # That is what the last term below is taking care of.
    high_non_incl = [int(low[d] + subs_factor[d] * rec_field_hr_path[d] + (
            math.ceil(dims_outp_hr_path[d] / subs_factor[d]) - 1) * subs_factor[d]) for d in range(3)]

    low_corrected = [max(low[d], 0) for d in range(3)]
    high_non_incl_corrected = [min(high_non_incl[d], dims_img[d]) for d in range(3)]

    low_to_put_slice_in_segm = [0 if low[d] >= 0 else abs(low[d]) // subs_factor[d] for d in range(3)]
    dims_of_slice_not_padded = [int(math.ceil((high_non_incl_corrected[d] - low_corrected[d]) * 1.0 / subs_factor[d])) for d in range(3)]
    # dims_of_slice_not_padded =! dims_lr_segm when sampling at borders.

    # I now have exactly where to get the slice from and where to put it in the new array.
    for channel_i in range(len(channels)):
        segment_lr[channel_i] *= calc_border_int_of_3d_img(channels[channel_i]) # Make black

        chan_slice_lr = channels[channel_i,
                                 low_corrected[0]: high_non_incl_corrected[0]: subs_factor[0],
                                 low_corrected[1]: high_non_incl_corrected[1]: subs_factor[1],
                                 low_corrected[2]: high_non_incl_corrected[2]: subs_factor[2]]
        segment_lr[channel_i,
                   low_to_put_slice_in_segm[0]: low_to_put_slice_in_segm[0] + dims_of_slice_not_padded[0],
                   low_to_put_slice_in_segm[1]: low_to_put_slice_in_segm[1] + dims_of_slice_not_padded[1],
                   low_to_put_slice_in_segm[2]: low_to_put_slice_in_segm[2] + dims_of_slice_not_padded[2]
                  ] = chan_slice_lr

    return segment_lr


def shuffle_samples(channs_of_samples_per_path, lbls_predicted_part_of_samples):
    n_paths_taking_inp = len(channs_of_samples_per_path)
    inp_to_zip = [sublist_for_path for sublist_for_path in channs_of_samples_per_path]
    inp_to_zip += [lbls_predicted_part_of_samples]

    combined = list(zip(*inp_to_zip))  # list() for python3 compatibility, as range cannot get assignment in shuffle()
    random.shuffle(combined)
    sublists_with_shuffled_samples = list(zip(*combined))

    shuffled_channs_of_samples_per_path = [sublist_for_path for sublist_for_path in
                                           sublists_with_shuffled_samples[:n_paths_taking_inp]]
    shuffled_lbls_predicted_part_of_samples = sublists_with_shuffled_samples[n_paths_taking_inp]

    return (shuffled_channs_of_samples_per_path, shuffled_lbls_predicted_part_of_samples)


# I must merge this with function: extractSegmentsGivenSliceCoords() that is used for Testing! Should be easy!
# This is used in training/val only.
def extractSegmentGivenSliceCoords(train_val_or_test,
                                   cnn3d,
                                   coord_center,
                                   channels,
                                   gt_lbl_img,
                                   inp_shapes_per_path,
                                   outp_pred_dims):
    # channels: numpy array [ n_channels, x, y, z ]
    # coord_center: indeces of the central voxel for the patch to be extracted.

    channs_of_sample_per_path = []
    # Sampling
    for path_idx in range(len(cnn3d.pathways[:1])):  # Hack. The rest of this loop can work for the whole .pathways...
        # ... BUT the loop does not check what happens if boundaries are out of limits, to fill with zeros.
        # This is done in get_subsampl_segment().
        # ... Update it in a nice way to be done here, and then take get_subsampl_segment()
        # out and make loop go for every pathway.

        if cnn3d.pathways[path_idx].pType() == pt.FC:
            continue
        subs_factor = cnn3d.pathways[path_idx].subs_factor()
        pathwayInputShapeRcz = inp_shapes_per_path[path_idx]
        leftBoundaryRcz = [coord_center[d] - subs_factor[d] * (pathwayInputShapeRcz[d] - 1) // 2 for d in range(3)]
        rightBoundaryRcz = [leftBoundaryRcz[d] + subs_factor[d] * pathwayInputShapeRcz[d] - 1 for d in range(3)]

        channelsForThisImagePart = channels[:,
                                            leftBoundaryRcz[0]: rightBoundaryRcz[0] + 1: subs_factor[0],
                                            leftBoundaryRcz[1]: rightBoundaryRcz[1] + 1: subs_factor[1],
                                            leftBoundaryRcz[2]: rightBoundaryRcz[2] + 1: subs_factor[2]]

        channs_of_sample_per_path.append(channelsForThisImagePart)

    # Extract the samples for secondary pathways. This whole for can go away,
    # if I update above code to check to slices out of limits.
    for pathway_i in range(len(cnn3d.pathways)):  # Except Normal 1st, cause that was done already.
        if cnn3d.pathways[pathway_i].pType() == pt.FC or cnn3d.pathways[pathway_i].pType() == pt.NORM:
            continue
        # this datastructure is similar to channelsForThisImagePart, but contains voxels from the subsampled image.
        segment_hr_dims = inp_shapes_per_path[pathway_i]

        slicesCoordsOfSegmForPrimaryPathway = [[leftBoundaryRcz[d], rightBoundaryRcz[d]] for d in range(3)]
        channsForThisSubsampledPartAndPathway = get_subsampl_segment(cnn3d.pathways[0].rec_field()[0],
                                                                     channels,
                                                                     slicesCoordsOfSegmForPrimaryPathway,
                                                                     cnn3d.pathways[pathway_i].subs_factor(),
                                                                     inp_shapes_per_path[pathway_i])

        channs_of_sample_per_path.append(channsForThisSubsampledPartAndPathway)

    # Get ground truth labels for training.
    numOfCentralVoxelsClassifRcz = outp_pred_dims
    leftBoundaryRcz = [coord_center[d] - (numOfCentralVoxelsClassifRcz[d] - 1) // 2 for d in range (3)]
    rightBoundaryRcz = [leftBoundaryRcz[d] + numOfCentralVoxelsClassifRcz[d] for d in range(3)]
    lbls_predicted_part_of_sample = gt_lbl_img[leftBoundaryRcz[0]: rightBoundaryRcz[0],
                                               leftBoundaryRcz[1]: rightBoundaryRcz[1],
                                               leftBoundaryRcz[2]: rightBoundaryRcz[2]]

    # Make COPIES of the segments, instead of having a VIEW (slice) of them.
    # This is so that the the whole volume are afterwards released from RAM.
    channs_of_sample_per_path = \
        [np.array(pathw_channs, copy=True, dtype='float32') for pathw_channs in channs_of_sample_per_path]
    lbls_predicted_part_of_sample = np.copy(lbls_predicted_part_of_sample)

    return channs_of_sample_per_path, lbls_predicted_part_of_sample


# ###########################################################
#
#  Below are functions for testing only.
#  There is duplication with training.
#  They are not the same, but could be merged.
#
# ###########################################################

# TODO: This is very similar to sample_idxs_of_segments() I believe, which is used for training.
#       Consider way to merge them.
def get_slice_coords_of_all_img_tiles(log,
                                      segment_hr_dims, # xyz dims of input to primary pathway (normal)
                                      strideOfSegmentsPerDimInVoxels,
                                      batch_size,
                                      inp_chan_dims,
                                      roi_mask
                                      ):
    # inp_chan_dims: Dimensions of the (padded) input channels. [x, y, z]
    log.print3("Starting to (tile) extract Segments from the images of the subject for Segmentation...")

    sliceCoordsOfSegmentsToReturn = []

    zLowBoundaryNext = 0
    zAxisCentralPartPredicted = False
    while not zAxisCentralPartPredicted:
        zFarBoundary = min(zLowBoundaryNext + segment_hr_dims[2], inp_chan_dims[2])  # Excluding.
        zLowBoundary = zFarBoundary - segment_hr_dims[2]
        zLowBoundaryNext = zLowBoundaryNext + strideOfSegmentsPerDimInVoxels[2]
        zAxisCentralPartPredicted = False if zFarBoundary < inp_chan_dims[2] else True  # IMPORTANT CRITERION

        cLowBoundaryNext = 0
        cAxisCentralPartPredicted = False
        while not cAxisCentralPartPredicted:
            cFarBoundary = min(cLowBoundaryNext + segment_hr_dims[1], inp_chan_dims[1])  # Excluding.
            cLowBoundary = cFarBoundary - segment_hr_dims[1]
            cLowBoundaryNext = cLowBoundaryNext + strideOfSegmentsPerDimInVoxels[1]
            cAxisCentralPartPredicted = False if cFarBoundary < inp_chan_dims[1] else True

            rLowBoundaryNext = 0
            rAxisCentralPartPredicted = False
            while not rAxisCentralPartPredicted:
                rFarBoundary = min(rLowBoundaryNext + segment_hr_dims[0], inp_chan_dims[0])  # Excluding.
                rLowBoundary = rFarBoundary - segment_hr_dims[0]
                rLowBoundaryNext = rLowBoundaryNext + strideOfSegmentsPerDimInVoxels[0]
                rAxisCentralPartPredicted = False if rFarBoundary < inp_chan_dims[0] else True

                # In case I pass a brain-mask, I ll use it to only predict inside it. Otherwise, whole image.
                if isinstance(roi_mask, np.ndarray):
                    if not np.any(roi_mask[rLowBoundary:rFarBoundary,
                                           cLowBoundary:cFarBoundary,
                                           zLowBoundary:zFarBoundary]
                                  ):  # all of it is out of the brain so skip it.
                        continue

                sliceCoordsOfSegmentsToReturn.append([[rLowBoundary, rFarBoundary - 1],
                                                      [cLowBoundary, cFarBoundary - 1],
                                                      [zLowBoundary, zFarBoundary - 1]])

    # I need to have a total number of image-parts that can be exactly-divided by the 'batch_size'.
    # For this reason, I add in the far end of the list multiple copies of the last element.
    total_number_of_image_parts = len(sliceCoordsOfSegmentsToReturn)
    number_of_imageParts_missing_for_exact_division = \
        batch_size - total_number_of_image_parts % batch_size if total_number_of_image_parts % batch_size != 0 else 0
    for extra_useless_image_part_i in range(number_of_imageParts_missing_for_exact_division):
        sliceCoordsOfSegmentsToReturn.append(sliceCoordsOfSegmentsToReturn[-1])

    # I think that since the parts are acquired in a certain order and are sorted this way in the list, it is easy
    # to know which part of the image they came from, as it depends only on the stride-size and the imagePart size.

    log.print3("Finished (tiling) extracting Segments from the images of the subject for Segmentation.")

    # sliceCoordsOfSegmentsToReturn: list with 3 dimensions. numberOfSegments x 3(rcz) x 2
    # (lower and upper limit of the segment, INCLUSIVE both sides)
    return sliceCoordsOfSegmentsToReturn


# I must merge this with function: extractSegmentGivenSliceCoords() that is used for Training/Validation! Should be easy
# This is used in testing only.
def extractSegmentsGivenSliceCoords(cnn3d,
                                    sliceCoordsOfSegmentsToExtract,
                                    channelsOfImageNpArray,
                                    inp_shapes_per_path,
                                    outp_pred_dims):
    # sliceCoordsOfSegmentsToExtract: list of length num_segments. Each elem: [[r_low, r_high], [c_l, c_h], [z_l, z_h]]
    # channelsOfImageNpArray: numpy array [ n_channels, x, y, z ]
    # Returns: list with length [num_pathways], where each element is a list of length [num_segments],
    #          of arrays [channels, r, c, z]
    # TODO: Change result to a list of arrays [num_segments, channs, r, c, z]

    numberOfSegmentsToExtract = len(sliceCoordsOfSegmentsToExtract)
    channsForSegmentsPerPathToReturn = [[] for i in range(cnn3d.getNumPathwaysThatRequireInput())]
    
    for segment_i in range(numberOfSegmentsToExtract):
        rLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][0][0]
        rFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][0][1]
        cLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][1][0]
        cFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][1][1]
        zLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][2][0]
        zFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][2][1]
        # segment for primary pathway
        channsForPrimaryPathForThisSegm = channelsOfImageNpArray[:,
                                                                 rLowBoundary:rFarBoundary + 1,
                                                                 cLowBoundary:cFarBoundary + 1,
                                                                 zLowBoundary:zFarBoundary + 1
                                                                 ]
        channsForSegmentsPerPathToReturn[0].append(channsForPrimaryPathForThisSegm)

        # Subsampled pathways
        for pathway_i in range(len(cnn3d.pathways)):  # Except Normal 1st, cause that was done already.
            if cnn3d.pathways[pathway_i].pType() == pt.FC or cnn3d.pathways[pathway_i].pType() == pt.NORM:
                continue

            channsForThisSubsPathForThisSegm = get_subsampl_segment(cnn3d.pathways[0].rec_field()[0],
                                                                    channelsOfImageNpArray,
                                                                    sliceCoordsOfSegmentsToExtract[segment_i],
                                                                    cnn3d.pathways[pathway_i].subs_factor(),
                                                                    inp_shapes_per_path[pathway_i])

            channsForSegmentsPerPathToReturn[pathway_i].append(channsForThisSubsPathForThisSegm)

    return channsForSegmentsPerPathToReturn


###########################################
# Checks whether the data is as expected  #
###########################################

def check_gt_vs_num_classes(log, job_id, img_gt, num_classes):
    if img_gt is None: # If manual labels are not provided. E.g. in testing.
        return

    max_in_gt = np.max(img_gt)
    if np.max(img_gt) > num_classes - 1:  # num_classes includes background=0
        msg = job_id + " ERROR:\t GT labels include value [" + str(max_in_gt) + "] greater than what CNN expects." +\
              "\n\t In model-config the number of classes was specified as [" + str(num_classes) + "]." + \
              "\n\t Check your data or change the number of classes in model-config." + \
              "\n\t Note: number of classes in model config should include the background as a class."
        log.print3(msg)
        raise ValueError(msg)

