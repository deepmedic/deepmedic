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

from deepmedic.dataManagement.io import loadVolume
from deepmedic.neuralnet.pathwayTypes import PathwayTypes as pt
from deepmedic.dataManagement.preprocessing import pad_imgs_of_case, normalize_int_of_subj, calc_border_int_of_3d_img
from deepmedic.dataManagement.augmentSample import augment_sample
from deepmedic.dataManagement.augmentImage import augment_imgs_of_case


# Order of calls:
# getSampledDataAndLabelsForSubepoch
#    get_random_subjects_to_train_subep
#    get_n_samples_per_subj_in_subep
#    load_subj_and_get_samples
#        load_imgs_of_subject
#        sample_coords_of_segments
#        extractSegmentGivenSliceCoords
#            getImagePartFromSubsampledImageForTraining
#    shuffleSegments


# Main sampling process during training. Executed in parallel while training on a batch on GPU.
# Called from training.do_training()
# TODO: I think this should be a "sampler" class and moved to training.py. To keep this file generic-sampling.
def getSampledDataAndLabelsForSubepoch(log,
                                       train_val_or_test,
                                       num_parallel_proc,
                                       run_input_checks,
                                       cnn3d,
                                       max_n_cases_per_subep,
                                       n_samples_per_subep,
                                       sampling_type,
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
    training_or_validation_str = "Training" if train_val_or_test == "train" else "Validation"

    log.print3(sampler_id +
               " :=:=:=:=:=:=: Starting to sample for next [" + training_or_validation_str + "]... :=:=:=:=:=:=:")

    total_number_of_subjects = len(paths_per_chan_per_subj)
    inds_of_subjects_for_subep = get_random_subjects_to_train_subep(total_number_of_subjects, max_n_cases_per_subep)

    log.print3(sampler_id + " Out of [" + str(total_number_of_subjects) + "] subjects given for [" +
               training_or_validation_str + "], we will sample from maximum [" + str(max_n_cases_per_subep) +
               "] per subepoch.")
    log.print3(sampler_id + " Shuffled indices of subjects that were randomly chosen: " + str(inds_of_subjects_for_subep))

    # List, with [numberOfPathwaysThatTakeInput] sublists.
    # Each sublist is list of [partImagesLoadedPerSubepoch] arrays [channels, R,C,Z].
    channs_of_samples_per_path_for_subep = [[] for i in range(cnn3d.getNumPathwaysThatRequireInput())]
    lbls_predicted_part_of_samples_for_subep = []  # Labels only for the central/predicted part of segments.
    # Can be different than max_n_cases_per_subep, because of available images number.
    n_subjects_for_subep = len(inds_of_subjects_for_subep)

    # Get how many samples I should get from each subject.
    n_samples_per_subj = get_n_samples_per_subj_in_subep(n_samples_per_subep, n_subjects_for_subep)

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

                         n_subjects_for_subep,
                         inds_of_subjects_for_subep,
                         n_samples_per_subj
                         ]

    log.print3(sampler_id + " Will sample from [" + str(n_subjects_for_subep) +
               "] subjects for next " + training_or_validation_str + "...")

    jobs_inds_to_do = list(range(n_subjects_for_subep))  # One job per subject.

    if num_parallel_proc <= 0:  # Sequentially
        for job_idx in jobs_inds_to_do:
            (channs_of_samples_from_job_per_path,
             lbls_predicted_part_of_samples_from_job) = load_subj_and_get_samples(*([job_idx] + args_sampling_job))
            for pathway_i in range(cnn3d.getNumPathwaysThatRequireInput()):
                # concat does not copy.
                channs_of_samples_per_path_for_subep[pathway_i] += channs_of_samples_from_job_per_path[pathway_i]
            lbls_predicted_part_of_samples_for_subep += lbls_predicted_part_of_samples_from_job  # concat does not copy.

    else:  # Parallelize sampling from each subject
        while len(jobs_inds_to_do) > 0:  # While jobs remain.
            jobs = collections.OrderedDict()

            log.print3(sampler_id + " ******* Spawning children processes to sample from [" +
                       str(len(jobs_inds_to_do)) + "] subjects*******")
            log.print3(sampler_id + " MULTIPROC: Number of CPUs detected: " + str(multiprocessing.cpu_count()) +
                       ". Requested to use max: [" + str(num_parallel_proc) + "]")
            num_workers = min(num_parallel_proc, multiprocessing.cpu_count())
            log.print3(sampler_id + " MULTIPROC: Spawning [" + str(num_workers) + "] processes to load data and sample.")
            worker_pool = multiprocessing.Pool(processes=num_workers, initializer=init_sampling_proc)

            try:  # Stacktrace in multiprocessing: https://jichu4n.com/posts/python-multiprocessing-and-exceptions/
                for job_idx in jobs_inds_to_do:  # submit jobs
                    jobs[job_idx] = worker_pool.apply_async(load_subj_and_get_samples, ([job_idx] + args_sampling_job))

                # copy with list(...), so that this loops normally even if something is removed from list.
                for job_idx in list(jobs_inds_to_do):
                    try:
                        # timeout in case process for some reason never started (happens in py3)
                        (channs_of_samples_from_job_per_path,
                         lbls_predicted_part_of_samples_from_job) = jobs[job_idx].get(timeout=30)
                        for pathway_i in range(cnn3d.getNumPathwaysThatRequireInput()):
                            # concat does not copy.
                            channs_of_samples_per_path_for_subep[pathway_i] += channs_of_samples_from_job_per_path[pathway_i]
                        # concat does not copy.
                        lbls_predicted_part_of_samples_for_subep += lbls_predicted_part_of_samples_from_job
                        jobs_inds_to_do.remove(job_idx)
                    except multiprocessing.TimeoutError:
                        log.print3(
                            sampler_id + "\n\n WARN: MULTIPROC: Caught TimeoutError when getting results of job [" +
                            str(job_idx) + "].\n WARN: MULTIPROC: Will resubmit job [" + str(job_idx) + "].\n")
                        if num_workers == 1:
                            break  # If this 1 worker got stuck, every job will wait timeout. Slow. Recreate pool.

            except (Exception, KeyboardInterrupt) as e:
                log.print3(
                    sampler_id + "\n\n ERROR: Caught exception in getSampledDataAndLabelsForSubepoch(): " + str(e) + "\n")
                log.print3(traceback.format_exc())
                worker_pool.terminate()
                worker_pool.join()  # Will wait. A KeybInt will kill this (py3)
                raise e
            except:  # Catches everything, even a sys.exit(1) exception.
                log.print3(
                    sampler_id + "\n\n ERROR: Unexpected error in getSampledDataAndLabelsForSubepoch(). System info: ",
                    sys.exc_info()[0])
                worker_pool.terminate()
                worker_pool.join()
                raise Exception("Unexpected error.")
            else:  # Nothing went wrong
                # Needed in case any processes are hanging. worker_pool.close() does not solve this.
                worker_pool.terminate()
                worker_pool.join()

    # Got all samples for subepoch. Now shuffle them, together segments and their labels.
    (channs_of_samples_per_path_for_subep,
     lbls_predicted_part_of_samples_for_subep) = shuffleSegments(channs_of_samples_per_path_for_subep,
                                                                 lbls_predicted_part_of_samples_for_subep)
    log.print3(sampler_id + " TIMING: Sampling for next [" + training_or_validation_str +
               "] lasted: {0:.1f}".format(time.time() - start_time_sampling) + " secs.")

    log.print3(sampler_id + " :=:=:=:=:=:=: Finished sampling for next [" + training_or_validation_str + "] :=:=:=:=:=:=:")

    channs_of_samples_arr_per_path = [np.asarray(channs_of_samples_for_path, dtype="float32") for
                                      channs_of_samples_for_path in channs_of_samples_per_path_for_subep]

    lbls_predicted_part_of_samples_arr = np.asarray(lbls_predicted_part_of_samples_for_subep,
                                                    dtype="int32")  # Could be int16 to save RAM?

    return channs_of_samples_arr_per_path, lbls_predicted_part_of_samples_arr


def init_sampling_proc():
    # This will make child-processes ignore the KeyboardInterupt (sigInt). Parent will handle it.
    # See: http://stackoverflow.com/questions/11312525/catch-ctrlc-sigint-and-exit-multiprocesses-gracefully-in-python/35134329#35134329
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_random_subjects_to_train_subep(total_number_of_subjects,
                                       max_subjects_on_gpu_for_subepoch,
                                       get_max_subjects_for_gpu_even_if_total_less=False):
    # Returns: list of indices
    subjects_indices = list(range(total_number_of_subjects))  # list() for python3, cause shuffle() cant get range
    random_order_chosen_subjects = []
    random.shuffle(subjects_indices)  # does it in place. Now they are shuffled

    if max_subjects_on_gpu_for_subepoch >= total_number_of_subjects:
        random_order_chosen_subjects += subjects_indices

        # This is if I want to have a certain amount on GPU, even if total subjects are less.
        if get_max_subjects_for_gpu_even_if_total_less:
            while len(random_order_chosen_subjects) < max_subjects_on_gpu_for_subepoch:
                random.shuffle(subjects_indices)
                number_of_extra_subjects_to_get_to_fill_gpu = min(
                    max_subjects_on_gpu_for_subepoch - len(random_order_chosen_subjects), total_number_of_subjects)
                random_order_chosen_subjects += (subjects_indices[:number_of_extra_subjects_to_get_to_fill_gpu])
            assert len(random_order_chosen_subjects) == max_subjects_on_gpu_for_subepoch
            
    else:
        random_order_chosen_subjects += subjects_indices[:max_subjects_on_gpu_for_subepoch]

    return random_order_chosen_subjects


def get_n_samples_per_subj_in_subep(n_samples, n_subjects):
    # Distribute samples of each cat to subjects.
    n_samples_per_subj = np.ones([n_subjects], dtype="int32") * (n_samples // n_subjects)
    n_undistributed_samples = n_samples % n_subjects
    # Distribute samples that were left by inexact division.
    for idx in range(n_undistributed_samples):
        n_samples_per_subj[random.randint(0, n_subjects - 1)] += 1
    return n_samples_per_subj


def load_subj_and_get_samples(job_idx,
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

                              n_subjects_for_subep,
                              inds_of_subjects_for_subep,
                              n_samples_per_subj):
    # train_val_or_test: 'train', 'val' or 'test'
    # paths_per_chan_per_subj: [[ for channel-0 [ one path per subj ]], ..., [for channel-n  [ one path per subj ] ]]
    # n_samples_per_cat_per_subj: np arr, shape [num sampling categories, num subjects in subepoch]
    # returns: ( channs_of_samples_per_path, lbls_predicted_part_of_samples )
    job_id = "[TRA|JOB:" + str(job_idx) + "|PID:" + str(os.getpid()) + "]" if train_val_or_test == 'train' \
        else "[VAL|JOB:" + str(job_idx) + "|PID:" + str(os.getpid()) + "]"
    
    log.print3(job_id + " (#" + str(job_idx) + "/" + str(n_subjects_for_subep) + ") Sampling job started. " +
               "Load & sample from subject of index (in user's list): " + str(inds_of_subjects_for_subep[job_idx]) )

    # List, with [numberOfPathwaysThatTakeInput] sublists.
    # Each sublist is list of [partImagesLoadedPerSubepoch] arrays [channels, R,C,Z].
    channs_of_samples_per_path = [[] for i in range(cnn3d.getNumPathwaysThatRequireInput())]
    lbls_predicted_part_of_samples = []  # Labels only for the central/predicted part of segments.

    dims_highres_segment = cnn3d.pathways[0].getShapeOfInput(train_val_or_test)[2:]
    
    # Load images of subject
    time_load_0 = time.time()
    (channels,  # nparray [channels,dim0,dim1,dim2]
     gt_lbl_img,
     roi_mask,
     wmaps_to_sample_per_cat) = load_imgs_of_subject(log, job_id,
                                                     inds_of_subjects_for_subep[job_idx],
                                                     paths_per_chan_per_subj,
                                                     paths_to_lbls_per_subj,
                                                     paths_to_wmaps_per_sampl_cat_per_subj, # Placeholder in test
                                                     paths_to_masks_per_subj)
     
    # Pre-process images of subject
    time_load = time.time() - time_load_0
    time_prep_0 = time.time()
    (channels,
    gt_lbl_img,
    roi_mask,
    wmaps_to_sample_per_cat,
    pad_left_right_per_axis) = preprocess_imgs_of_subj(log, job_id,
                                                       channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat,
                                                       run_input_checks, cnn3d.num_classes, # checks
                                                       pad_input_imgs, cnn3d.recFieldCnn, dims_highres_segment, # pad
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

    # Get number of samples per sampling-category for the specific subject (class, foregr/backgr, etc)
    (n_samples_per_cat, valid_cats) = sampling_type.distribute_n_samples_to_categs(n_samples_per_subj[job_idx],
                                                                                   sampling_maps_per_cat)

    str_samples_per_cat = " Samples per category: "
    for cat_i in range(sampling_type.get_n_sampling_cats()):
        cat_string = sampling_type.get_sampling_cats_as_str()[cat_i]
        n_samples_for_cat = n_samples_per_cat[cat_i]
        sampling_map = sampling_maps_per_cat[cat_i]
        # Check if the class is valid for sampling.
        # Invalid if eg there is no such class in the subject's manual segmentation.
        if not valid_cats[cat_i]:
            log.print3( job_id + " WARN: Invalid sampling category! Sampling map just zeros! No [" + cat_string +
                        "] samples from this subject!")
            assert n_samples_for_cat == 0
            
        coords_of_samples = sample_coords_of_segments(log,
                                                      job_idx,
                                                      n_samples_for_cat,
                                                      dims_highres_segment,
                                                      dims_of_scan,
                                                      sampling_map)
        str_samples_per_cat += "[" + cat_string + ": " + str(len(coords_of_samples[0][0])) + "/" + str(
            n_samples_for_cat) + "] "

        # Use the just sampled coordinates of slices to actually extract the segments (data) from the subject's images.
        time_augm_samples = 0
        for image_part_i in range(len(coords_of_samples[0][0])):
            coord_center = coords_of_samples[0][:, image_part_i]

            (channs_of_sample_per_path,
             lbls_predicted_part_of_sample) = extractSegmentGivenSliceCoords(train_val_or_test,
                                                                             cnn3d,
                                                                             coord_center,
                                                                             channels,
                                                                             gt_lbl_img)

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
        
    log.print3(job_id + str_samples_per_cat + ". TIMING: " +
               "[Load: {0:.1f}".format(time_load) + "] "
               "[Preproc: {0:.1f}".format(time_prep) + "] " +
               "[Augm-Img: {0:.1f}".format(time_augm_img) + "] " +
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
    # paths_per_chan_per_subj: List of lists. One sublist per case. Each should contain...
    # ... as many elements(strings-filenamePaths) as numberOfChannels, pointing to (nii) channels of this case.
    
    log.print3(job_id + " Loading subject with 1st channel at: " + str(paths_per_chan_per_subj[subj_i][0]))
    
    numberOfNormalScaleChannels = len(paths_per_chan_per_subj[0])
        
    # Load the channels of the patient.
    inp_chan_dims = None  # Dimensions of the (padded) input channels.
    channels = None
    for channel_i in range(numberOfNormalScaleChannels):
        fullFilenamePathOfChannel = paths_per_chan_per_subj[subj_i][channel_i]
        if fullFilenamePathOfChannel != "-":  # normal case, filepath was given.
            channelData = loadVolume(fullFilenamePathOfChannel)
            
            if channels is None:
                # Initialize the array in which all the channels for the patient will be placed.
                inp_chan_dims = list(channelData.shape)
                channels = np.zeros((numberOfNormalScaleChannels, inp_chan_dims[0], inp_chan_dims[1], inp_chan_dims[2]))

            channels[channel_i] = channelData
        else:  # "-" was given in the config-listing file. Do Min-fill!
            log.print3(job_id + " WARN: No modality #" + str(channel_i) + " given. Will make zero-filled channel.")
            channels[channel_i] = 0.0
    
    # Load the class labels.
    if paths_to_lbls_per_subj is not None:
        fullFilenamePathOfGtLabels = paths_to_lbls_per_subj[subj_i]
        gt_lbl_img = loadVolume(fullFilenamePathOfGtLabels)

        if gt_lbl_img.dtype.kind not in ['i', 'u']:
            dtype_gt_lbls = 'int16'
            log.print3(job_id + " WARN: Loaded labels are dtype [" + str(gt_lbl_img.dtype) + "]."
                       " Rounding and casting to [" + dtype_gt_lbls + "]!")
            gt_lbl_img = np.rint(gt_lbl_img).astype(dtype_gt_lbls)
    else:
        gt_lbl_img = None  # For validation and testing

    if paths_to_masks_per_subj is not None:
        fullFilenamePathOfRoiMask = paths_to_masks_per_subj[subj_i]
        roi_mask = loadVolume(fullFilenamePathOfRoiMask)
        
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
            filepathsToTheWeightMapsOfAllPatientsForThisCategory = paths_to_wmaps_per_sampl_cat_per_subj[cat_i]
            filepathToTheWeightMapOfThisPatientForThisCategory = filepathsToTheWeightMapsOfAllPatientsForThisCategory[
                subj_i]
            weightedMapForThisCatData = loadVolume(filepathToTheWeightMapOfThisPatientForThisCategory)
            if not np.all(weightedMapForThisCatData >= 0):
                raise ValueError("Negative values found in weightmap. Unexpected. Zero or positives allowed.")
            wmaps_to_sample_per_cat[cat_i] = weightedMapForThisCatData
    else:
        wmaps_to_sample_per_cat = None

    return channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat


def preprocess_imgs_of_subj(log, job_id, channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat,
                            run_input_checks, n_classes,
                            pad_input_imgs, dims_rec_field, dims_highres_segment,
                            norm_prms):
    # job_id: Should be "" in testing.
    
    if run_input_checks:
        check_gt_vs_num_classes(log, gt_lbl_img, n_classes)
    
    (channels,
     gt_lbl_img,
     roi_mask,
     wmaps_to_sample_per_cat,
     pad_left_right_per_axis) = pad_imgs_of_case(channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat,
                                                 pad_input_imgs, dims_rec_field, dims_highres_segment)
    
    channels = normalize_int_of_subj(log, channels, roi_mask, norm_prms, job_id)
    
    return channels, gt_lbl_img, roi_mask, wmaps_to_sample_per_cat, pad_left_right_per_axis


# made for 3d
def sample_coords_of_segments(log,
                              job_id,
                              n_samples,
                              dims_of_segment,
                              dims_of_scan,
                              sampling_map):
    """
    Returns: [ coordsOfCentralVoxelsOfPartsSampled, sliceCoordsOfImagePartsSampled ]
             Coordinates (xyz indices) of the "central" voxel of sampled segments (1 voxel to the left if dimension is even).
             Also returns the indices of the image parts, left and right indices, INCLUSIVE BOTH SIDES.
    
    > coordsOfCentralVoxelsOfPartsSampled: array with shape: 3(xyz) x n_samples. 
        Example: [ xCoordsForCentralVoxelOfEachPart, yCoordsForCentralVoxelOfEachPart, zCoordsForCentralVoxelOfEachPart ]
        >> x/y/z-CoordsForCentralVoxelOfEachPart: 1-dim array with n_samples, holding the x-indices of samples in image.
    > sliceCoordsOfImagePartsSampled: 3(xyz) x NumberOfImagePartSamples x 2.
        The last dimension has [0] for the lower boundary of the slice, and [1] for the higher boundary. INCLUSIVE BOTH SIDES.
        Example: [ x-sliceCoordsOfImagePart, y-sliceCoordsOfImagePart, z-sliceCoordsOfImagePart ]
    """
    # Check if the weight map is fully-zeros. In this case, return no element.
    # Note: Currently, the caller function is checking this case already and does not let this being called.
    # Which is still fine.
    if np.sum(sampling_map) == 0:
        log.print3(job_id + " WARN: Sampling map was found just zeros! No image segments were sampled from this subject!")
        return [[[], [], []], [[], [], []]]

    imagePartsSampled = []

    # Now out of these, I need to randomly select one, which will be an ImagePart's central voxel.
    # But I need to be CAREFUL and get one that IS NOT closer to the image boundaries than the dimensions of the
    # ImagePart permit.

    # I look for lesions that are not closer to the image boundaries than the ImagePart dimensions allow.
    # KernelDim is always odd. BUT ImagePart dimensions can be odd or even.
    # If odd, ok, floor(dim/2) from central.
    # If even, dim/2-1 voxels towards the begining of the axis and dim/2 towards the end.
    # Ie, "central" imagePart voxel is 1 closer to begining.
    # BTW imagePartDim takes kernel into account (ie if I want 9^3 voxels classified per imagePart with kernel 5x5,
    # I want 13 dim ImagePart)

    # dim1: 1 row per r,c,z. Dim2: left/right width not to sample from (=half segment).
    halfImagePartBoundaries = np.zeros((len(dims_of_segment), 2), dtype='int32')

    # The below starts all zero. Will be Multiplied by other true-false arrays expressing if the relevant
    # voxels are within boundaries.
    # In the end, the final vector will be true only for the indices of lesions that are within all boundaries.
    booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries = np.zeros(sampling_map.shape, dtype="int32")

    # This loop leads to booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries to be true for the indices ...
    # ...that allow getting an imagePart CENTERED on them, and be safely within image boundaries. Note that ...
    # ... if the imagePart is of even dimension, the "central" voxel is one voxel to the left.
    for rcz_i in range(len(dims_of_segment)):
        if dims_of_segment[rcz_i] % 2 == 0:  # even
            dimensionDividedByTwo = dims_of_segment[rcz_i] // 2
            # central of ImagePart is 1 vox closer to begining of axes.
            halfImagePartBoundaries[rcz_i] = [dimensionDividedByTwo - 1,
                                              dimensionDividedByTwo]
        else:  # odd
            # If odd, middle voxel is the "central". Eg 5/2 = 2, with 3rd voxel being the central.
            dimensionDividedByTwoFloor = math.floor(dims_of_segment[rcz_i] // 2)
            halfImagePartBoundaries[rcz_i] = [dimensionDividedByTwoFloor, dimensionDividedByTwoFloor]
            # used to be [halfImagePartBoundaries[0][0]: -halfImagePartBoundaries[0][1]],
            # but in 2D case halfImagePartBoundaries might be ==0, causes problem and you get a null slice.
    booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries[
        halfImagePartBoundaries[0][0]: dims_of_scan[0] - halfImagePartBoundaries[0][1],
        halfImagePartBoundaries[1][0]: dims_of_scan[1] - halfImagePartBoundaries[1][1],
        halfImagePartBoundaries[2][0]: dims_of_scan[2] - halfImagePartBoundaries[2][1]] = 1

    constrainedWithImageBoundariesMaskToSample = sampling_map * \
                                                 booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries
    # normalize the probabilities to sum to 1, cause the function needs it as so.
    constrainedWithImageBoundariesMaskToSample = constrainedWithImageBoundariesMaskToSample / (
                1.0 * np.sum(constrainedWithImageBoundariesMaskToSample))

    flattenedConstrainedWithImageBoundariesMaskToSample = constrainedWithImageBoundariesMaskToSample.flatten()

    # This is going to be a 3xNumberOfImagePartSamples array.
    indicesInTheFlattenArrayThatWereSampledAsCentralVoxelsOfImageParts = np.random.choice(
        constrainedWithImageBoundariesMaskToSample.size,
        size=n_samples,
        replace=True,
        p=flattenedConstrainedWithImageBoundariesMaskToSample)
    # np.unravel_index([listOfIndicesInFlattened], dims) returns a tuple of arrays (eg 3 of them if 3 dimImage), 
    # where each of the array in the tuple has the same shape as the listOfIndices. 
    # They have the r/c/z coords that correspond to the index of the flattened version.
    # So, coordsOfCentralVoxelsOfPartsSampled will be array of shape: 3(rcz) x n_samples.
    coordsOfCentralVoxelsOfPartsSampled = np.asarray(
        np.unravel_index(indicesInTheFlattenArrayThatWereSampledAsCentralVoxelsOfImageParts,
                         constrainedWithImageBoundariesMaskToSample.shape  # the shape of the roi_mask/scan.
                         )
        )
    # Array with shape: 3(rcz) x NumberOfImagePartSamples x 2.
    # Last dimension has [0] for lowest boundary of slice, and [1] for highest boundary. INCLUSIVE BOTH SIDES.
    sliceCoordsOfImagePartsSampled = np.zeros(list(coordsOfCentralVoxelsOfPartsSampled.shape) + [2], dtype="int32")
    sliceCoordsOfImagePartsSampled[:, :, 0] = coordsOfCentralVoxelsOfPartsSampled - \
                                              halfImagePartBoundaries[:,
                                                                      np.newaxis,
                                                                      0]  # np.newaxis broadcasts. To broadcast the -+.
    sliceCoordsOfImagePartsSampled[:, :, 1] = coordsOfCentralVoxelsOfPartsSampled + \
                                              halfImagePartBoundaries[:, np.newaxis, 1]

    # coordsOfCentralVoxelsOfPartsSampled: Array of dimensions 3(rcz) x NumberOfImagePartSamples.
    # sliceCoordsOfImagePartsSampled: Array of dimensions 3(rcz) x NumberOfImagePartSamples x 2. ...
    # ... The last dim has [0] for the lower boundary of the slice, and [1] for the higher boundary.
    # ... The slice coordinates returned are INCLUSIVE BOTH sides.
    imagePartsSampled = [coordsOfCentralVoxelsOfPartsSampled, sliceCoordsOfImagePartsSampled]
    return imagePartsSampled


def getImagePartFromSubsampledImageForTraining(dimsOfPrimarySegment,
                                               recFieldCnn,
                                               subsampledImageChannels,
                                               image_part_slices_coords,
                                               subSamplingFactor,
                                               subsampledImagePartDimensions
                                               ):
    """
    This returns an image part from the sampled data, given the image_part_slices_coords,
    which has the coordinates where the normal-scale image part starts and ends (inclusive).
    (Actually, in this case, the right (end) part of image_part_slices_coords is not used.)
    
    The way it works is NOT optimal. From the beginning of the normal-resolution part,
    it goes further to the left 1 receptive-field and then forward xSubsamplingFactor receptive-fields.
    This stops it from being used with arbitrary size of subsampled segment (decoupled by the high-res segment).
    Now, the subsampled patch has to be of the same size as the normal-scale.
    To change this, I should find where THE FIRST TOP LEFT CENTRAL (predicted) VOXEL is, 
    and do the back-one-(sub)patch + front-3-(sub)patches from there, not from the begining of the patch.
    
    Current way it works (correct):
    If I have eg subsample factor=3 and 9 central-pred-voxels, I get 3 "central" voxels/patches for the
    subsampled-part. Straightforward. If I have a number of central voxels that is not an exact multiple of
    the subfactor, eg 10 central-voxels, I get 3+1 central voxels in the subsampled-part. 
    When the cnn is convolving them, they will get repeated to 4(last-layer-neurons)*3(factor) = 12, 
    and will get sliced down to 10, in order to have same dimension with the 1st pathway.
    """
    subsampledImageDimensions = subsampledImageChannels[0].shape

    subsampledChannelsForThisImagePart = np.ones((len(subsampledImageChannels),
                                                  subsampledImagePartDimensions[0],
                                                  subsampledImagePartDimensions[1],
                                                  subsampledImagePartDimensions[2]),
                                                 dtype='float32')

    numberOfCentralVoxelsClassifiedForEachImagePart_rDim = dimsOfPrimarySegment[0] - recFieldCnn[0] + 1
    numberOfCentralVoxelsClassifiedForEachImagePart_cDim = dimsOfPrimarySegment[1] - recFieldCnn[1] + 1
    numberOfCentralVoxelsClassifiedForEachImagePart_zDim = dimsOfPrimarySegment[2] - recFieldCnn[2] + 1

    # Calculate the slice that I should get, and where I should put it in the imagePart
    # (eg if near the borders, and I cant grab a whole slice-imagePart).
    rSlotsPreviously = ((subSamplingFactor[0] - 1) // 2) * recFieldCnn[0] if subSamplingFactor[0] % 2 == 1 \
        else (subSamplingFactor[0] - 2) // 2 * recFieldCnn[0] + recFieldCnn[0] // 2
    cSlotsPreviously = ((subSamplingFactor[1] - 1) // 2) * recFieldCnn[1] if subSamplingFactor[1] % 2 == 1 \
        else (subSamplingFactor[1] - 2) // 2 * recFieldCnn[1] + recFieldCnn[1] // 2
    zSlotsPreviously = ((subSamplingFactor[2] - 1) // 2) * recFieldCnn[2] if subSamplingFactor[2] % 2 == 1 \
        else (subSamplingFactor[2] - 2) // 2 * recFieldCnn[2] + recFieldCnn[2] // 2
    # 1*17
    rToCentralVoxelOfAnAveragedArea = subSamplingFactor[0] // 2 if subSamplingFactor[0] % 2 == 1 else (
                subSamplingFactor[
                    0] // 2 - 1)  # one closer to the beginning of dim. Same happens when I get parts of image.
    cToCentralVoxelOfAnAveragedArea = subSamplingFactor[1] // 2 if subSamplingFactor[1] % 2 == 1 else (
                subSamplingFactor[1] // 2 - 1)
    zToCentralVoxelOfAnAveragedArea = subSamplingFactor[2] // 2 if subSamplingFactor[2] % 2 == 1 else (
                subSamplingFactor[2] // 2 - 1)
    # This is where to start taking voxels from the subsampled image. From the beginning of the imagePart(1 st patch)...
    # ... go forward a few steps to the voxel that is like the "central" in this subsampled (eg 3x3) area.
    # ...Then go backwards -Patchsize to find the first voxel of the subsampled.

    # These indices can run out of image boundaries. I ll correct them afterwards.
    rlow = image_part_slices_coords[0][0] + rToCentralVoxelOfAnAveragedArea - rSlotsPreviously
    # If the patch is 17x17, I want a 17x17 subsampled Patch. BUT if the imgPART is 25x25 (9voxClass),
    # I want 3 subsampledPatches in my subsampPart to cover this area!
    # That is what the last term below is taking care of.
    # CAST TO INT because ceil returns a float, and later on when computing
    # rHighNonInclToPutTheNotPaddedInSubsampledImPart I need to do INTEGER DIVISION.
    rhighNonIncl = int(rlow + subSamplingFactor[0] * recFieldCnn[0] + (
                math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_rDim * 1.0) / subSamplingFactor[0]) - 1) *
                       subSamplingFactor[0])  # excluding index in segment
    clow = image_part_slices_coords[1][0] + cToCentralVoxelOfAnAveragedArea - cSlotsPreviously
    chighNonIncl = int(clow + subSamplingFactor[1] * recFieldCnn[1] + (
                math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_cDim * 1.0) / subSamplingFactor[1]) - 1) *
                       subSamplingFactor[1])
    zlow = image_part_slices_coords[2][0] + zToCentralVoxelOfAnAveragedArea - zSlotsPreviously
    zhighNonIncl = int(zlow + subSamplingFactor[2] * recFieldCnn[2] + (
                math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_zDim * 1.0) / subSamplingFactor[2]) - 1) *
                       subSamplingFactor[2])

    rlowCorrected = max(rlow, 0)
    clowCorrected = max(clow, 0)
    zlowCorrected = max(zlow, 0)
    rhighNonInclCorrected = min(rhighNonIncl, subsampledImageDimensions[0])
    chighNonInclCorrected = min(chighNonIncl, subsampledImageDimensions[1])
    zhighNonInclCorrected = min(zhighNonIncl, subsampledImageDimensions[2])  # This gave 7

    rLowToPutTheNotPaddedInSubsampledImPart = 0 if rlow >= 0 else abs(rlow) // subSamplingFactor[0]
    cLowToPutTheNotPaddedInSubsampledImPart = 0 if clow >= 0 else abs(clow) // subSamplingFactor[1]
    zLowToPutTheNotPaddedInSubsampledImPart = 0 if zlow >= 0 else abs(zlow) // subSamplingFactor[2]

    dimensionsOfTheSliceOfSubsampledImageNotPadded = [
        int(math.ceil((rhighNonInclCorrected - rlowCorrected) * 1.0 / subSamplingFactor[0])),
        int(math.ceil((chighNonInclCorrected - clowCorrected) * 1.0 / subSamplingFactor[1])),
        int(math.ceil((zhighNonInclCorrected - zlowCorrected) * 1.0 / subSamplingFactor[2]))
        ]

    # I now have exactly where to get the slice from and where to put it in the new array.
    for channel_i in range(len(subsampledImageChannels)):
        intensityZeroOfChannel = calc_border_int_of_3d_img(subsampledImageChannels[channel_i])
        subsampledChannelsForThisImagePart[channel_i] *= intensityZeroOfChannel

        sliceOfSubsampledImageNotPadded = subsampledImageChannels[channel_i][
                                          rlowCorrected: rhighNonInclCorrected: subSamplingFactor[0],
                                          clowCorrected: chighNonInclCorrected: subSamplingFactor[1],
                                          zlowCorrected: zhighNonInclCorrected: subSamplingFactor[2]
                                          ]
        subsampledChannelsForThisImagePart[
            channel_i,
            rLowToPutTheNotPaddedInSubsampledImPart: rLowToPutTheNotPaddedInSubsampledImPart +
                                                     dimensionsOfTheSliceOfSubsampledImageNotPadded[0],
            cLowToPutTheNotPaddedInSubsampledImPart: cLowToPutTheNotPaddedInSubsampledImPart +
                                                     dimensionsOfTheSliceOfSubsampledImageNotPadded[1],
            zLowToPutTheNotPaddedInSubsampledImPart: zLowToPutTheNotPaddedInSubsampledImPart +
                                                     dimensionsOfTheSliceOfSubsampledImageNotPadded[2]
            ] = sliceOfSubsampledImageNotPadded

    # placeholderReturn = np.ones([3,19,19,19], dtype="float32") #channel, dims
    return subsampledChannelsForThisImagePart


def shuffleSegments(channs_of_samples_per_path, lbls_predicted_part_of_samples):
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
                                   gt_lbl_img):
    # channels: numpy array [ n_channels, x, y, z ]
    # coord_center: indeces of the central voxel for the patch to be extracted.

    channs_of_sample_per_path = []
    # Sampling
    for pathway in cnn3d.pathways[:1]:  # Hack. The rest of this loop can work for the whole .pathways...
        # ... BUT the loop does not check what happens if boundaries are out of limits, to fill with zeros.
        # This is done in getImagePartFromSubsampledImageForTraining().
        # ... Update it in a nice way to be done here, and then take getImagePartFromSubsampledImageForTraining()
        # out and make loop go for every pathway.

        if pathway.pType() == pt.FC:
            continue
        subSamplingFactor = pathway.subsFactor()
        pathwayInputShapeRcz = pathway.getShapeOfInput(train_val_or_test)[2:]
        leftBoundaryRcz = [coord_center[0] - subSamplingFactor[0] * (pathwayInputShapeRcz[0] - 1) // 2,
                           coord_center[1] - subSamplingFactor[1] * (pathwayInputShapeRcz[1] - 1) // 2,
                           coord_center[2] - subSamplingFactor[2] * (pathwayInputShapeRcz[2] - 1) // 2]
        rightBoundaryRcz = [leftBoundaryRcz[0] + subSamplingFactor[0] * pathwayInputShapeRcz[0],
                            leftBoundaryRcz[1] + subSamplingFactor[1] * pathwayInputShapeRcz[1],
                            leftBoundaryRcz[2] + subSamplingFactor[2] * pathwayInputShapeRcz[2]]

        channelsForThisImagePart = channels[:,
                                   leftBoundaryRcz[0]: rightBoundaryRcz[0]: subSamplingFactor[0],
                                   leftBoundaryRcz[1]: rightBoundaryRcz[1]: subSamplingFactor[1],
                                   leftBoundaryRcz[2]: rightBoundaryRcz[2]: subSamplingFactor[2]]

        channs_of_sample_per_path.append(channelsForThisImagePart)

    # Extract the samples for secondary pathways. This whole for can go away,
    # if I update above code to check to slices out of limits.
    for pathway_i in range(len(cnn3d.pathways)):  # Except Normal 1st, cause that was done already.
        if cnn3d.pathways[pathway_i].pType() == pt.FC or cnn3d.pathways[pathway_i].pType() == pt.NORM:
            continue
        # this datastructure is similar to channelsForThisImagePart, but contains voxels from the subsampled image.
        dimsOfPrimarySegment = cnn3d.pathways[pathway_i].getShapeOfInput(train_val_or_test)[2:]

        # rightmost  are placeholders here.
        slicesCoordsOfSegmForPrimaryPathway = [[leftBoundaryRcz[0], rightBoundaryRcz[0] - 1],
                                               [leftBoundaryRcz[1], rightBoundaryRcz[1] - 1],
                                               [leftBoundaryRcz[2], rightBoundaryRcz[2] - 1]]
        channsForThisSubsampledPartAndPathway = getImagePartFromSubsampledImageForTraining(
            dimsOfPrimarySegment=dimsOfPrimarySegment,
            recFieldCnn=cnn3d.recFieldCnn,
            subsampledImageChannels=channels,
            image_part_slices_coords=slicesCoordsOfSegmForPrimaryPathway,
            subSamplingFactor=cnn3d.pathways[pathway_i].subsFactor(),
            subsampledImagePartDimensions=cnn3d.pathways[pathway_i].getShapeOfInput(train_val_or_test)[2:]
            )

        channs_of_sample_per_path.append(channsForThisSubsampledPartAndPathway)

    # Get ground truth labels for training.
    numOfCentralVoxelsClassifRcz = cnn3d.finalTargetLayer_outputShape[train_val_or_test][2:]
    leftBoundaryRcz = [coord_center[0] - (numOfCentralVoxelsClassifRcz[0] - 1) // 2,
                       coord_center[1] - (numOfCentralVoxelsClassifRcz[1] - 1) // 2,
                       coord_center[2] - (numOfCentralVoxelsClassifRcz[2] - 1) // 2]
    rightBoundaryRcz = [leftBoundaryRcz[0] + numOfCentralVoxelsClassifRcz[0],
                        leftBoundaryRcz[1] + numOfCentralVoxelsClassifRcz[1],
                        leftBoundaryRcz[2] + numOfCentralVoxelsClassifRcz[2]]
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

# TODO: This is very similar to sample_coords_of_segments() I believe, which is used for training.
#       Consider way to merge them.
def get_slice_coords_of_all_img_tiles(log,
                                      dimsOfPrimarySegment, # xyz dims of input to primary pathway (normal)
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
        zFarBoundary = min(zLowBoundaryNext + dimsOfPrimarySegment[2], inp_chan_dims[2])  # Excluding.
        zLowBoundary = zFarBoundary - dimsOfPrimarySegment[2]
        zLowBoundaryNext = zLowBoundaryNext + strideOfSegmentsPerDimInVoxels[2]
        zAxisCentralPartPredicted = False if zFarBoundary < inp_chan_dims[2] else True  # IMPORTANT CRITERION

        cLowBoundaryNext = 0
        cAxisCentralPartPredicted = False
        while not cAxisCentralPartPredicted:
            cFarBoundary = min(cLowBoundaryNext + dimsOfPrimarySegment[1], inp_chan_dims[1])  # Excluding.
            cLowBoundary = cFarBoundary - dimsOfPrimarySegment[1]
            cLowBoundaryNext = cLowBoundaryNext + strideOfSegmentsPerDimInVoxels[1]
            cAxisCentralPartPredicted = False if cFarBoundary < inp_chan_dims[1] else True

            rLowBoundaryNext = 0
            rAxisCentralPartPredicted = False
            while not rAxisCentralPartPredicted:
                rFarBoundary = min(rLowBoundaryNext + dimsOfPrimarySegment[0], inp_chan_dims[0])  # Excluding.
                rLowBoundary = rFarBoundary - dimsOfPrimarySegment[0]
                rLowBoundaryNext = rLowBoundaryNext + strideOfSegmentsPerDimInVoxels[0]
                rAxisCentralPartPredicted = False if rFarBoundary < inp_chan_dims[0] else True

                # In case I pass a brain-mask, I ll use it to only predict inside it. Otherwise, whole image.
                if isinstance(roi_mask, np.ndarray):
                    if not np.any(roi_mask[rLowBoundary:rFarBoundary,
                                  cLowBoundary:cFarBoundary,
                                  zLowBoundary:zFarBoundary]
                                  ):  # all of it is out of the brain so skip it.
                        continue

                sliceCoordsOfSegmentsToReturn.append(
                    [[rLowBoundary, rFarBoundary - 1], [cLowBoundary, cFarBoundary - 1],
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
                                    recFieldCnn):
    # channelsOfImageNpArray: numpy array [ n_channels, x, y, z ]
    numberOfSegmentsToExtract = len(sliceCoordsOfSegmentsToExtract)
    channsForSegmentsPerPathToReturn = [[] for i in range(
        cnn3d.getNumPathwaysThatRequireInput())]  # [pathway, image parts, channels, r, c, z]
    # RCZ dims of input to primary pathway (NORMAL). Which should be the first one in .pathways.
    dimsOfPrimarySegment = cnn3d.pathways[0].getShapeOfInput("test")[2:]

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
            # the right hand values are placeholders in this case.
            slicesCoordsOfSegmForPrimaryPathway = [[rLowBoundary, rFarBoundary - 1],
                                                   [cLowBoundary, cFarBoundary - 1],
                                                   [zLowBoundary, zFarBoundary - 1]]
            channsForThisSubsPathForThisSegm = getImagePartFromSubsampledImageForTraining(
                dimsOfPrimarySegment=dimsOfPrimarySegment,
                recFieldCnn=recFieldCnn,
                subsampledImageChannels=channelsOfImageNpArray,
                image_part_slices_coords=slicesCoordsOfSegmForPrimaryPathway,
                subSamplingFactor=cnn3d.pathways[pathway_i].subsFactor(),
                subsampledImagePartDimensions=cnn3d.pathways[pathway_i].getShapeOfInput("test")[2:]
                )
            channsForSegmentsPerPathToReturn[pathway_i].append(channsForThisSubsPathForThisSegm)

    return channsForSegmentsPerPathToReturn


###########################################
# Checks whether the data is as expected  #
###########################################

def check_gt_vs_num_classes(log, img_gt, num_classes):
    job_id = "[" + str(os.getpid()) + "]"
    max_in_gt = np.max(img_gt)
    if np.max(img_gt) > num_classes - 1:  # num_classes includes background=0
        msg = job_id + " ERROR:\t GT labels include value [" + str(max_in_gt) + "] greater than what CNN expects." +\
              "\n\t In model-config the number of classes was specified as [" + str(num_classes) + "]." + \
              "\n\t Check your data or change the number of classes in model-config." + \
              "\n\t Note: number of classes in model config should include the background as a class."
        log.print3(msg)
        raise ValueError(msg)
