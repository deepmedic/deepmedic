# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import time
import numpy as np
import math

from deepmedic.logging.accuracyMonitor import AccuracyMonitorForEpSegm
from deepmedic.dataManagement.sampling import load_imgs_of_subject, preproc_imgs_of_subj
from deepmedic.dataManagement.sampling import get_slice_coords_of_all_img_tiles
from deepmedic.dataManagement.sampling import extractSegmentsGivenSliceCoords
from deepmedic.dataManagement.io import savePredImgToNiiWithOriginalHdr, saveFmImgToNiiWithOriginalHdr, \
    save4DImgWithAllFmsToNiiWithOriginalHdr
from deepmedic.dataManagement.preprocessing import unpad_3d_img
from deepmedic.neuralnet.pathwayTypes import PathwayTypes as pt
from deepmedic.logging.utils import strListFl4fNA, getMeanPerColOf2dListExclNA, print_progress_step_test



def calc_num_fms_to_save(cnn_pathways, fm_idxs):
    fm_num = 0
    for pathway in cnn_pathways:
        fm_idxs_pathway = fm_idxs[pathway.pType()]
        if fm_idxs_pathway:
            for layer_i in range(len(pathway.get_blocks())):
                fm_idxs_layer_pathway = fm_idxs_pathway[layer_i]
                if fm_idxs_layer_pathway:
                    # If the user specifies to grab more feature maps than exist (eg 9999),
                    # correct it, replacing it with the number of FMs in the layer.
                    fm_this_layer_num = pathway.get_block(layer_i).get_n_fms_out()
                    fm_idxs_layer_pathway[1] = min(fm_idxs_layer_pathway[1], fm_this_layer_num)
                    fm_num += fm_idxs_layer_pathway[1] - fm_idxs_layer_pathway[0]
    return fm_num


def stitch_predicted_to_prob_maps(prob_maps_per_class, idx_next_tile_in_pred_vols, 
                                  prob_maps_batch, batch_size, slice_coords, unpred_margin, stride):
    # prob_maps_per_class: The whole volume that is going to be the final output of the system.
    # prob_maps_batch: the predictions of the cnn for tiles/segments in a batch. Must be stitched together.

    for tile_i in range(batch_size):
        # Now put the label-cube in the new-label-segmentation-image, at the correct position.
        # The very first label goes not in index 0,0,0 but half-patch further away!
        # At the position of the central voxel of the top-left patch!
        slice_coords_tile = slice_coords[idx_next_tile_in_pred_vols]
        top_left = [slice_coords_tile[0][0], slice_coords_tile[1][0], slice_coords_tile[2][0]]
        prob_maps_per_class[:,
                            top_left[0] + unpred_margin[0][0]: top_left[0] + unpred_margin[0][0] + stride[0],
                            top_left[1] + unpred_margin[1][0]: top_left[1] + unpred_margin[1][0] + stride[1],
                            top_left[2] + unpred_margin[2][0]: top_left[2] + unpred_margin[2][0] + stride[2]
                            ] = prob_maps_batch[tile_i]
        idx_next_tile_in_pred_vols += 1

    return idx_next_tile_in_pred_vols, prob_maps_per_class


def calculate_num_voxels_sub(num_central_voxels, pathway):
    num_voxels_sub = np.zeros(3)
    for i in range(3):
        num_voxels_sub[i] = num_central_voxels[i] - 1 if pathway.pType() != pt.SUBS else \
            int(math.ceil((num_central_voxels[i] * 1.0) / pathway.subs_factor()[i]) - 1)

    return [int(a) for a in num_voxels_sub]


def calculate_num_central_voxels_dir(num_central_voxels, pathway):
    num_voxels_dir = np.zeros(3)
    # the math.ceil / subsamplingFactor is a trick to make it work for even subsamplingFactor too.
    # Eg 9/2=4.5 => Get 5. Combined with the trick at repeat,
    # I get my correct number of central voxels hopefully.
    for i in range(3):
        num_voxels_dir[i] = int(math.ceil((num_central_voxels[i] * 1.0) / pathway.subs_factor()[i])) \
            if pathway.pType() == pt.SUBS else int(num_central_voxels[i])

    return [int(a) for a in num_voxels_dir]


def stitch_predicted_to_fms(array_fms_to_save, idx_next_tile_in_fm_vols,
                            fms_per_layer_and_path_for_batch, batch_size, slice_coords, unpred_margin, stride,
                            outp_pred_dims, cnn_pathways, idxs_fms_to_save):
    # array_fms_to_save: The whole feature maps that are going to be the final output of the system.
    # fms_per_layer_and_path_for_batch: FM activations in CNN for tiles/segments in a batch. Must be stitched.
    # idx_curr is the index in the multidimensional array that holds all the to-be-visualised-fms.
    # ... It is the one that corresponds to the next to-be-visualised layer_idx.
    idx_curr = 0
    # layer_idx is the index over all the layers in the returned list.
    # ... I will work only with the ones specified to visualise.
    layer_idx = 0

    for pathway in cnn_pathways:
        for layer_i in range(len(pathway.get_blocks())):
            if idxs_fms_to_save[pathway.pType()] == [] or idxs_fms_to_save[pathway.pType()][layer_i] == []:
                continue
            fms_to_extract_idxs = idxs_fms_to_save[pathway.pType()][layer_i]
            fms_layer = fms_per_layer_and_path_for_batch[layer_idx]
            # We specify a range of fms to visualise from a layer.
            # curr_idx : fms_to_fill_high_idx defines were to put them in the multidimensional-image-array.
            fms_to_fill_high_idx = idx_curr + fms_to_extract_idxs[1] - fms_to_extract_idxs[0]

            fm_to_reconstruct = array_fms_to_save[idx_curr:fms_to_fill_high_idx]

            # ========================================================================================
            # ====the following calculations could be move OUTSIDE THE FOR LOOPS, by using the kernel-size
            # parameter (from the cnn instance) instead of the shape of the returned value.
            # ====fmsReturnedForATestBatchForCertainLayer.shape[2] - (outp_pred_dims[0]-1)
            # is essentially the width of the patch left after the convolutions.
            # ====These calculations are pathway and layer-specific. So they could be done once, prior to
            # image processing, and results cached in a list to be accessed during the loop.

            (num_voxels_sub_r,
             num_voxels_sub_c,
             num_voxels_sub_z) = calculate_num_voxels_sub(outp_pred_dims, pathway)

            r_patch_dim = fms_layer.shape[2] - num_voxels_sub_r
            c_patch_dim = fms_layer.shape[3] - num_voxels_sub_c
            z_patch_dim = fms_layer.shape[4] - num_voxels_sub_z

            # the -1 so that if width is even, I'll get the left voxel from the centre as 1st,
            # which I THINK is how I am getting the patches from the original image.
            r_top_left_central_voxel = int((r_patch_dim - 1) // 2)
            c_top_left_central_voxel = int((c_patch_dim - 1) // 2)
            z_top_left_central_voxel = int((z_patch_dim - 1) // 2)

            (num_central_voxels_r,
             num_central_voxels_c,
             num_central_voxels_z) = calculate_num_central_voxels_dir(outp_pred_dims, pathway)

            # ============================================================================================

            # Grab the central voxels of the predicted fms from the cnn in this batch.
            central_voxels_all_fms = fms_layer[:,  # batchsize
                                               :,  # number of featuremaps
                                               r_top_left_central_voxel:r_top_left_central_voxel + num_central_voxels_r,
                                               c_top_left_central_voxel:c_top_left_central_voxel + num_central_voxels_c,
                                               z_top_left_central_voxel:z_top_left_central_voxel + num_central_voxels_z]

            # If the pathway that is visualised currently is the subsampled,
            # I need to upsample the central voxels to the normal resolution,
            # before reconstructing the image-fm.

            # subsampled layer. Remember that this returns smaller dimension outputs,
            # because it works in the subsampled space.
            # I need to repeat it, to bring it to the dimensions of the normal-voxel-space.
            if pathway.pType() == pt.SUBS:
                expanded_output_r = np.repeat(central_voxels_all_fms, pathway.subs_factor()[0], axis=2)
                expanded_output_rc = np.repeat(expanded_output_r, pathway.subs_factor()[1], axis=3)
                expanded_output_rcz = np.repeat(expanded_output_rc, pathway.subs_factor()[2], axis=4)
                # The below is a trick to get correct number of voxels even when subsampling factor is
                # even or not exact divisor of the number of central voxels.
                # ...This trick is coupled with the ceil() when getting the
                # numberOfCentralVoxelsToGetInDirectionR above.
                central_voxels_all_fms_batch = expanded_output_rcz[:,
                                                                   :,
                                                                   0:outp_pred_dims[0],
                                                                   0:outp_pred_dims[1],
                                                                   0:outp_pred_dims[2]
                                                                   ]
            else:
                central_voxels_all_fms_batch = central_voxels_all_fms

            # ----For every image part within this batch, reconstruct the corresponding part of the feature
            # maps of the layer we are currently visualising in this loop.
            for tile_batch_idx in range(batch_size):
                # Now put the label-cube in the new-label-segmentation-image, at the correct position.
                # The very first label goes not in index 0,0,0 but half-patch further away! At the position
                # of the central voxel of the top-left patch!
                slice_coords_tile = slice_coords[idx_next_tile_in_fm_vols + tile_batch_idx]
                coords_top_left_voxel = [slice_coords_tile[0][0],
                                         slice_coords_tile[1][0],
                                         slice_coords_tile[2][0]]

                # I put the central-predicted-voxels of all FMs to the corresponding,
                # newly created images all at once.
                fm_to_reconstruct[:,  # last dimension is the number-of-Fms, I create an image for each.

                                  coords_top_left_voxel[0] + unpred_margin[0][0]:
                                  coords_top_left_voxel[0] + unpred_margin[0][0] + stride[0],

                                  coords_top_left_voxel[1] + unpred_margin[1][0]:
                                  coords_top_left_voxel[1] + unpred_margin[1][0] + stride[1],

                                  coords_top_left_voxel[2] + unpred_margin[2][0]:
                                  coords_top_left_voxel[2] + unpred_margin[2][0] + stride[2]

                                  ] = central_voxels_all_fms_batch[tile_batch_idx]

            idx_curr = fms_to_fill_high_idx

            layer_idx += 1

    # all the image parts before this were reconstructed for all layers and feature maps.
    # Next batch-iteration should start from this
    idx_next_tile_in_fm_vols += batch_size

    return idx_next_tile_in_fm_vols, array_fms_to_save


def prepare_feeds_dict(feeds, channs_of_tiles_per_path):
    # TODO: Can we rename the input feeds so that they are easier to deal with?
    feeds_dict = {feeds['x']: np.asarray(channs_of_tiles_per_path[0], dtype='float32')}
    for path_i in range(len(channs_of_tiles_per_path[1:])):
        feeds_dict.update(
            {feeds['x_sub_' + str(path_i)]: np.asarray(channs_of_tiles_per_path[1 + path_i], dtype='float32')})

    return feeds_dict
    

def predict_whole_volume_by_tiling(log, sessionTf, cnn3d,
                                   channels, roi_mask, inp_shapes_per_path, unpred_margin,
                                   batchsize, save_fms_flag, idxs_fms_to_save):
    # One of the main routines. Segment whole volume tile-by-tile.
    
    # For tiling the volume: Stride is how much I move in each dimension to get the next tile.
    # I stride exactly the number of voxels that are predicted per forward pass.
    outp_pred_dims = cnn3d.calc_outp_dims_given_inp(inp_shapes_per_path[0])
    stride_of_tiling = outp_pred_dims # [str-x, str-y, str-z]
    # Find the total number of feature maps that will be created:
    # NOTE: save_fms_flag should contain an entry per pathwayType, even if just [].
    # If not [], the list should contain one entry per layer of the pathway, even if just [].
    # The layer entries, if not [], they should have to integers, lower and upper FM to visualise.
    n_fms_to_save = calc_num_fms_to_save(cnn3d.pathways, idxs_fms_to_save) if save_fms_flag else 0
    
    # Arrays that will be returned.
    inp_chan_dims = list(channels.shape[1:]) # Dimensions of (padded) input channels.
    # The main output. Predicted probability-maps for the whole volume, one per class.
    # Will be constructed by stitching together the predictions from each tile.
    prob_maps_vols = np.zeros([cnn3d.num_classes]+inp_chan_dims, dtype="float32")
    # create the big array that will hold all the fms (for feature extraction).
    array_fms_to_save = np.zeros([n_fms_to_save] + inp_chan_dims, dtype="float32") if save_fms_flag else None

    # Tile the image and get all slices of the tiles that it fully breaks down to.
    slice_coords_all_tiles = get_slice_coords_of_all_img_tiles(log,
                                                               inp_shapes_per_path[0],
                                                               stride_of_tiling,
                                                               batchsize,
                                                               inp_chan_dims,
                                                               roi_mask)

    n_tiles_for_subj = len(slice_coords_all_tiles)
    log.print3("Ready to make predictions for all image segments (parts).")
    log.print3("Total number of Segments to process:" + str(n_tiles_for_subj))
    
    idx_next_tile_in_pred_vols = 0
    idx_next_tile_in_fm_vols = 0
    n_batches = n_tiles_for_subj // batchsize
    t_fwd_pass_subj = 0 # time it took for forward pass over all tiles of subject.
    print_progress_step_test(log, n_batches, 0, batchsize, n_tiles_for_subj)    
    for batch_i in range(n_batches):
        
        # Extract data for the segments of this batch.
        # ( I could modularize extractDataOfASegmentFromImagesUsingSampledSliceCoords()
        # of training and use it here as well. )
        slice_coords_of_tiles_batch = slice_coords_all_tiles[batch_i * batchsize: (batch_i + 1) * batchsize]
        channs_of_tiles_per_path = extractSegmentsGivenSliceCoords(cnn3d,
                                                                   slice_coords_of_tiles_batch,
                                                                   channels,
                                                                   inp_shapes_per_path,
                                                                   outp_pred_dims)

        # ============================== Perform forward pass ====================================
        t_fwd_start = time.time()
        ops_to_fetch = cnn3d.get_main_ops('test')
        list_of_ops = [ops_to_fetch['pred_probs']] + ops_to_fetch['list_of_fms_per_layer']
        feeds_dict = prepare_feeds_dict(cnn3d.get_main_feeds('test'), channs_of_tiles_per_path)
        # Forward pass
        out_val_of_ops = sessionTf.run(fetches=list_of_ops, feed_dict=feeds_dict)
        prob_maps_batch = out_val_of_ops[0]
        fms_per_layer_and_path_for_batch = out_val_of_ops[1:] # [] if no FMs specified.
        t_fwd_pass_subj += time.time() - t_fwd_start
        
        # ================ Construct probability maps (volumes) by Stitching  ====================
        # Stitch predictions for tiles of this batch, to create the probability maps for whole volume.
        # Each prediction for a tile needs to be placed in the correct location in the volume.
        (idx_next_tile_in_pred_vols,
         prob_maps_vols) = stitch_predicted_to_prob_maps(prob_maps_vols,
                                                         idx_next_tile_in_pred_vols,
                                                         prob_maps_batch,
                                                         batchsize,
                                                         slice_coords_all_tiles,
                                                         unpred_margin,
                                                         stride_of_tiling)

        # ============== Construct feature maps (volumes) by Stitching =====================
        if save_fms_flag:
            (idx_next_tile_in_fm_vols,
             array_fms_to_save) = stitch_predicted_to_fms(array_fms_to_save,
                                                          idx_next_tile_in_fm_vols,
                                                          fms_per_layer_and_path_for_batch,
                                                          batchsize,
                                                          slice_coords_all_tiles,
                                                          unpred_margin,
                                                          stride_of_tiling,
                                                          outp_pred_dims,
                                                          cnn3d.pathways,
                                                          idxs_fms_to_save)
        print_progress_step_test(log, n_batches, batch_i + 1, batchsize, n_tiles_for_subj)
        # Done with batch
        
    log.print3("TIMING: Segmentation of subject: [Forward Pass:] {0:.2f}".format(t_fwd_pass_subj) + " secs.")

    return prob_maps_vols, array_fms_to_save


def unpad_img(img, unpad_input, pad_left_right_per_axis):
    # unpad_input: If True, pad_left_right_per_axis == ((0,0), (0,0), (0,0)).
    #              unpad_3d_img deals with no padding. So, this check is not required.
    if not unpad_input:
        return img
    if img is None: # Deals with the case something has not been given. E.g. roi_mask or gt_lbls.
        return None
    return unpad_3d_img(img, pad_left_right_per_axis)


def unpad_list_of_imgs(list_imgs, unpad_input, pad_left_right_per_axis):
    if not unpad_input or list_imgs is None:
        return list_imgs
    
    list_unpadded_imgs = []
    for img in list_imgs:
        list_unpadded_imgs.append(unpad_img(img, unpad_input, pad_left_right_per_axis)) # Deals with None.
    return list_unpadded_imgs


def save_pred_seg(pred_seg, save_pred_seg_bool, suffix_seg, seg_names, filepaths, subj_i, log):
    # filepaths: list of all filepaths to each channel image of each subject. To get header.
    # Save the image. Pass the filename paths of the normal image to duplicate the header info.
    if save_pred_seg_bool:
        savePredImgToNiiWithOriginalHdr(pred_seg,
                                        seg_names,
                                        filepaths,
                                        subj_i,
                                        suffix_seg,
                                        np.dtype(np.int16),
                                        log)


def save_prob_maps(prob_maps, save_prob_maps_bool, suffix_prob_map, prob_names, filepaths, subj_i, log):
    # filepaths: list of all filepaths to each channel img of each subject. To get header.
    # Save the image. Pass the filename paths of the normal image to duplicate the header info.
    for class_i in range(len(prob_maps)):
        if (len(save_prob_maps_bool) >= class_i + 1) and save_prob_maps_bool[class_i]:
            suffix = suffix_prob_map + str(class_i)
            prob_map = prob_maps[class_i]
            savePredImgToNiiWithOriginalHdr(prob_map,
                                            prob_names,
                                            filepaths,
                                            subj_i,
                                            suffix,
                                            np.dtype(np.float32),
                                            log)


def save_fms_individual(save_flag, multidim_fm_array, cnn_pathways, fm_idxs, fms_names, filepaths, subj_i, log):
    if not save_flag:
        return
    
    idx_curr = 0
    for pathway_i in range(len(cnn_pathways)):
        pathway = cnn_pathways[pathway_i]
        fms_idx_pathway = fm_idxs[pathway.pType()]
        if fms_idx_pathway:
            for layer_i in range(len(pathway.get_blocks())):
                fms_idx_layer_pathway = fms_idx_pathway[layer_i]
                if fms_idx_layer_pathway:
                    for fmActualNumber in range(fms_idx_layer_pathway[0], fms_idx_layer_pathway[1]):
                        fm_to_save = multidim_fm_array[idx_curr]
                        
                        saveFmImgToNiiWithOriginalHdr(fm_to_save,
                                                      fms_names,
                                                      filepaths,
                                                      subj_i,
                                                      pathway_i,
                                                      layer_i,
                                                      fmActualNumber,
                                                      log)
                        idx_curr += 1


def calculate_dice(pred_seg, gt_lbl):
    union_correct = pred_seg * gt_lbl
    tp_num = np.sum(union_correct)
    gt_pos_num = np.sum(gt_lbl)
    dice = (2.0 * tp_num) / (np.sum(pred_seg) + gt_pos_num) if gt_pos_num != 0 else -1
    return dice


def calc_metrics_for_subject(metrics_per_subj_per_c, subj_i, pred_seg, pred_seg_in_roi,
                             gt_lbl, gt_lbl_in_roi, n_classes, na_pattern):
    # Calculate DSC per class.
    for c in range(n_classes):
        if c == 0:  # do the eval for WHOLE FOREGROUND segmentation (all classes merged except background)
            # Merge every class except the background (assumed to be label == 0 )
            pred_seg_bin_c = pred_seg > 0 # Now it's binary
            pred_seg_bin_c_in_roi = pred_seg_in_roi > 0
            gt_lbl_bin_c = gt_lbl > 0
            gt_lbl_bin_c_in_roi = gt_lbl_in_roi > 0
        else:
            pred_seg_bin_c = np.rint(pred_seg) == c # randint for valid comparison, in case array is float)
            pred_seg_bin_c_in_roi = np.rint(pred_seg_in_roi) == c
            gt_lbl_bin_c = np.rint(gt_lbl) == c
            gt_lbl_bin_c_in_roi = np.rint(gt_lbl_in_roi) == c
            
        # Calculate the 3 Dices.
        # Dice1 = Allpredicted/allLesions,
        # Dice2 = PredictedWithinRoiMask / AllLesions ,
        # Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask.
        
        # Dice1 = Allpredicted/allLesions
        dice_1 = calculate_dice(pred_seg_bin_c, gt_lbl_bin_c)
        metrics_per_subj_per_c['dice1'][subj_i][c] = dice_1 if dice_1 != -1 else na_pattern

        # Dice2 = PredictedWithinRoiMask / AllLesions
        dice_2 = calculate_dice(pred_seg_bin_c_in_roi, gt_lbl_bin_c)
        metrics_per_subj_per_c['dice2'][subj_i][c] = dice_2 if dice_2 != -1 else na_pattern
        
        # Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask
        dice_3 = calculate_dice(pred_seg_bin_c_in_roi, gt_lbl_bin_c_in_roi)
        metrics_per_subj_per_c['dice3'][subj_i][c] = dice_3 if dice_3 != -1 else na_pattern
        
    return metrics_per_subj_per_c


def report_metrics_for_subject(log, metrics_per_subj_per_c, subj_i, na_pattern, val_test_print):
    log.print3("+++++++++++ Reporting Segmentation Metrics for Subject #" + str(subj_i) + " +++++++++++")
    log.print3("ACCURACY: (" + str(val_test_print) + ")" +
           " The Per-Class DICE Coefficients for subject with index #" + str(subj_i) + " equal:" +
           " DICE1=" + strListFl4fNA(metrics_per_subj_per_c['dice1'][subj_i], na_pattern) +
           " DICE2=" + strListFl4fNA(metrics_per_subj_per_c['dice2'][subj_i], na_pattern) +
           " DICE3=" + strListFl4fNA(metrics_per_subj_per_c['dice3'][subj_i], na_pattern))
    print_dice_explanations(log)
    
    
def print_dice_explanations(log):
    log.print3("EXPLANATION: DICE1/2/3 are lists with the DICE per class."
        "\n\t For Class-0, we calculate DICE for whole foreground: all labels merged except background, label=0."
        "\n\t Useful for multi-class problems.")
    log.print3("EXPLANATION: DICE1 is calculated as segmentation over whole volume VS whole Ground Truth (GT)."
        "\n\t DICE2 is the segmentation within the ROI vs GT."
        "\n\t DICE3 is segmentation within the ROI vs the GT within the ROI.")
    log.print3("EXPLANATION: If an ROI mask has been provided, you should be consulting DICE2 or DICE3.")
    
    
def calc_stats_of_metrics(metrics_per_subj_per_c, na_pattern):
    # mean_metrics: a dictionary. Key is the name of the metric. 
    #               Value is a list [list_subj_1, ..., list_subj_s]
    #               each list_subj_i is a list with the dice for each class, [dsc_class_1, ..., dsc_class_c]
    mean_metrics = {}
    for k in metrics_per_subj_per_c.keys():
        mean_metrics[k] = getMeanPerColOf2dListExclNA(metrics_per_subj_per_c[k], na_pattern)
    return mean_metrics


def report_mean_metrics(log, mean_metrics, na_pattern, val_test_print):
    # dices_1/2/3: A list with NUMBER_OF_SUBJECTS sublists.
    #              Each sublist has one dice-score per class.
    log.print3("")
    log.print3("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.print3("++++++++++++++++++++++++ Segmentation of all subjects finished ++++++++++++++++++++++++++++")
    log.print3("++++++++++++++ Reporting Average Segmentation Metrics over all subjects +++++++++++++++++++")
    
    log.print3("ACCURACY: (" + str(val_test_print) + ")" +
               " The Per-Class average DICE Coefficients over all subjects are:" +
               " DICE1=" + strListFl4fNA(mean_metrics['dice1'], na_pattern) +
               " DICE2=" + strListFl4fNA(mean_metrics['dice2'], na_pattern) +
               " DICE3=" + strListFl4fNA(mean_metrics['dice3'], na_pattern))
    
    print_dice_explanations(log)
    
    return mean_metrics


# Main routine for testing.
def inference_on_whole_volumes(sessionTf,
                               cnn3d,
                               log,
                               val_or_test,
                               savePredictedSegmAndProbsDict,
                               paths_per_chan_per_subj,
                               paths_to_lbls_per_subj,
                               paths_to_masks_per_subj,
                               namesForSavingSegmAndProbs,
                               suffixForSegmAndProbsDict,
                               # Hyper parameters
                               batchsize,
                               # Data compatibility checks
                               run_input_checks,
                               # Pre-Processing
                               pad_input,
                               norm_prms,
                               # Saving feature maps
                               save_fms_flag,
                               idxs_fms_to_save,
                               namesForSavingFms,
                               # Sampling
                               inp_shapes_per_path):
    # save_fms_flag: should contain an entry per pathwayType, even if just []...
    #       ... If not [], the list should contain one entry per layer of the pathway, even if just [].
    #       ... The layer entries, if not [], they should have to integers, lower and upper FM to visualise.
    #       ... Excluding the highest index.

    val_test_print = "Validation" if val_or_test == "val" else "Testing"
    
    log.print3("")
    log.print3("##########################################################################################")
    log.print3("#\t\t  Starting full Segmentation of " + str(val_test_print) + " subjects   \t\t\t#")
    log.print3("##########################################################################################")

    t_start = time.time()

    NA_PATTERN = AccuracyMonitorForEpSegm.NA_PATTERN
    n_classes = cnn3d.num_classes
    n_subjects = len(paths_per_chan_per_subj)
    unpred_margin = cnn3d.calc_unpredicted_margin(inp_shapes_per_path[0])
    # One dice score for whole foreground (0) AND one for each actual class
    # Dice1 - AllpredictedLes/AllLesions
    # Dice2 - predictedInsideRoiMask/AllLesions
    # Dice3 - predictedInsideRoiMask/ LesionsInsideRoiMask (for comparisons)
    # Each is a list of dimensions: n_subjects X n_classes
    # initialization of the lists (values will be replaced)
    metrics_per_subj_per_c = {"dice1": [[-1] * n_classes for _ in range(n_subjects)],
                              "dice2": [[-1] * n_classes for _ in range(n_subjects)],
                              "dice3": [[-1] * n_classes for _ in range(n_subjects)]}
    
    for subj_i in range(n_subjects):
        log.print3("")
        log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.print3("~~~~~~~~\t Segmenting subject with index #" + str(subj_i) + " \t~~~~~~~~")
        
        (channels,  # nparray [channels,dim0,dim1,dim2]
         gt_lbl_img,
         roi_mask,
         _) = load_imgs_of_subject(log, "",
                                   subj_i,
                                   paths_per_chan_per_subj,
                                   paths_to_lbls_per_subj,
                                   None, # weightmaps, not for test
                                   paths_to_masks_per_subj)
        (channels,
        gt_lbl_img,
        roi_mask,
        _,
        pad_left_right_per_axis) = preproc_imgs_of_subj(log, "",
                                                        channels, gt_lbl_img, roi_mask, None,
                                                        run_input_checks, n_classes, # checks
                                                        pad_input, unpred_margin,
                                                        norm_prms)
    
        # ============== Augmentation ==================
        # TODO: Add augmentation here. And aggregate results after prediction of the whole volumes
        
        # ============== Predict whole volume ==================
        # array_fms_to_save will be None if not saving them.
        (prob_maps_vols,
         array_fms_to_save) = predict_whole_volume_by_tiling(log, sessionTf, cnn3d,
                                                             channels, roi_mask, inp_shapes_per_path, unpred_margin, 
                                                             batchsize, save_fms_flag, idxs_fms_to_save )
        
        # ========================== Post-Processing =========================
        pred_seg = np.argmax(prob_maps_vols, axis=0)  # The segmentation.

        # Unpad all images.        
        pred_seg_u          = unpad_img(pred_seg, pad_input, pad_left_right_per_axis)
        gt_lbl_u            = unpad_img(gt_lbl_img, pad_input, pad_left_right_per_axis)
        roi_mask_u          = unpad_img(roi_mask, pad_input, pad_left_right_per_axis)
        prob_maps_vols_u    = unpad_list_of_imgs(prob_maps_vols, pad_input, pad_left_right_per_axis)
        array_fms_to_save_u = unpad_list_of_imgs(array_fms_to_save, pad_input, pad_left_right_per_axis)
        
        # Poster-process outside the ROI, e.g. by deleting any predictions outside it.
        pred_seg_u_in_roi = pred_seg_u if roi_mask_u is None else pred_seg_u * roi_mask_u
        gt_lbl_u_in_roi = gt_lbl_u if (gt_lbl_u is None or roi_mask_u is None) else gt_lbl_u * roi_mask_u
        for c in range(n_classes):
            prob_map = prob_maps_vols_u[c]
            prob_maps_vols_u[c] = prob_map if roi_mask_u is None else prob_map * roi_mask_u
        prob_maps_vols_u_in_roi = prob_maps_vols_u # Just to follow naming convention for clarity.
        
        # ======================= Save Output Volumes ========================
        # Save predicted segmentations
        save_pred_seg(pred_seg_u_in_roi,
                      savePredictedSegmAndProbsDict["segm"], suffixForSegmAndProbsDict["segm"],
                      namesForSavingSegmAndProbs, paths_per_chan_per_subj, subj_i, log)

        # Save probability maps
        save_prob_maps(prob_maps_vols_u_in_roi,
                       savePredictedSegmAndProbsDict["prob"], suffixForSegmAndProbsDict["prob"],
                       namesForSavingSegmAndProbs, paths_per_chan_per_subj, subj_i, log)

        # Save feature maps
        save_fms_individual(save_fms_flag, array_fms_to_save_u, cnn3d.pathways, idxs_fms_to_save,
                            namesForSavingFms, paths_per_chan_per_subj, subj_i, log)
        
        
        # ================= Evaluate DSC for this subject ========================
        if paths_to_lbls_per_subj is not None:  # GT was provided.
            metrics_per_subj_per_c = calc_metrics_for_subject(metrics_per_subj_per_c, subj_i,
                                                              pred_seg_u, pred_seg_u_in_roi,
                                                              gt_lbl_u, gt_lbl_u_in_roi,
                                                              n_classes, NA_PATTERN)
            report_metrics_for_subject(log, metrics_per_subj_per_c, subj_i, NA_PATTERN, val_test_print)
            
        # Done with subject.
        
    # ==================== Report average Dice Coefficient over all subjects ==================
    mean_metrics = None # To return something even if ground truth has not been given (in testing)
    if paths_to_lbls_per_subj is not None and n_subjects > 0:  # GT was given. Calculate.
        mean_metrics = calc_stats_of_metrics(metrics_per_subj_per_c, NA_PATTERN)
        report_mean_metrics(log, mean_metrics, NA_PATTERN, val_test_print)

    log.print3("TIMING: " + val_test_print + " process lasted: {0:.2f}".format(time.time() - t_start) + " secs.")
    log.print3("##########################################################################################")
    log.print3("#\t\t  Finished full Segmentation of " + str(val_test_print) + " subjects   \t\t\t#")
    log.print3("##########################################################################################")

    return mean_metrics
