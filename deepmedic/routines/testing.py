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

from deepmedic.logging.accuracyMonitor import AccuracyOfEpochMonitorSegmentation
from deepmedic.dataManagement.sampling import load_imgs_of_subject
from deepmedic.dataManagement.sampling import getCoordsOfAllSegmentsOfAnImage
from deepmedic.dataManagement.sampling import extractSegmentsGivenSliceCoords
from deepmedic.dataManagement.io import savePredImgToNiiWithOriginalHdr, saveFmImgToNiiWithOriginalHdr, \
    save4DImgWithAllFmsToNiiWithOriginalHdr
from deepmedic.dataManagement.preprocessing import unpadCnnOutputs

from deepmedic.neuralnet.pathwayTypes import PathwayTypes as pt
from deepmedic.logging.utils import strListFl4fNA, getMeanPerColOf2dListExclNA


def calculate_dice(pred_labels, gt_labels):
    union_correct = pred_labels * gt_labels
    tp_num = np.sum(union_correct)
    gt_pos_num = np.sum(gt_labels)
    dice = (2.0 * tp_num) / (np.sum(pred_labels) + gt_pos_num) if gt_pos_num != 0 else -1
    return dice


def print_dice_explanations(log):
    log.print3(
        "EXPLANATION: DICE1/2/3 are lists with the DICE per class. For Class-0, we calculate DICE for whole foreground,"
        " i.e all labels merged, except the background label=0. Useful for multi-class problems.")
    log.print3(
        "EXPLANATION: DICE1 is calculated as segmentation over whole volume VS whole Ground Truth (GT). DICE2 is the "
        "segmentation within the ROI vs GT. DICE3 is segmentation within the ROI vs the GT within the ROI.")
    log.print3("EXPLANATION: If an ROI mask has been provided, you should be consulting DICE2 or DICE3.")


def find_num_fm(cnn3d_pathways, fm_idxs):
    fm_num = 0
    for pathway in cnn3d_pathways:
        fm_idxs_pathway = fm_idxs[pathway.pType()]
        if fm_idxs_pathway:
            for layer_i in range(len(pathway.getLayers())):
                fm_idxs_layer_pathway = fm_idxs_pathway[layer_i]
                if fm_idxs_layer_pathway:
                    # If the user specifies to grab more feature maps than exist (eg 9999),
                    # correct it, replacing it with the number of FMs in the layer.
                    fm_this_layer_num = pathway.getLayer(layer_i).getNumberOfFeatureMaps()
                    fm_idxs_layer_pathway[1] = min(fm_idxs_layer_pathway[1], fm_this_layer_num)
                    fm_num += fm_idxs_layer_pathway[1] - fm_idxs_layer_pathway[0]
    return fm_num


def construct_prob_maps(prob_maps_per_class, batch_size, slice_coords, half_rec_field, stride, prediction_test_batch,
                        img_part_idx):

    for image_part in range(batch_size):
        # Now put the label-cube in the new-label-segmentation-image, at the correct position.
        # The very first label goes not in index 0,0,0 but half-patch further away!
        # At the position of the central voxel of the top-left patch!
        slice_coords_segment = slice_coords[img_part_idx]
        top_left = [slice_coords_segment[0][0], slice_coords_segment[1][0], slice_coords_segment[2][0]]
        prob_maps_per_class[:,
                            top_left[0] + half_rec_field[0]: top_left[0] + half_rec_field[0] + stride[0],
                            top_left[1] + half_rec_field[1]: top_left[1] + half_rec_field[1] + stride[1],
                            top_left[2] + half_rec_field[2]: top_left[2] + half_rec_field[2] + stride[2]
                            ] = prediction_test_batch[image_part]
        img_part_idx += 1

    return img_part_idx, prob_maps_per_class


def calculate_num_voxels_sub(num_central_voxels, pathway):
    num_voxels_sub = np.zeros(3)

    for i in range(3):
        num_voxels_sub[i] = num_central_voxels[i] - 1 if pathway.pType() != pt.SUBS else \
            int(math.ceil((num_central_voxels[i] * 1.0) / pathway.subsFactor()[i]) - 1)

    return [int(a) for a in num_voxels_sub]


def calculate_num_central_voxels_dir(num_central_voxels, pathway):
    num_voxels_dir = np.zeros(3)

    # the math.ceil / subsamplingFactor is a trick to make it work for even subsamplingFactor too.
    # Eg 9/2=4.5 => Get 5. Combined with the trick at repeat,
    # I get my correct number of central voxels hopefully.

    for i in range(3):
        num_voxels_dir[i] = int(math.ceil((num_central_voxels[i] * 1.0) / pathway.subsFactor()[i])) \
            if pathway.pType() == pt.SUBS else int(num_central_voxels[i])

    return [int(a) for a in num_voxels_dir]


def construct_fms(fms_to_extract_img, img_part_idx, cnn3d_pathways, fm_idxs, fms_sorted, num_central_voxels,
                  half_rec_field, stride, slice_coords, batch_size):
    # idx_curr is the index in the
    # multidimensional array that holds all the to-be-visualised-fms. It is the one that corresponds to
    # the next to-be-visualised layer_idx.
    idx_curr = 0
    # layer_idx is the index over all the layers in the
    # returned list. I will work only with the ones specified to visualise.
    layer_idx = 0

    for pathway in cnn3d_pathways:
        for layer_i in range(len(pathway.getLayers())):
            if fm_idxs[pathway.pType()] == [] or fm_idxs[pathway.pType()][layer_i] == []:
                continue
            fms_to_extract_idxs = fm_idxs[pathway.pType()][layer_i]
            fms_layer = fms_sorted[layer_idx]
            # We specify a range of fms to visualise from a layer.
            # curr_idx : fms_to_fill_high_idx defines were to put them in the multidimensional-image-array.
            fms_to_fill_high_idx = idx_curr + fms_to_extract_idxs[1] - fms_to_extract_idxs[0]

            fm_to_reconstruct = fms_to_extract_img[idx_curr:fms_to_fill_high_idx]

            # ========================================================================================
            # ====the following calculations could be move OUTSIDE THE FOR LOOPS, by using the kernel-size
            # parameter (from the cnn instance) instead of the shape of the returned value.
            # ====fmsReturnedForATestBatchForCertainLayer.shape[2] - (num_central_voxels[0]-1)
            # is essentially the width of the patch left after the convolutions.
            # ====These calculations are pathway and layer-specific. So they could be done once, prior to
            # image processing, and results cached in a list to be accessed during the loop.

            (num_voxels_sub_r,
             num_voxels_sub_c,
             num_voxels_sub_z) = calculate_num_voxels_sub(num_central_voxels, pathway)

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
             num_central_voxels_z) = calculate_num_central_voxels_dir(num_central_voxels, pathway)

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
                expanded_output_r = np.repeat(central_voxels_all_fms, pathway.subsFactor()[0], axis=2)
                expanded_output_rc = np.repeat(expanded_output_r, pathway.subsFactor()[1], axis=3)
                expanded_output_rcz = np.repeat(expanded_output_rc, pathway.subsFactor()[2], axis=4)
                # The below is a trick to get correct number of voxels even when subsampling factor is
                # even or not exact divisor of the number of central voxels.
                # ...This trick is coupled with the ceil() when getting the
                # numberOfCentralVoxelsToGetInDirectionR above.
                central_voxels_all_fms_batch = expanded_output_rcz[:,
                                                                   :,
                                                                   0:num_central_voxels[0],
                                                                   0:num_central_voxels[1],
                                                                   0:num_central_voxels[2]
                                                                   ]
            else:
                central_voxels_all_fms_batch = central_voxels_all_fms

            # ----For every image part within this batch, reconstruct the corresponding part of the feature
            # maps of the layer we are currently visualising in this loop.
            for img_part_batch_idx in range(batch_size):
                # Now put the label-cube in the new-label-segmentation-image, at the correct position.
                # The very first label goes not in index 0,0,0 but half-patch further away! At the position
                # of the central voxel of the top-left patch!
                slice_coords_seg = slice_coords[
                    img_part_idx + img_part_batch_idx]
                coords_top_left_voxel = [slice_coords_seg[0][0],
                                         slice_coords_seg[1][0],
                                         slice_coords_seg[2][0]]

                # I put the central-predicted-voxels of all FMs to the corresponding,
                # newly created images all at once.
                fm_to_reconstruct[:,  # last dimension is the number-of-Fms, I create an image for each.

                                  coords_top_left_voxel[0] + half_rec_field[0]:
                                  coords_top_left_voxel[0] + half_rec_field[0] + stride[0],

                                  coords_top_left_voxel[1] + half_rec_field[1]:
                                  coords_top_left_voxel[1] + half_rec_field[1] + stride[1],

                                  coords_top_left_voxel[2] + half_rec_field[2]:
                                  coords_top_left_voxel[2] + half_rec_field[2] + stride[2]

                                  ] = central_voxels_all_fms_batch[img_part_batch_idx]

            idx_curr = fms_to_fill_high_idx

            layer_idx += 1

    # all the image parts before this were reconstructed for all layers and feature maps.
    # Next batch-iteration should start from this
    img_part_idx += batch_size

    return img_part_idx, fms_to_extract_img


def print_progress_step(log, num_batches, batch_i, batch_size, num_segments_for_case):
    progress_step = max(1, num_batches // 5)

    if batch_i == 0 or ((batch_i + 1) % progress_step) == 0 or (batch_i + 1) == num_batches:
        log.print3(
            "Processed " + str((batch_i + 1) * batch_size) + "/" + str(num_segments_for_case) + " segments.")


def load_feed_dict(feeds, channels_segments_per_path, loading_time):
    start_loading_time = time.time()

    feeds_dict = {feeds['x']: np.asarray(channels_segments_per_path[0], dtype='float32')}
    for path_i in range(len(channels_segments_per_path[1:])):
        feeds_dict.update(
            {feeds['x_sub_' + str(path_i)]: np.asarray(channels_segments_per_path[1 + path_i], dtype='float32')})

    end_loading_time = time.time()
    loading_time += end_loading_time - start_loading_time

    return feeds_dict, loading_time


def forward_pass(session, list_of_ops, feeds_dict, fwd_pass_time):
    start_testing_time = time.time()

    # Forward pass
    # featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = \
    #     cnn3d.cnnTestAndVisualiseAllFmsFunction( *input_args_to_net )
    fms_and_pred_probs = session.run(fetches=list_of_ops, feed_dict=feeds_dict)

    end_testing_time = time.time()
    fwd_pass_time += end_testing_time - start_testing_time

    return fms_and_pred_probs, fwd_pass_time


def get_unpadded_pred_seg(prob_maps_per_class, pad_input_imgs, axes_padding_left_right):
    pred_seg = np.argmax(prob_maps_per_class, axis=0)  # The segmentation.
    unpadded_pred_seg = pred_seg if not pad_input_imgs else \
        unpadCnnOutputs(pred_seg, axes_padding_left_right)

    return unpadded_pred_seg


def get_unpadded_roi_mask(roi_mask, pad_input_imgs, axes_padding_left_right):
    # Multiply with the below to zero-out anything outside the RoiMask if given.
    # Provided that RoiMask is binary [0,1].
    unpadded_roi_mask = 1
    if isinstance(roi_mask, np.ndarray):  # If roiMask was given:
        unpadded_roi_mask = roi_mask if not pad_input_imgs else \
            unpadCnnOutputs(roi_mask, axes_padding_left_right)

    return unpadded_roi_mask


def save_pred_seg(unpadded_pred_seg, unpadded_roi_mask, save_pred_seg_bool, suffix_seg, seg_names, filepaths,
                  image_i, log):

    if save_pred_seg_bool:  # save predicted segmentation
        # Save the image. Pass the filename paths of the normal image so that I can duplicate the header info,
        # eg RAS transformation
        unpadded_pred_seg_in_roi = unpadded_pred_seg * unpadded_roi_mask
        savePredImgToNiiWithOriginalHdr(unpadded_pred_seg_in_roi,
                                        seg_names,
                                        filepaths,
                                        image_i,
                                        suffix_seg,
                                        np.dtype(np.int16),
                                        log)


def save_prob_maps(num_classes, save_prob_maps_bool, suffix_prob_map, prob_maps,
                   pad_input_imgs, axes_padding_left_right, unpadded_roi_mask,
                   prob_names, filepaths, image_i, log):

    for class_i in range(0, num_classes):
        if (len(save_prob_maps_bool) >= class_i + 1) and (save_prob_maps_bool[class_i]):  # save per class probMap
            suffix = suffix_prob_map + str(class_i)
            # Save the image. Pass the filename paths of the normal image so that I can duplicate the header info,
            # eg RAS transformation.
            prob_map_i = prob_maps[class_i, :, :, :]
            unpadded_prob_map_i = prob_map_i if not pad_input_imgs \
                else unpadCnnOutputs(prob_map_i, axes_padding_left_right)
            unpadded_prob_map_i_in_roi = unpadded_prob_map_i * unpadded_roi_mask
            savePredImgToNiiWithOriginalHdr(unpadded_prob_map_i_in_roi,
                                            prob_names,
                                            filepaths,
                                            image_i,
                                            suffix,
                                            np.dtype(np.float32),
                                            log)


def save_fms_individual(cnn3d_pathways, fm_idxs,
                        multidim_fm_array, pad_input_imgs,
                        axes_padding_left_right, fms_names, filepaths,
                        image_i, log):
    idx_curr = 0
    for pathway_i in range(len(cnn3d_pathways)):
        pathway = cnn3d_pathways[pathway_i]
        fms_idx_pathway = fm_idxs[pathway.pType()]
        if fms_idx_pathway:
            for layer_i in range(len(pathway.getLayers())):
                fms_idx_layer_pathway = fms_idx_pathway[layer_i]
                if fms_idx_layer_pathway:
                    # If the user specifies to grab more feature maps than exist (eg 9999), correct it,
                    # replacing it with the number of FMs in the layer.
                    for fmActualNumber in range(fms_idx_layer_pathway[0], fms_idx_layer_pathway[1]):
                        fm_to_save = multidim_fm_array[idx_curr]
                        unpadded_fm_to_save = fm_to_save if not pad_input_imgs else \
                            unpadCnnOutputs(fm_to_save, axes_padding_left_right)

                        saveFmImgToNiiWithOriginalHdr(unpadded_fm_to_save,
                                                      fms_names,
                                                      filepaths,
                                                      image_i,
                                                      pathway_i,
                                                      layer_i,
                                                      fmActualNumber,
                                                      log)

                        idx_curr += 1


def save_fms_multidim(multidim_fm_array, pad_input_imgs,
                      axes_padding_left_right, fms_names, filepaths,
                      image_i, log):
    multidim_fm_array_4dim = np.transpose(multidim_fm_array, (1, 2, 3, 0))
    unpadded_multidim_fm_array_4dim = multidim_fm_array_4dim if not pad_input_imgs else \
        unpadCnnOutputs(multidim_fm_array_4dim, axes_padding_left_right)
    # Save a multidimensional Nii image. 3D Image, with the 4th dimension being all the Fms...
    save4DImgWithAllFmsToNiiWithOriginalHdr(
        unpadded_multidim_fm_array_4dim,
        fms_names,
        filepaths,
        image_i,
        log)


def calculate_dsc(dices_1, dices_2, dices_3, gt_labels, unpadded_pred_seg,
                  pad_input_imgs, axes_padding_left_right, unpadded_roi_mask,
                  num_classes, na_pattern, log, image_i, validation_or_testing_str):

    log.print3("+++++++++++++++++++++ Reporting Segmentation Metrics for the subject #" + str(image_i) +
               " ++++++++++++++++++++++++++")
    # Unpad whatever needed.
    unpadded_gt_labels = gt_labels if not pad_input_imgs else unpadCnnOutputs(gt_labels, axes_padding_left_right)

    # calculate DSC per class.
    for class_i in range(0, num_classes):
        if class_i == 0:  # do the eval for WHOLE FOREGROUND segmentation (all classes merged except background)
            # Merge every class except the background (assumed to be label == 0 )
            pred_seg_binary_i = unpadded_pred_seg > 0
            gt_labels_binary_i = unpadded_gt_labels > 0
        else:
            pred_seg_binary_i = unpadded_pred_seg == class_i
            gt_labels_binary_i = unpadded_gt_labels == class_i

        pred_seg_binary_i_in_roi = pred_seg_binary_i * unpadded_roi_mask

        # Calculate the 3 Dices.
        # Dice1 = Allpredicted/allLesions,
        # Dice2 = PredictedWithinRoiMask / AllLesions ,
        # Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask.

        # Dice1 = Allpredicted/allLesions
        dice_1 = calculate_dice(pred_seg_binary_i, gt_labels_binary_i)
        dices_1[image_i][class_i] = dice_1 if dice_1 != -1 else na_pattern

        # Dice2 = PredictedWithinRoiMask / AllLesions
        dice_2 = calculate_dice(pred_seg_binary_i_in_roi, gt_labels_binary_i)
        dices_2[image_i][class_i] = dice_2 if dice_2 != -1 else na_pattern

        # Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask
        dice_3 = calculate_dice(pred_seg_binary_i_in_roi, gt_labels_binary_i * unpadded_roi_mask)
        dices_3[image_i][class_i] = dice_3 if dice_3 != -1 else na_pattern

    log.print3("ACCURACY: (" + str(validation_or_testing_str) +
               ") The Per-Class DICE Coefficients for subject with index #" + str(image_i) +
               " equal: DICE1=" + strListFl4fNA(dices_1[image_i], na_pattern) +
               " DICE2=" + strListFl4fNA(dices_2[image_i], na_pattern) +
               " DICE3=" + strListFl4fNA(dices_3[image_i], na_pattern))

    print_dice_explanations(log)

    return dices_1, dices_2, dices_3


def calculate_mean_dsc(log, dices_1, dices_2, dices_3, na_pattern, validation_or_testing_str):
    log.print3(
        "+++++++++++++++++++++++++++++++ Segmentation of all subjects finished +++++++++++++++++++++++++++++++++++")
    log.print3(
        "+++++++++++++++++++++ Reporting Average Segmentation Metrics over all subjects ++++++++++++++++++++++++++")

    mean_dice_1 = getMeanPerColOf2dListExclNA(dices_1, na_pattern)
    mean_dice_2 = getMeanPerColOf2dListExclNA(dices_2, na_pattern)
    mean_dice_3 = getMeanPerColOf2dListExclNA(dices_3, na_pattern)

    log.print3("ACCURACY: (" + str(validation_or_testing_str) +
               ") The Per-Class average DICE Coefficients over all subjects are: DICE1=" +
               strListFl4fNA(mean_dice_1, na_pattern) +
               " DICE2=" + strListFl4fNA(mean_dice_2, na_pattern) +
               " DICE3=" + strListFl4fNA(mean_dice_3, na_pattern))

    print_dice_explanations(log)

    return mean_dice_1, mean_dice_2, mean_dice_3


def print_time_log(log, extract_time, loading_time, fwd_pass_time):
    log.print3("TIMING: Segmentation of subject: [Extracting:] {0:.2f}".format(extract_time) +
               " [Loading:] {0:.2f}".format(loading_time) +
               " [ForwardPass:] {0:.2f}".format(fwd_pass_time) +
               " [Total:] {0:.2f}".format(
                   extract_time + loading_time + fwd_pass_time) + " secs.")


def dsc_to_dict(dices_1, dices_2, dices_3):
    metrics_dict_list = []
    for i in range(len(dices_1)):
        metrics_dict_list.append({'mean_dice1': dices_1[i],
                                  'mean_dice2': dices_2[i],
                                  'mean_dice3': dices_3[i]})

    return metrics_dict_list


# Main routine for testing.
def inferenceWholeVolumes(sessionTf,
                          cnn3d,
                          log,
                          val_or_test,
                          savePredictedSegmAndProbsDict,
                          listOfFilepathsToEachChannelOfEachPatient,
                          listOfFilepathsToGtLabelsOfEachPatient,
                          listOfFilepathsToRoiMaskFastInfOfEachPatient,
                          namesForSavingSegmAndProbs,
                          suffixForSegmAndProbsDict,
                          # Hyper parameters
                          batchsize,

                          # ----Preprocessing------
                          pad_input_imgs,

                          # --------For FM visualisation---------
                          saveIndividualFmImagesForVisualisation,
                          saveMultidimensionalImageWithAllFms,
                          indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                          namesForSavingFms):
    # saveIndividualFmImagesForVisualisation: should contain an entry per pathwayType, even if just []...
    #       ... If not [], the list should contain one entry per layer of the pathway, even if just [].
    #       ... The layer entries, if not [], they should have to integers, lower and upper FM to visualise.
    #       ... Excluding the highest index.

    validation_or_testing_str = "Validation" if val_or_test == "val" else "Testing"

    log.print3(
        "###########################################################################################################")
    log.print3("############################# Starting full Segmentation of " +
               str(validation_or_testing_str) + " subjects ##########################")
    log.print3(
        "###########################################################################################################")

    start_time = time.time()

    NA_PATTERN = AccuracyOfEpochMonitorSegmentation.NA_PATTERN

    NUMBER_OF_CLASSES = cnn3d.num_classes

    total_number_of_images = len(listOfFilepathsToEachChannelOfEachPatient)

    # one dice score for whole + for each class)
    # Dice1 - AllpredictedLes/AllLesions
    # Dice2 - predictedInsideRoiMask/AllLesions
    # Dice3 - predictedInsideRoiMask/ LesionsInsideRoiMask (for comparisons)
    # A list of dimensions: total_number_of_images X NUMBER_OF_CLASSES
    # init dice scores
    diceCoeffs1 = [[-1] * NUMBER_OF_CLASSES for _ in range(total_number_of_images)]
    diceCoeffs2 = [[-1] * NUMBER_OF_CLASSES for _ in range(total_number_of_images)]
    diceCoeffs3 = [[-1] * NUMBER_OF_CLASSES for _ in range(total_number_of_images)]

    recFieldCnn = cnn3d.recFieldCnn

    # stride is how much I move in each dimension to acquire the next imagePart.
    # I move exactly the number I segment in the centre of each image part
    # (originally this was 9^3 segmented per imagePart).
    numberOfCentralVoxelsClassified = cnn3d.finalTargetLayer.outputShape["test"][2:]
    strideOfImagePartsPerDimensionInVoxels = numberOfCentralVoxelsClassified

    rczHalfRecFieldCnn = [(recFieldCnn[i] - 1) // 2 for i in range(3)]

    # Find the total number of feature maps that will be created:
    # NOTE: saveIndividualFmImagesForVisualisation should contain an entry per pathwayType, even if just [].
    # If not [], the list should contain one entry per layer of the pathway, even if just [].
    # The layer entries, if not [], they should have to integers, lower and upper FM to visualise.
    if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
        totalNumberOfFMsToProcess = find_num_fm(cnn3d.pathways, indicesOfFmsToVisualisePerPathwayTypeAndPerLayer)

    for image_i in range(total_number_of_images):

        log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.print3("~~~~~~~~~~~~~~~~~~~~ Segmenting subject with index #" + str(image_i) + " ~~~~~~~~~~~~~~~~~~~~")

        # load the image channels in cpu
        (imageChannels,
         gtLabelsImage,  # only for accurate/correct DICE1-2 calculation
         roiMask,
         arrayWithWeightMapsWhereToSampleForEachCategory,  # only used in training. Placeholder here.
         tupleOfPaddingPerAxesLeftRight  # ((padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)).
         # ^ All 0s when no padding.
         ) = load_imgs_of_subject(
            log,
            None,
            "test",
            False,  # run_input_checks.
            image_i,
            listOfFilepathsToEachChannelOfEachPatient,
            listOfFilepathsToGtLabelsOfEachPatient,
            None,
            listOfFilepathsToRoiMaskFastInfOfEachPatient,
            cnn3d.num_classes,
            pad_input_imgs,
            recFieldCnn,  # only used if padInputsBool
            cnn3d.pathways[0].getShapeOfInput("test")[2:]  # dimsOfPrimeSegmentRcz, for padding
        )

        niiDimensions = list(imageChannels.shape[1:])
        # The predicted probability-maps for the whole volume, one per class.
        # Will be constructed by stitching together the predictions from each segment.
        predProbMapsPerClass = np.zeros([NUMBER_OF_CLASSES]+niiDimensions, dtype="float32")
        # create the big array that will hold all the fms (for feature extraction, to save as a big multi-dim image).
        if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
            multidimensionalImageWithAllToBeVisualisedFmsArray = np.zeros([totalNumberOfFMsToProcess] + niiDimensions,
                                                                          dtype="float32")

        # Tile the image and get all slices of the segments that it fully breaks down to.
        sliceCoordsOfSegmentsInImage = getCoordsOfAllSegmentsOfAnImage(log,
                                                                       cnn3d.pathways[0].getShapeOfInput("test")[2:],
                                                                       # dimsOfPrimarySegment
                                                                       strideOfImagePartsPerDimensionInVoxels,
                                                                       batchsize,
                                                                       imageChannels,
                                                                       roiMask)

        num_segments_for_case = len(sliceCoordsOfSegmentsInImage)
        log.print3(
            "Starting to segment each image-part by calling the cnn.cnnTestModel(i). "
            "This part takes a few mins per volume...")

        log.print3("Total number of Segments to process:" + str(num_segments_for_case))

        imagePartOfConstructedProbMap_i = 0
        imagePartOfConstructedFeatureMaps_i = 0
        num_batches = num_segments_for_case // batchsize
        extractTimePerSubject = 0
        loadingTimePerSubject = 0
        fwdPassTimePerSubject = 0
        for batch_i in range(num_batches):

            print_progress_step(log, num_batches, batch_i, batchsize, num_segments_for_case)

            # Extract the data for the segments of this batch.
            # ( I could modularize extractDataOfASegmentFromImagesUsingSampledSliceCoords()
            # of training and use it here as well. )
            start_extract_time = time.time()

            sliceCoordsOfSegmentsInBatch = sliceCoordsOfSegmentsInImage[batch_i * batchsize: (batch_i + 1) * batchsize]
            channsOfSegmentsPerPath = extractSegmentsGivenSliceCoords(cnn3d,
                                                                      sliceCoordsOfSegmentsInBatch,
                                                                      imageChannels,
                                                                      recFieldCnn)
            end_extract_time = time.time()
            extractTimePerSubject += end_extract_time - start_extract_time

            # ======= Run the inference ============

            ops_to_fetch = cnn3d.get_main_ops('test')
            list_of_ops = [ops_to_fetch['pred_probs']] + ops_to_fetch['list_of_fms_per_layer']

            # No loading of data in bulk as in training, cause here it's only 1 batch per iteration.
            (feeds_dict, loadingTimePerSubject) = load_feed_dict(cnn3d.get_main_feeds('test'),
                                                                 channsOfSegmentsPerPath, loadingTimePerSubject)

            # Forward pass
            (featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch, fwdPassTimePerSubject) = \
                forward_pass(sessionTf, list_of_ops, feeds_dict, fwdPassTimePerSubject)

            predictionForATestBatch = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[0]

            # If no FMs visualised, this should return []
            listWithTheFmsOfAllLayersSortedByPathwayTypeForTheBatch = \
                featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[1:]
            # No reshape needed, because I now do it internally. But to dimensions (batchSize, FMs, R,C,Z).

            # ~~~~~~~~~~~~~~~~CONSTRUCT THE PREDICTED PROBABILITY MAPS~~~~~~~~~~~~~~
            # From the results of this batch, create the prediction image by putting the predictions to the correct
            # place in the image.
            (imagePartOfConstructedProbMap_i,
             predProbMapsPerClass) = construct_prob_maps(predProbMapsPerClass,
                                                         batchsize,
                                                         sliceCoordsOfSegmentsInImage,
                                                         rczHalfRecFieldCnn,
                                                         strideOfImagePartsPerDimensionInVoxels,
                                                         predictionForATestBatch,
                                                         imagePartOfConstructedProbMap_i)
            # ~~~~~~~~~~~~~FINISHED CONSTRUCTING THE PREDICTED PROBABILITY MAPS~~~~~~~

            # ~~~~~~~~~~~~~~CONSTRUCT THE FEATURE MAPS FOR VISUALISATION~~~~~~~~~~~~~~~~~
            if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
                (imagePartOfConstructedFeatureMaps_i,
                 multidimensionalImageWithAllToBeVisualisedFmsArray) = \
                    construct_fms(multidimensionalImageWithAllToBeVisualisedFmsArray,
                                  imagePartOfConstructedFeatureMaps_i,
                                  cnn3d.pathways,
                                  indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                                  listWithTheFmsOfAllLayersSortedByPathwayTypeForTheBatch,
                                  numberOfCentralVoxelsClassified,
                                  rczHalfRecFieldCnn,
                                  strideOfImagePartsPerDimensionInVoxels,
                                  sliceCoordsOfSegmentsInImage,
                                  batchsize)

            # ~~~~~~~~~~~~~~~~~~FINISHED CONSTRUCTING THE FEATURE MAPS FOR VISUALISATION~~~~~~~~~~

        print_time_log(log, extractTimePerSubject, loadingTimePerSubject, fwdPassTimePerSubject)

        # ================ SAVE PREDICTIONS =====================
        # == saving predicted segmentations ==
        unpaddedPredSegmentation = get_unpadded_pred_seg(predProbMapsPerClass,
                                                         pad_input_imgs, tupleOfPaddingPerAxesLeftRight)
        unpaddedRoiMaskIfGivenElse1 = get_unpadded_roi_mask(roiMask, pad_input_imgs, tupleOfPaddingPerAxesLeftRight)

        save_pred_seg(unpaddedPredSegmentation, unpaddedRoiMaskIfGivenElse1,
                      savePredictedSegmAndProbsDict["segm"], suffixForSegmAndProbsDict["segm"],
                      namesForSavingSegmAndProbs, listOfFilepathsToEachChannelOfEachPatient, image_i, log)

        # == saving probability maps ==
        save_prob_maps(NUMBER_OF_CLASSES, savePredictedSegmAndProbsDict["prob"], suffixForSegmAndProbsDict["prob"],
                       predProbMapsPerClass, pad_input_imgs, tupleOfPaddingPerAxesLeftRight,
                       unpaddedRoiMaskIfGivenElse1, namesForSavingSegmAndProbs,
                       listOfFilepathsToEachChannelOfEachPatient, image_i, log)

        # == saving feature maps ==
        if saveIndividualFmImagesForVisualisation:
            save_fms_individual(cnn3d.pathways, indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                                multidimensionalImageWithAllToBeVisualisedFmsArray, pad_input_imgs,
                                tupleOfPaddingPerAxesLeftRight, namesForSavingFms,
                                listOfFilepathsToEachChannelOfEachPatient,
                                image_i, log)

        if saveMultidimensionalImageWithAllFms:
            save_fms_multidim(multidimensionalImageWithAllToBeVisualisedFmsArray, pad_input_imgs,
                              tupleOfPaddingPerAxesLeftRight, namesForSavingFms,
                              listOfFilepathsToEachChannelOfEachPatient,
                              image_i, log)
        # ================= FINISHED SAVING RESULTS ====================

        # ================= EVALUATE DSC FOR EACH SUBJECT ========================
        if listOfFilepathsToGtLabelsOfEachPatient is not None:  # GT was provided for DSC calculation. Do calculation
            (diceCoeffs1, diceCoeffs2, diceCoeffs3) = calculate_dsc(diceCoeffs1, diceCoeffs2, diceCoeffs3,
                                                                    gtLabelsImage, unpaddedPredSegmentation,
                                                                    pad_input_imgs, tupleOfPaddingPerAxesLeftRight,
                                                                    unpaddedRoiMaskIfGivenElse1, NUMBER_OF_CLASSES,
                                                                    NA_PATTERN, log, image_i, validation_or_testing_str)

    # = Loops for all patients have finished. Now lets just report the average DSC over all the processed patients. =
    if listOfFilepathsToGtLabelsOfEachPatient is not None and total_number_of_images > 0:  # GT was given. Calculate
        (meanDiceCoeffs1, meanDiceCoeffs2, meanDiceCoeffs3) = calculate_mean_dsc(log,
                                                                                 diceCoeffs1, diceCoeffs2, diceCoeffs3,
                                                                                 NA_PATTERN, validation_or_testing_str)
        metrics_dict_list = dsc_to_dict(meanDiceCoeffs1, meanDiceCoeffs2, meanDiceCoeffs3)
    else:
        metrics_dict_list = {}

    end_time = time.time()

    log.print3(
        "TIMING: " + validation_or_testing_str + " process lasted: {0:.2f}".format(end_time - start_time) + " secs.")
    log.print3(
        "###########################################################################################################")
    log.print3("############################# Finished full Segmentation of " +
               str(validation_or_testing_str) +
               " subjects ##########################")
    log.print3(
        "###########################################################################################################")

    return metrics_dict_list
