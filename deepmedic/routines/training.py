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
from multiprocessing.pool import ThreadPool
import traceback

import numpy as np

from deepmedic.logging.accuracyMonitor import AccuracyMonitorForEpSegm
from deepmedic.neuralnet.wrappers import CnnWrapperForSampling
from deepmedic.dataManagement.sampling import getSampledDataAndLabelsForSubepoch
from deepmedic.routines.testing import inference_on_whole_volumes

from deepmedic.logging.utils import datetimeNowAsStr


# The main subroutine of do_training, that runs for every batch of validation and training.
def trainOrValidateForSubepoch(log,
                               sessionTf,
                               train_or_val,
                               num_batches,
                               batchsize,
                               cnn3d,
                               acc_monitor_ep,
                               channsOfSegmentsForSubepPerPathway,
                               labelsForCentralOfSegmentsForSubep):
    """
    Returned array is of dimensions [NumberOfClasses x 6]
    For each class: [mean acc of subep, mean acc on positives, mean acc on negatives, mean DSC of subep, mean cost of subep]
    In the case of VALIDATION, mean_cost_subep is just a placeholder. Only valid when training.
    """

    costs_of_batches = []
    # Each row of array below holds number of:
    #     Real Positives, Real Neg, True Predicted Pos, True Predicted Neg in subepoch, in this order.
    array_per_class_RpRnTpTn_in_subep = np.zeros([cnn3d.num_classes, 4], dtype="int32")

    print_progress_step = max(1, num_batches // 5)

    for batch_i in range(num_batches):

        if train_or_val == "train":
            if batch_i == 0 or ((batch_i + 1) % print_progress_step) == 0 or (batch_i + 1) == num_batches:
                log.print3("[TRAINING] Trained on " + str(batch_i + 1) + "/" + str(
                    num_batches) + " of the batches for this subepoch...")

            ops_to_fetch = cnn3d.get_main_ops('train')
            list_of_ops = [ops_to_fetch['cost']] + ops_to_fetch['list_rp_rn_tp_tn'] + [
                ops_to_fetch['updates_grouped_op']]

            index_to_data_for_batch_min = batch_i * batchsize
            index_to_data_for_batch_max = (batch_i + 1) * batchsize

            feeds = cnn3d.get_main_feeds('train')
            feeds_dict = {feeds['x']: channsOfSegmentsForSubepPerPathway[0][
                                      index_to_data_for_batch_min: index_to_data_for_batch_max]}
            for subs_path_i in range(cnn3d.numSubsPaths):
                feeds_dict.update({feeds['x_sub_' + str(subs_path_i)]: channsOfSegmentsForSubepPerPathway[
                                                                           subs_path_i + 1][
                                                                       index_to_data_for_batch_min: index_to_data_for_batch_max]})
            feeds_dict.update({feeds['y_gt']: labelsForCentralOfSegmentsForSubep[
                                              index_to_data_for_batch_min: index_to_data_for_batch_max]})
            # Training step. Returns a list containing the results of fetched ops.
            results_of_run = sessionTf.run(fetches=list_of_ops, feed_dict=feeds_dict)

            cnn3d.updateMatricesOfBnMovingAvForInference(sessionTf)  # I should put this inside the 3dCNN.

            cost_this_batch = results_of_run[0]
            list_RpRnPpPn_per_class = results_of_run[1:-1]  # [-1] is from updates_grouped_op, returns nothing

        else:  # validation
            if batch_i == 0 or ((batch_i + 1) % print_progress_step) == 0 or (batch_i + 1) == num_batches:
                log.print3("[VALIDATION] Validated on " +
                           str(batch_i + 1) + "/" + str(num_batches) +
                           " of the batches for this subepoch...")

            ops_to_fetch = cnn3d.get_main_ops('val')
            list_of_ops = ops_to_fetch['list_rp_rn_tp_tn']

            index_to_data_for_batch_min = batch_i * batchsize
            index_to_data_for_batch_max = (batch_i + 1) * batchsize

            feeds = cnn3d.get_main_feeds('val')
            feeds_dict = {feeds['x']: channsOfSegmentsForSubepPerPathway[0][
                                      index_to_data_for_batch_min: index_to_data_for_batch_max]}
            for subs_path_i in range(cnn3d.numSubsPaths):
                feeds_dict.update({feeds['x_sub_' + str(subs_path_i)]: channsOfSegmentsForSubepPerPathway[
                                                                           subs_path_i + 1][
                                                                       index_to_data_for_batch_min: index_to_data_for_batch_max]})
            feeds_dict.update({feeds['y_gt']: labelsForCentralOfSegmentsForSubep[
                                              index_to_data_for_batch_min: index_to_data_for_batch_max]})
            # Validation step. Returns a list containing the results of fetched ops.
            results_of_run = sessionTf.run(fetches=list_of_ops, feed_dict=feeds_dict)

            cost_this_batch = 999  # placeholder in case of validation.
            list_RpRnPpPn_per_class = results_of_run

        # list_RpRnPpPn_per_class holds Real Pos, Real Neg, True Pred Pos, True Pred Neg ...
        # ... for all classes, in this order, flattened. First RpRnTpTn are for 'WHOLE' class.
        array_per_class_RpRnTpTn_in_batch = np.asarray(list_RpRnPpPn_per_class, dtype="int32").reshape(
            array_per_class_RpRnTpTn_in_subep.shape, order='C')

        # To later calculate the mean error and cost over the subepoch
        costs_of_batches.append(cost_this_batch)  # only really used in training.
        array_per_class_RpRnTpTn_in_subep += array_per_class_RpRnTpTn_in_batch

    # ======== Calculate and Report accuracy over subepoch
    # In case of validation, mean_cost_subep is just a placeholder.
    # Cause this does not get calculated and reported in this case.
    mean_cost_subep = acc_monitor_ep.NA_PATTERN if (train_or_val == "val") else sum(costs_of_batches) / float(
        num_batches)
    # This function does NOT flip the class-0 background to foreground!
    acc_monitor_ep.update_metrics_after_subep(mean_cost_subep, array_per_class_RpRnTpTn_in_subep)
    acc_monitor_ep.log_acc_subep_to_txt()
    acc_monitor_ep.log_acc_subep_to_tensorboard()
    # Done


# ------------------------------ MAIN TRAINING ROUTINE -------------------------------------
def do_training(sessionTf,
                saver_all,
                cnn3d,
                trainer,
                tensorboard_loggers,
                
                log,
                fileToSaveTrainedCnnModelTo,

                val_on_samples,
                savePredictedSegmAndProbsDict,

                namesForSavingSegmAndProbs,
                suffixForSegmAndProbsDict,

                paths_per_chan_per_subj_train,
                paths_per_chan_per_subj_val,

                paths_to_lbls_per_subj_train,
                paths_to_lbls_per_subj_val,

                paths_to_wmaps_per_sampl_cat_per_subj_train,
                paths_to_wmaps_per_sampl_cat_per_subj_val,

                paths_to_masks_per_subj_train,
                paths_to_masks_per_subj_val,

                n_epochs,  # Every epoch the CNN model is saved.
                n_subepochs,  # per epoch. Every subepoch Accuracy is reported
                max_n_cases_per_subep_train,  # Max num of subjects loaded every subep for sampling
                n_samples_per_subep_train,
                n_samples_per_subep_val,
                num_parallel_proc_sampling,  # -1: seq. 0: thread for sampling. >0: multiprocess sampling

                # -------Sampling Type---------
                sampling_type_inst_tr,
                # Instance of the deepmedic/samplingType.SamplingType class for training and validation
                sampling_type_inst_val,
                batchsize_train,
                batchsize_val_samples,
                batchsize_val_whole,

                # -------Data Augmentation-------
                augm_img_prms,
                augm_sample_prms,

                # Validation
                val_on_whole_volumes,
                n_epochs_between_val_on_whole_vols,

                # --------For FM visualisation---------
                save_fms_flag,
                idxs_fms_to_save,
                namesForSavingFms,

                # --- Data Compatibility Checks ---
                run_input_checks,

                # -------- Pre-processing ------
                pad_input,
                norm_prms
                ):
    id_str = "[MAIN|PID:" + str(os.getpid()) + "]"
    start_time_train = time.time()

    # I cannot pass cnn3d to the sampling function, because the pp module used to reload theano. 
    # This created problems in the GPU when cnmem is used. Not sure this is needed with Tensorflow. Probably.
    cnn3dWrapper = CnnWrapperForSampling(cnn3d)

    args_for_sampling_train = (log,
                               "train",
                               num_parallel_proc_sampling,
                               run_input_checks,
                               cnn3dWrapper,
                               max_n_cases_per_subep_train,
                               n_samples_per_subep_train,
                               sampling_type_inst_tr,
                               paths_per_chan_per_subj_train,
                               paths_to_lbls_per_subj_train,
                               paths_to_masks_per_subj_train,
                               paths_to_wmaps_per_sampl_cat_per_subj_train,
                               pad_input,
                               norm_prms,
                               augm_img_prms,
                               augm_sample_prms
                               )
    args_for_sampling_val = (log,
                             "val",
                             num_parallel_proc_sampling,
                             run_input_checks,
                             cnn3dWrapper,
                             max_n_cases_per_subep_train,
                             n_samples_per_subep_val,
                             sampling_type_inst_val,
                             paths_per_chan_per_subj_val,
                             paths_to_lbls_per_subj_val,
                             paths_to_masks_per_subj_val,
                             paths_to_wmaps_per_sampl_cat_per_subj_val,
                             pad_input,
                             norm_prms,
                             None,  # no augmentation in val.
                             None  # no augmentation in val.
                             )

    sampling_job_submitted_train = False
    sampling_job_submitted_val = False
    # For parallel extraction of samples for next train/val while processing previous iteration.
    worker_pool = None
    if num_parallel_proc_sampling > -1:  # Use multiprocessing.
        worker_pool = ThreadPool(processes=1)  # Or multiprocessing.Pool(...), same API.

    try:
        n_eps_trained_model = trainer.get_num_epochs_trained_tfv().eval(session=sessionTf)
        while n_eps_trained_model < n_epochs:
            epoch = n_eps_trained_model

            acc_monitor_ep_tr = AccuracyMonitorForEpSegm(log, 0,
                                                         n_eps_trained_model,
                                                         cnn3d.num_classes,
                                                         n_subepochs,
                                                         tensorboard_logger=tensorboard_loggers['train']
                                                            if tensorboard_loggers is not None else None)
            acc_monitor_ep_val = None
            if val_on_samples or val_on_whole_volumes:
                acc_monitor_ep_val = AccuracyMonitorForEpSegm(log, 1,
                                                              n_eps_trained_model,
                                                              cnn3d.num_classes,
                                                              n_subepochs,
                                                              tensorboard_logger=tensorboard_loggers['val']
                                                                if tensorboard_loggers is not None else None)
            
            val_on_whole_vols_after_this_ep = False
            if val_on_whole_volumes and (n_eps_trained_model + 1) % n_epochs_between_val_on_whole_vols == 0:
                val_on_whole_vols_after_this_ep = True
                
            log.print3("")
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            log.print3("~~\t\t\t Starting new Epoch! Epoch #" + str(epoch) + "/" + str(n_epochs) + "  \t\t\t~~")
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            start_time_ep = time.time()

            for subepoch in range(n_subepochs):
                log.print3("")
                log.print3("***********************************************************************************")
                log.print3("*\t\t\t Starting new Subepoch: #" + str(subepoch) + "/" + str(n_subepochs) + " \t\t\t*")
                log.print3("***********************************************************************************")

                # -------------------- GET DATA FOR THIS SUBEPOCH's VALIDATION -----------------------
                if val_on_samples:
                    if worker_pool is None:  # Sequential processing.
                        log.print3(id_str + " NO MULTIPROC: Sampling for subepoch #" + str(
                            subepoch) + " [VALIDATION] will be done by main thread.")
                        (channsOfSegmentsForSubepPerPathwayVal,
                         labelsForCentralOfSegmentsForSubepVal) = getSampledDataAndLabelsForSubepoch(
                            *args_for_sampling_val)
                    elif sampling_job_submitted_val:  # done parallel with training of previous epoch.
                        (channsOfSegmentsForSubepPerPathwayVal,
                         labelsForCentralOfSegmentsForSubepVal) = parallel_sampling_job_val.get()
                        sampling_job_submitted_val = False
                    else:  # Not previously submitted in case of first epoch or after a full-volumes validation.
                        assert subepoch == 0
                        log.print3(id_str + " MULTIPROC: Before Validation in subepoch #" + str(
                            subepoch) + ", submitting sampling job for next [VALIDATION].")
                        parallel_sampling_job_val = worker_pool.apply_async(
                            getSampledDataAndLabelsForSubepoch, args_for_sampling_val)
                        (channsOfSegmentsForSubepPerPathwayVal,
                         labelsForCentralOfSegmentsForSubepVal) = parallel_sampling_job_val.get()
                        sampling_job_submitted_val = False

                    # ----------- SUBMIT PARALLEL JOB TO GET TRAINING DATA FOR NEXT TRAINING -----------------
                    if worker_pool is not None:
                        log.print3(id_str + " MULTIPROC: Before Validation in subepoch #" + str(
                            subepoch) + ", submitting sampling job for next [TRAINING].")
                        parallel_sampling_job_tr = worker_pool.apply_async(
                            getSampledDataAndLabelsForSubepoch, args_for_sampling_train)
                        sampling_job_submitted_train = True

                    # ------------------------------------DO VALIDATION--------------------------------
                    log.print3("V-V-V-V- Validating for subepoch before starting training iterations -V-V-V-V")
                    start_time_val_subep = time.time()
                    # Calc num of batches from extracted samples, in case not extracted as much as requested.
                    num_batches_val = len(channsOfSegmentsForSubepPerPathwayVal[0]) // batchsize_val_samples
                    trainOrValidateForSubepoch(log,
                                               sessionTf,
                                               "val",
                                               num_batches_val,
                                               batchsize_val_samples,
                                               cnn3d,
                                               acc_monitor_ep_val,
                                               channsOfSegmentsForSubepPerPathwayVal,
                                               labelsForCentralOfSegmentsForSubepVal)
                    log.print3("TIMING: Validation on batches of subepoch #" + str(subepoch) +\
                               " lasted: {0:.1f}".format(time.time() - start_time_val_subep) + " secs.")

                # ----------------------- GET DATA FOR THIS SUBEPOCH's TRAINING ------------------------------
                if worker_pool is None:  # Sequential processing.
                    log.print3(id_str + " NO MULTIPROC: Sampling for subepoch #" + str(
                        subepoch) + " [TRAINING] will be done by main thread.")
                    (channsOfSegmentsForSubepPerPathwayTrain,
                     labelsForCentralOfSegmentsForSubepTrain) = getSampledDataAndLabelsForSubepoch(
                        *args_for_sampling_train)
                elif sampling_job_submitted_train:  # done parallel with train/val of previous epoch.
                    (channsOfSegmentsForSubepPerPathwayTrain,
                     labelsForCentralOfSegmentsForSubepTrain) = parallel_sampling_job_tr.get()
                    sampling_job_submitted_train = False
                else:  # Not previously submitted in case of first epoch or after a full-volumes validation.
                    assert subepoch == 0
                    log.print3(id_str + " MULTIPROC: Before Training in subepoch #" + str(
                        subepoch) + ", submitting sampling job for next [TRAINING].")
                    parallel_sampling_job_tr = worker_pool.apply_async(getSampledDataAndLabelsForSubepoch,
                                                                                  args_for_sampling_train)
                    (channsOfSegmentsForSubepPerPathwayTrain,
                     labelsForCentralOfSegmentsForSubepTrain) = parallel_sampling_job_tr.get()
                    sampling_job_submitted_train = False

                # ----- SUBMIT PARALLEL JOB TO GET VAL / TRAIN (if no val) DATA FOR NEXT SUBEPOCH -----
                if worker_pool is not None and not (val_on_whole_vols_after_this_ep and (subepoch == n_subepochs - 1)):
                    if val_on_samples:
                        log.print3(id_str + " MULTIPROC: Before Training in subepoch #" + str(
                            subepoch) + ", submitting sampling job for next [VALIDATION].")
                        parallel_sampling_job_val = worker_pool.apply_async(
                            getSampledDataAndLabelsForSubepoch, args_for_sampling_val)
                        sampling_job_submitted_val = True
                    else:
                        log.print3(id_str + " MULTIPROC: Before Training in subepoch #" + str(
                            subepoch) + ", submitting sampling job for next [TRAINING].")
                        parallel_sampling_job_tr = worker_pool.apply_async(
                            getSampledDataAndLabelsForSubepoch, args_for_sampling_train)
                        sampling_job_submitted_train = True

                # ------------------------------ START TRAINING IN BATCHES -----------------------------
                log.print3("-T-T-T-T-T- Now Training for this subepoch... This may take a few minutes... -T-T-T-T-T-")
                start_time_train_subep = time.time()
                # Calc num of batches from extracted samples, in case not extracted as much as requested.
                num_batches_train = len(channsOfSegmentsForSubepPerPathwayTrain[0]) // batchsize_train
                trainOrValidateForSubepoch(log,
                                           sessionTf,
                                           "train",
                                           num_batches_train,
                                           batchsize_train,
                                           cnn3d,
                                           acc_monitor_ep_tr,
                                           channsOfSegmentsForSubepPerPathwayTrain,
                                           labelsForCentralOfSegmentsForSubepTrain)
                log.print3("TIMING: Training on batches of this subepoch #" + str(subepoch) +\
                           " lasted: {0:.1f}".format(time.time() - start_time_train_subep) + " secs.")

            log.print3("")
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            log.print3("~~~~~~ Epoch #" + str(epoch) + " finished. Reporting Accuracy over whole epoch. ~~~~~~~")
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            if val_on_samples:
                acc_monitor_ep_val.report_metrics_samples_ep()
            acc_monitor_ep_tr.report_metrics_samples_ep()

            mean_val_acc_of_ep = acc_monitor_ep_val.get_avg_accuracy_ep() if val_on_samples else None
            # Updates LR schedule if needed, and increases number of epochs trained.
            trainer.run_updates_end_of_ep(log, sessionTf, mean_val_acc_of_ep)
            n_eps_trained_model = trainer.get_num_epochs_trained_tfv().eval(session=sessionTf)

            log.print3("SAVING: Epoch #" + str(epoch) + " finished. Saving CNN model.")
            filename_to_save_with = fileToSaveTrainedCnnModelTo + "." + datetimeNowAsStr()
            saver_all.save(sessionTf, filename_to_save_with + ".model.ckpt", write_meta_graph=False)

            log.print3("TIMING: Whole Epoch #" + str(epoch) +\
                       " lasted: {0:.1f}".format(time.time() - start_time_ep) + " secs.")
            log.print3("~~~~~~~~~~~~~~~~~~~~ End of Training Epoch. Model was Saved. ~~~~~~~~~~~~~~~~~~~~~~~~~~")

            if val_on_whole_vols_after_this_ep:
                log.print3("***Start validation by segmenting whole subjects for Epoch #" + str(epoch) + "***")

                mean_metrics_val_whole_vols = inference_on_whole_volumes(sessionTf,
                                                                         cnn3d,
                                                                         log,
                                                                         "val",
                                                                         savePredictedSegmAndProbsDict,
                                                                         paths_per_chan_per_subj_val,
                                                                         paths_to_lbls_per_subj_val,
                                                                         paths_to_masks_per_subj_val,
                                                                         namesForSavingSegmAndProbs,
                                                                         suffixForSegmAndProbsDict,
                                                                         # Hyper parameters
                                                                         batchsize_val_whole,
                                                                         # Data compatibility checks
                                                                         run_input_checks,
                                                                         # Pre-Processing
                                                                         pad_input,
                                                                         norm_prms,
                                                                         # Saving feature maps
                                                                         save_fms_flag,
                                                                         idxs_fms_to_save,
                                                                         namesForSavingFms)
                
                acc_monitor_ep_val.report_metrics_whole_vols(mean_metrics_val_whole_vols)

            del acc_monitor_ep_tr
            del acc_monitor_ep_val

        log.print3("TIMING: Training process lasted: {0:.1f}".format(time.time() - start_time_train) + " secs.")

    except (Exception, KeyboardInterrupt) as e:
        log.print3("\n\n ERROR: Caught exception in do_training(): " + str(e) + "\n")
        log.print3(traceback.format_exc())
        if worker_pool is not None:
            log.print3("Terminating worker pool.")
            worker_pool.terminate()
            worker_pool.join()  # Will wait. A KeybInt will kill this (py3)
        return 1
    else:
        if worker_pool is not None:
            log.print3("Closing worker pool.")
            worker_pool.close()
            worker_pool.join()

    # Save the final trained model.
    filename_to_save_with = fileToSaveTrainedCnnModelTo + ".final." + datetimeNowAsStr()
    log.print3("Saving the final model at:" + str(filename_to_save_with))
    saver_all.save(sessionTf, filename_to_save_with + ".model.ckpt", write_meta_graph=False)

    log.print3("The whole do_training() function has finished.")
    return 0
