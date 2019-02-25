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

from deepmedic.logging.accuracyMonitor import AccuracyOfEpochMonitorSegmentation
from deepmedic.neuralnet.wrappers import CnnWrapperForSampling
from deepmedic.dataManagement.sampling import getSampledDataAndLabelsForSubepoch
from deepmedic.routines.testing import inferenceWholeVolumes

from deepmedic.logging.utils import datetimeNowAsStr


# The main subroutine of do_training, that runs for every batch of validation and training.
def trainOrValidateForSubepoch( log,
                                sessionTf,
                                train_or_val,
                                num_batches,
                                cnn3d,
                                subepoch,
                                acc_monitor_for_ep,
                                channsOfSegmentsForSubepPerPathway,
                                labelsForCentralOfSegmentsForSubep) :
    """
    Returned array is of dimensions [NumberOfClasses x 6]
    For each class: [meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch, meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, mean_cost_subep]
    In the case of VALIDATION, mean_cost_subep is just a placeholder. Only valid when training.
    """
    
    costs_of_batches = []
    #each row in the array below will hold the number of Real Positives, Real Negatives, True Predicted Positives and True Predicted Negatives in the subepoch, in this order.
    array_per_class_RpRnTpTn_in_subep = np.zeros([ cnn3d.num_classes, 4 ], dtype="int32")
    
    print_progress_step = max(1, num_batches//5)
    
    for batch_i in range(num_batches):
        
        if train_or_val=="train" :
            if batch_i == 0 or ((batch_i+1) % print_progress_step) == 0 or (batch_i+1) == num_batches :
                log.print3( "[TRAINING] Trained on "+str(batch_i+1)+"/"+str(num_batches)+" of the batches for this subepoch...")
            
            ops_to_fetch = cnn3d.get_main_ops('train')
            list_of_ops = [ ops_to_fetch['cost'] ] + ops_to_fetch['list_rp_rn_tp_tn'] + [ ops_to_fetch['updates_grouped_op'] ]
            
            index_to_data_for_batch_min = batch_i * cnn3d.batchSize["train"]
            index_to_data_for_batch_max = (batch_i + 1) * cnn3d.batchSize["train"]
            
            feeds = cnn3d.get_main_feeds('train')
            feeds_dict = { feeds['x'] : channsOfSegmentsForSubepPerPathway[0][ index_to_data_for_batch_min : index_to_data_for_batch_max ] }
            for subs_path_i in range(cnn3d.numSubsPaths) :
                feeds_dict.update( { feeds['x_sub_'+str(subs_path_i)]: channsOfSegmentsForSubepPerPathway[ subs_path_i+1 ][ index_to_data_for_batch_min : index_to_data_for_batch_max ] } )
            feeds_dict.update( { feeds['y_gt'] : labelsForCentralOfSegmentsForSubep[ index_to_data_for_batch_min : index_to_data_for_batch_max ] } )
            # Training step. Returns a list containing the results of fetched ops.
            results_of_run = sessionTf.run( fetches=list_of_ops, feed_dict=feeds_dict )
            
            cnn3d.updateMatricesOfBnMovingAvForInference(sessionTf) #I should put this inside the 3dCNN.
            
            cost_this_batch = results_of_run[0]
            list_RpRnPpPn_per_class = results_of_run[1:-1] # [-1] is from updates_grouped_op that returns nothing.
            
        else : #validation
            if batch_i == 0 or ((batch_i+1) % print_progress_step) == 0 or (batch_i+1) == num_batches :
                log.print3( "[VALIDATION] Validated on "+str(batch_i+1)+"/"+str(num_batches)+" of the batches for this subepoch...")
                
            ops_to_fetch = cnn3d.get_main_ops('val')
            list_of_ops = ops_to_fetch['list_rp_rn_tp_tn']
            
            index_to_data_for_batch_min = batch_i * cnn3d.batchSize["val"]
            index_to_data_for_batch_max = (batch_i + 1) * cnn3d.batchSize["val"]
            
            feeds = cnn3d.get_main_feeds('val')
            feeds_dict = { feeds['x'] : channsOfSegmentsForSubepPerPathway[0][ index_to_data_for_batch_min : index_to_data_for_batch_max ] }
            for subs_path_i in range(cnn3d.numSubsPaths) :
                feeds_dict.update( { feeds['x_sub_'+str(subs_path_i)]: channsOfSegmentsForSubepPerPathway[ subs_path_i+1 ][ index_to_data_for_batch_min : index_to_data_for_batch_max ] } )
            feeds_dict.update( { feeds['y_gt'] : labelsForCentralOfSegmentsForSubep[ index_to_data_for_batch_min : index_to_data_for_batch_max ] } )
            # Validation step. Returns a list containing the results of fetched ops.
            results_of_run = sessionTf.run( fetches=list_of_ops, feed_dict=feeds_dict )
            
            cost_this_batch = 999 #placeholder in case of validation.
            list_RpRnPpPn_per_class = results_of_run
            
        # list_RpRnPpPn_per_class holds Real Pos, Real Neg, True Pred Pos, True Pred Neg ...
        # ... for all classes, in this order, flattened. First RpRnTpTn are for 'WHOLE' class.
        array_per_class_RpRnTpTn_in_batch = np.asarray(list_RpRnPpPn_per_class, dtype="int32").reshape(array_per_class_RpRnTpTn_in_subep.shape, order='C')
        
        # To later calculate the mean error and cost over the subepoch
        costs_of_batches.append(cost_this_batch) #only really used in training.
        array_per_class_RpRnTpTn_in_subep += array_per_class_RpRnTpTn_in_batch
    
    #======== Calculate and Report accuracy over subepoch
    # In case of validation, mean_cost_subep is just a placeholder. Cause this does not get calculated and reported in this case.
    mean_cost_subep = acc_monitor_for_ep.NA_PATTERN if (train_or_val == "val") else sum(costs_of_batches) / float(num_batches)
    # This function does NOT flip the class-0 background to foreground!
    acc_monitor_for_ep.updateMonitorAccuraciesWithNewSubepochEntries(mean_cost_subep, array_per_class_RpRnTpTn_in_subep)
    acc_monitor_for_ep.reportAccuracyForLastSubepoch()
    # Done


#------------------------------ MAIN TRAINING ROUTINE -------------------------------------
def do_training(sessionTf,
                saver_all,
                cnn3d,
                trainer,
                log,
                
                fileToSaveTrainedCnnModelTo,
                
                val_on_samples_during_train,
                savePredictedSegmAndProbsDict,
                
                namesForSavingSegmAndProbs,
                suffixForSegmAndProbsDict,
                
                listOfFilepathsToEachChannelOfEachPatientTraining,
                listOfFilepathsToEachChannelOfEachPatientValidation,
                
                listOfFilepathsToGtLabelsOfEachPatientTraining,
                providedGtForValidationBool,
                listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc,
                
                providedWeightMapsToSampleForEachCategoryTraining,
                forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatientTraining,
                providedWeightMapsToSampleForEachCategoryValidation,
                forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatientValidation,
                
                providedRoiMaskForTrainingBool,
                listOfFilepathsToRoiMaskOfEachPatientTraining, # Also needed for normalization-augmentation
                providedRoiMaskForValidationBool,
                listOfFilepathsToRoiMaskOfEachPatientValidation,
                
                n_epochs, # Every epoch the CNN model is saved.
                num_subepochs, # per epoch. Every subepoch Accuracy is reported
                maxNumSubjectsLoadedPerSubepoch,  # Max num of subjects loaded every subepoch for segments extraction.
                imagePartsLoadedInGpuPerSubepoch,
                imagePartsLoadedInGpuPerSubepochValidation,
                num_parallel_proc_sampling, # -1: seq. 0: thread for sampling. >0: multiprocess sampling
                
                #-------Sampling Type---------
                samplingTypeInstanceTraining, # Instance of the deepmedic/samplingType.SamplingType class for training and validation
                samplingTypeInstanceValidation,
                
                #-------Preprocessing-----------
                padInputImagesBool,
                #-------Data Augmentation-------
                doIntAugm_shiftMuStd_multiMuStd,
                reflectImageWithHalfProbDuringTraining,
                
                useSameSubChannelsAsSingleScale,
                
                listOfFilepathsToEachSubsampledChannelOfEachPatientTraining, # deprecated, not supported
                listOfFilepathsToEachSubsampledChannelOfEachPatientValidation, # deprecated, not supported
                
                # Validation
                val_on_whole_volumes,
                num_epochs_between_val_on_whole_volumes,
                
                #--------For FM visualisation---------
                saveIndividualFmImagesForVisualisation,
                saveMultidimensionalImageWithAllFms,
                indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                namesForSavingFms,
                
                #-------- Others --------
                run_input_checks
                ):
    
    id_str = "[MAIN|PID:"+str(os.getpid())+"]"
    start_time_train = time.time()
    
    # I cannot pass cnn3d to the sampling function, because the pp module used to reload theano. 
    # This created problems in the GPU when cnmem is used. Not sure this is needed with Tensorflow. Probably.
    cnn3dWrapper = CnnWrapperForSampling(cnn3d)
    
    args_for_sampling_train = ( log,
                                "train",
                                num_parallel_proc_sampling,
                                run_input_checks,
                                cnn3dWrapper,
                                maxNumSubjectsLoadedPerSubepoch,
                                imagePartsLoadedInGpuPerSubepoch,
                                samplingTypeInstanceTraining,
                                listOfFilepathsToEachChannelOfEachPatientTraining,
                                listOfFilepathsToGtLabelsOfEachPatientTraining,
                                providedRoiMaskForTrainingBool,
                                listOfFilepathsToRoiMaskOfEachPatientTraining,
                                providedWeightMapsToSampleForEachCategoryTraining,
                                forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatientTraining,
                                useSameSubChannelsAsSingleScale,
                                listOfFilepathsToEachSubsampledChannelOfEachPatientTraining,
                                padInputImagesBool,
                                doIntAugm_shiftMuStd_multiMuStd,
                                reflectImageWithHalfProbDuringTraining )
    args_for_sampling_val = (   log,
                                "val",
                                num_parallel_proc_sampling,
                                run_input_checks,
                                cnn3dWrapper,
                                maxNumSubjectsLoadedPerSubepoch,
                                imagePartsLoadedInGpuPerSubepochValidation,
                                samplingTypeInstanceValidation,
                                listOfFilepathsToEachChannelOfEachPatientValidation,
                                listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc,
                                providedRoiMaskForValidationBool,
                                listOfFilepathsToRoiMaskOfEachPatientValidation,
                                providedWeightMapsToSampleForEachCategoryValidation,
                                forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatientValidation,
                                useSameSubChannelsAsSingleScale,
                                listOfFilepathsToEachSubsampledChannelOfEachPatientValidation,
                                padInputImagesBool,
                                [False,[],[]], #don't perform intensity-augmentation during validation.
                                [0,0,0] ) #don't perform reflection-augmentation during validation.
    
    sampling_job_submitted_train = False
    sampling_job_submitted_val = False
    # For parallel extraction of samples for next train/val while processing previous iteration.
    worker_pool = None
    if num_parallel_proc_sampling > -1 : # Use multiprocessing.
        worker_pool = ThreadPool(processes=1) # Or multiprocessing.Pool(...), same API.
    
    try:
        model_num_epochs_trained = trainer.get_num_epochs_trained_tfv().eval(session=sessionTf)
        while model_num_epochs_trained < n_epochs :
            epoch = model_num_epochs_trained
            
            acc_monitor_for_ep_train = AccuracyOfEpochMonitorSegmentation(log, 0, model_num_epochs_trained, cnn3d.num_classes, num_subepochs)
            acc_monitor_for_ep_val = None if not val_on_samples_during_train else \
                                            AccuracyOfEpochMonitorSegmentation(log, 1, model_num_epochs_trained, cnn3d.num_classes, num_subepochs )
                                            
            val_on_whole_volumes_after_ep = False
            if val_on_whole_volumes and (model_num_epochs_trained+1) % num_epochs_between_val_on_whole_volumes == 0:
                val_on_whole_volumes_after_ep = True
                
                
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            log.print3("~~~~~~~~~~~~~\t Starting new Epoch! Epoch #"+str(epoch)+"/"+str(n_epochs)+" \t~~~~~~~~~~~~~")
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            start_time_ep = time.time()
            
            for subepoch in range(num_subepochs):
                log.print3("***************************************************************************************")
                log.print3("*******\t\t Starting new Subepoch: #"+str(subepoch)+"/"+str(num_subepochs)+" \t\t********")
                log.print3("***************************************************************************************")
                
                #-------------------------GET DATA FOR THIS SUBEPOCH's VALIDATION---------------------------------
                if val_on_samples_during_train :
                    if worker_pool is None: # Sequential processing.
                        log.print3(id_str+" NO MULTIPROC: Sampling for subepoch #"+str(subepoch)+" [VALIDATION] will be done by main thread.")
                        [channsOfSegmentsForSubepPerPathwayVal,
                        labelsForCentralOfSegmentsForSubepVal] = getSampledDataAndLabelsForSubepoch( *args_for_sampling_val )
                    elif sampling_job_submitted_val : #It was done in parallel with the training of the previous epoch, just grab results.
                        [channsOfSegmentsForSubepPerPathwayVal,
                        labelsForCentralOfSegmentsForSubepVal] = parallelJobToGetDataForNextValidation.get()
                        sampling_job_submitted_val = False
                    else : # Not previously submitted in case of first epoch or after a full-volumes validation.
                        assert subepoch == 0
                        log.print3(id_str+" MULTIPROC: Before Validation in subepoch #"+str(subepoch)+", submitting sampling job for next [VALIDATION].")
                        parallelJobToGetDataForNextValidation = worker_pool.apply_async(getSampledDataAndLabelsForSubepoch, args_for_sampling_val)
                        [channsOfSegmentsForSubepPerPathwayVal,
                        labelsForCentralOfSegmentsForSubepVal] = parallelJobToGetDataForNextValidation.get()
                        sampling_job_submitted_val = False

                    
                    #------------------------SUBMIT PARALLEL JOB TO GET TRAINING DATA FOR NEXT TRAINING-----------------
                    if worker_pool is not None:
                        log.print3(id_str+" MULTIPROC: Before Validation in subepoch #"+str(subepoch)+", submitting sampling job for next [TRAINING].")
                        parallelJobToGetDataForNextTraining = worker_pool.apply_async(getSampledDataAndLabelsForSubepoch, args_for_sampling_train)
                        sampling_job_submitted_train = True
                    
                    #------------------------------------DO VALIDATION--------------------------------
                    log.print3("-V-V-V-V-V- Now Validating for this subepoch before commencing the training iterations... -V-V-V-V-V-")
                    start_time_val_subep = time.time()
                    # Compute num of batches from num of extracted samples, in case we did not extract as many as initially requested.
                    num_batches_val = len(channsOfSegmentsForSubepPerPathwayVal[0]) // cnn3d.batchSize["val"]
                    trainOrValidateForSubepoch( log,
                                                sessionTf,
                                                "val",
                                                num_batches_val,
                                                cnn3d,
                                                subepoch,
                                                acc_monitor_for_ep_val,
                                                channsOfSegmentsForSubepPerPathwayVal,
                                                labelsForCentralOfSegmentsForSubepVal )
                    end_time_val_subep = time.time()
                    log.print3("TIMING: Validation on batches of this subepoch #"+str(subepoch)+" lasted: {0:.1f}".format(end_time_val_subep-start_time_val_subep)+" secs.")
                
                #-------------------------GET DATA FOR THIS SUBEPOCH's TRAINING---------------------------------
                if worker_pool is None: # Sequential processing.
                    log.print3(id_str+" NO MULTIPROC: Sampling for subepoch #"+str(subepoch)+" [TRAINING] will be done by main thread.")
                    [channsOfSegmentsForSubepPerPathwayTrain,
                    labelsForCentralOfSegmentsForSubepTrain] = getSampledDataAndLabelsForSubepoch( *args_for_sampling_train )
                elif sampling_job_submitted_train: # Sampling job should have been done in parallel with previous train/val. Just grab results.
                    [channsOfSegmentsForSubepPerPathwayTrain,
                    labelsForCentralOfSegmentsForSubepTrain] = parallelJobToGetDataForNextTraining.get()
                    sampling_job_submitted_train = False
                else:  # Not previously submitted in case of first epoch or after a full-volumes validation.
                    assert subepoch == 0
                    log.print3(id_str+" MULTIPROC: Before Training in subepoch #"+str(subepoch)+", submitting sampling job for next [TRAINING].")
                    parallelJobToGetDataForNextTraining = worker_pool.apply_async(getSampledDataAndLabelsForSubepoch, args_for_sampling_train)
                    [channsOfSegmentsForSubepPerPathwayTrain,
                    labelsForCentralOfSegmentsForSubepTrain] = parallelJobToGetDataForNextTraining.get()
                    sampling_job_submitted_train = False

                        
                #------------------------SUBMIT PARALLEL JOB TO GET VALIDATION/TRAINING DATA (if val is/not performed) FOR NEXT SUBEPOCH-----------------
                if worker_pool is not None and not (val_on_whole_volumes_after_ep and (subepoch == num_subepochs-1)):
                    if val_on_samples_during_train :
                        log.print3(id_str+" MULTIPROC: Before Training in subepoch #"+str(subepoch)+", submitting sampling job for next [VALIDATION].")
                        parallelJobToGetDataForNextValidation = worker_pool.apply_async(getSampledDataAndLabelsForSubepoch, args_for_sampling_val)
                        sampling_job_submitted_val = True
                    else :
                        log.print3(id_str+" MULTIPROC: Before Training in subepoch #"+str(subepoch)+", submitting sampling job for next [TRAINING].")
                        parallelJobToGetDataForNextTraining = worker_pool.apply_async(getSampledDataAndLabelsForSubepoch, args_for_sampling_train)
                        sampling_job_submitted_train = True
                
                #-------------------------------START TRAINING IN BATCHES------------------------------
                log.print3("-T-T-T-T-T- Now Training for this subepoch... This may take a few minutes... -T-T-T-T-T-")
                start_time_train_subep = time.time()
                # Compute num of batches from num of extracted samples, in case we did not extract as many as initially requested.
                num_batches_train = len(channsOfSegmentsForSubepPerPathwayTrain[0]) // cnn3d.batchSize["train"]
                trainOrValidateForSubepoch( log,
                                            sessionTf,
                                            "train",
                                            num_batches_train,
                                            cnn3d,
                                            subepoch,
                                            acc_monitor_for_ep_train,
                                            channsOfSegmentsForSubepPerPathwayTrain,
                                            labelsForCentralOfSegmentsForSubepTrain )
                end_time_train_subep = time.time()
                log.print3("TIMING: Training on batches of this subepoch #"+str(subepoch)+" lasted: {0:.1f}".format(end_time_train_subep-start_time_train_subep)+" secs.")
                
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            log.print3("~~~~~~ Epoch #" + str(epoch) + " finished. Reporting Accuracy over whole epoch. ~~~~~~~" )
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            if val_on_samples_during_train:
                acc_monitor_for_ep_val.reportMeanAccyracyOfEpoch()
            acc_monitor_for_ep_train.reportMeanAccyracyOfEpoch()
            
            mean_val_acc_of_ep = acc_monitor_for_ep_val.getMeanEmpiricalAccuracyOfEpoch() if val_on_samples_during_train else None
            trainer.run_updates_end_of_ep(log, sessionTf, mean_val_acc_of_ep) # Updates LR schedule if needed, and increases number of epochs trained.
            model_num_epochs_trained = trainer.get_num_epochs_trained_tfv().eval(session=sessionTf)
            
            del acc_monitor_for_ep_train; del acc_monitor_for_ep_val;
            
            log.print3("SAVING: Epoch #"+str(epoch)+" finished. Saving CNN model.")
            filename_to_save_with = fileToSaveTrainedCnnModelTo + "." + datetimeNowAsStr()
            saver_all.save( sessionTf, filename_to_save_with+".model.ckpt", write_meta_graph=False )
            
            end_time_ep = time.time()
            log.print3("TIMING: The whole Epoch #"+str(epoch)+" lasted: {0:.1f}".format(end_time_ep-start_time_ep)+" secs.")
            log.print3("~~~~~~~~~~~~~~~~~~~~ End of Training Epoch. Model was Saved. ~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            
            if val_on_whole_volumes_after_ep:
                log.print3("***Starting validation with Full Inference / Segmentation on validation subjects for Epoch #"+str(epoch)+"...***")
                
                res_code = inferenceWholeVolumes(
                                sessionTf,
                                cnn3d,
                                log,
                                "val",
                                savePredictedSegmAndProbsDict,
                                listOfFilepathsToEachChannelOfEachPatientValidation,
                                providedGtForValidationBool,
                                listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc,
                                providedRoiMaskForValidationBool,
                                listOfFilepathsToRoiMaskOfEachPatientValidation,
                                namesForSavingSegmAndProbs = namesForSavingSegmAndProbs,
                                suffixForSegmAndProbsDict = suffixForSegmAndProbsDict,
                                
                                #----Preprocessing------
                                padInputImagesBool=padInputImagesBool,
                                
                                #for the cnn extension
                                useSameSubChannelsAsSingleScale=useSameSubChannelsAsSingleScale,
                                listOfFilepathsToEachSubsampledChannelOfEachPatient=listOfFilepathsToEachSubsampledChannelOfEachPatientValidation,
                                
                                #--------For FM visualisation---------
                                saveIndividualFmImagesForVisualisation=saveIndividualFmImagesForVisualisation,
                                saveMultidimensionalImageWithAllFms=saveMultidimensionalImageWithAllFms,
                                indicesOfFmsToVisualisePerPathwayTypeAndPerLayer=indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                                namesForSavingFms=namesForSavingFms
                                )
        end_time_train = time.time()
        log.print3("TIMING: Training process lasted: {0:.1f}".format(end_time_train-start_time_train)+" secs.")
        
    except (Exception, KeyboardInterrupt) as e:
        log.print3("\n\n ERROR: Caught exception in do_training(): " + str(e) + "\n")
        log.print3( traceback.format_exc() )
        if worker_pool is not None:
            log.print3("Terminating worker pool.")
            worker_pool.terminate()
            worker_pool.join() # Will wait. A KeybInt will kill this (py3)
        return 1
    else:
        if worker_pool is not None:
            log.print3("Closing worker pool.")
            worker_pool.close()
            worker_pool.join()
    
    # Save the final trained model.
    filename_to_save_with = fileToSaveTrainedCnnModelTo + ".final." + datetimeNowAsStr()
    log.print3("Saving the final model at:" + str(filename_to_save_with))
    saver_all.save( sessionTf, filename_to_save_with+".model.ckpt", write_meta_graph=False )
            
    log.print3("The whole do_training() function has finished.")
    return 0

