# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import sys
import time
from multiprocessing.pool import ThreadPool, Pool
import traceback
import signal

import numpy as np

from deepmedic.logging.accuracyMonitor import AccuracyOfEpochMonitorSegmentation
from deepmedic.neuralnet.wrappers import CnnWrapperForSampling
from deepmedic.dataManagement.sampling import getSampledDataAndLabelsForSubepoch
from deepmedic.routines.testing import performInferenceOnWholeVolumes

from deepmedic.logging.utils import datetimeNowAsStr


# The main subroutine of do_training, that runs for every batch of validation and training.
def trainOrValidateForSubepoch( log,
                                sessionTf,
                                train_or_val,
                                number_of_batches,
                                cnn3d,
                                subepoch,
                                accuracyMonitorForEpoch,
                                channsOfSegmentsForSubepPerPathway,
                                labelsForCentralOfSegmentsForSubep) :
    """
    Returned array is of dimensions [NumberOfClasses x 6]
    For each class: [meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch, meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, meanCostOfSubepoch]
    In the case of VALIDATION, meanCostOfSubepoch is just a placeholder. Only valid when training.
    """
    
    costsOfBatches = []
    #each row in the array below will hold the number of Real Positives, Real Negatives, True Predicted Positives and True Predicted Negatives in the subepoch, in this order.
    arrayWithPerClassRpRnTpTnInSubepoch = np.zeros([ cnn3d.num_classes, 4 ], dtype="int32")
    
    for batch_i in range(number_of_batches):
        printProgressStep = max(1, number_of_batches//5)
        
        if train_or_val=="train" :
            if  batch_i%printProgressStep == 0 :
                log.print3( "[TRAINING] Trained on "+str(batch_i)+"/"+str(number_of_batches)+" of the batches for this subepoch...")
            
            ops_to_fetch = cnn3d.get_main_ops('train')
            list_of_ops = [ ops_to_fetch['cost'] ] + ops_to_fetch['list_rp_rn_tp_tn'] + [ ops_to_fetch['updates_grouped_op'] ]
            
            index_to_data_for_batch_min = batch_i * cnn3d.batchSize["train"]
            index_to_data_for_batch_max = (batch_i + 1) * cnn3d.batchSize["train"]
            
            feeds = cnn3d.get_main_feeds('train')
            feeds_dict = { feeds['x'] : channsOfSegmentsForSubepPerPathway[0][ index_to_data_for_batch_min : index_to_data_for_batch_max ] }
            for subsPath_i in range(cnn3d.numSubsPaths) :
                feeds_dict.update( { feeds['x_sub_'+str(subsPath_i)]: channsOfSegmentsForSubepPerPathway[ subsPath_i+1 ][ index_to_data_for_batch_min : index_to_data_for_batch_max ] } )
            feeds_dict.update( { feeds['y_gt'] : labelsForCentralOfSegmentsForSubep[ index_to_data_for_batch_min : index_to_data_for_batch_max ] } )
            # Training step
            results_from_train = sessionTf.run( fetches=list_of_ops, feed_dict=feeds_dict )
            
            listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining = results_from_train[:-1] # [-1] is from the updates_grouped_op that returns nothing.
            
            cnn3d.updateMatricesOfBnMovingAvForInference(sessionTf) #I should put this inside the 3dCNN.
            
            costOfThisBatch = listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining[0]
            listWithNumberOfRpRnPpPnForEachClass = listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining[1:]
            
        else : #validation
            if  batch_i%printProgressStep == 0 :
                log.print3( "[VALIDATION] Validated on "+str(batch_i)+"/"+str(number_of_batches)+" of the batches for this subepoch...")
                
            ops_to_fetch = cnn3d.get_main_ops('val')
            list_of_ops = ops_to_fetch['list_rp_rn_tp_tn']
            
            index_to_data_for_batch_min = batch_i * cnn3d.batchSize["val"]
            index_to_data_for_batch_max = (batch_i + 1) * cnn3d.batchSize["val"]
            
            feeds = cnn3d.get_main_feeds('val')
            feeds_dict = { feeds['x'] : channsOfSegmentsForSubepPerPathway[0][ index_to_data_for_batch_min : index_to_data_for_batch_max ] }
            for subsPath_i in range(cnn3d.numSubsPaths) :
                feeds_dict.update( { feeds['x_sub_'+str(subsPath_i)]: channsOfSegmentsForSubepPerPathway[ subsPath_i+1 ][ index_to_data_for_batch_min : index_to_data_for_batch_max ] } )
            feeds_dict.update( { feeds['y_gt'] : labelsForCentralOfSegmentsForSubep[ index_to_data_for_batch_min : index_to_data_for_batch_max ] } )
            # Validation step
            listWithMeanErrorAndRpRnTpTnForEachClassFromValidation = sessionTf.run( fetches=list_of_ops, feed_dict=feeds_dict )
            
            costOfThisBatch = 999 #placeholder in case of validation.
            listWithNumberOfRpRnPpPnForEachClass = listWithMeanErrorAndRpRnTpTnForEachClassFromValidation[:]
            
        # listWithNumberOfRpRnPpPnForEachClass holds Real Pos, Real Neg, True Pred Pos, True Pred Neg ...
        # ... for all classes, in this order, flattened. First RpRnTpTn are for 'WHOLE' class.
        arrayWithPerClassRpRnPpPnForBatch = np.asarray(listWithNumberOfRpRnPpPnForEachClass, dtype="int32").reshape(arrayWithPerClassRpRnTpTnInSubepoch.shape, order='C')
        
        # To later calculate the mean error and cost over the subepoch
        costsOfBatches.append(costOfThisBatch) #only really used in training.
        arrayWithPerClassRpRnTpTnInSubepoch += arrayWithPerClassRpRnPpPnForBatch
    
    #======== Calculate and Report accuracy over subepoch
    # In case of validation, meanCostOfSubepoch is just a placeholder. Cause this does not get calculated and reported in this case.
    meanCostOfSubepoch = accuracyMonitorForEpoch.NA_PATTERN if (train_or_val == "val") else sum(costsOfBatches) / float(number_of_batches)
    # This function does NOT flip the class-0 background to foreground!
    accuracyMonitorForEpoch.updateMonitorAccuraciesWithNewSubepochEntries(meanCostOfSubepoch, arrayWithPerClassRpRnTpTnInSubepoch)
    accuracyMonitorForEpoch.reportAccuracyForLastSubepoch()
    # Done


#------------------------------ MAIN TRAINING ROUTINE -------------------------------------
def do_training(sessionTf,
                saver_all,
                cnn3d,
                trainer,
                log,
                
                fileToSaveTrainedCnnModelTo,
                
                performValidationOnSamplesDuringTrainingProcessBool,
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
                number_of_subepochs, # per epoch. Every subepoch Accuracy is reported
                maxNumSubjectsLoadedPerSubepoch,  # Max num of cases loaded every subepoch for segments extraction.
                imagePartsLoadedInGpuPerSubepoch,
                imagePartsLoadedInGpuPerSubepochValidation,
                
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
                performFullInferenceOnValidationImagesEveryFewEpochsBool,
                everyThatManyEpochsComputeDiceOnTheFullValidationImages,
                
                #--------For FM visualisation---------
                saveIndividualFmImagesForVisualisation,
                saveMultidimensionalImageWithAllFms,
                indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                namesForSavingFms,
                
                #-------- Others --------
                run_input_checks
                ):
    
    start_training_time = time.time()
    
    # I cannot pass cnn3d to the sampling function, because the pp module used to reload theano. 
    # This created problems in the GPU when cnmem is used. Not sure this is needed with Tensorflow. Probably.
    cnn3dWrapper = CnnWrapperForSampling(cnn3d) 
    
    tupleWithArgsForTraining = (    log,
                                    "train",
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
                                    reflectImageWithHalfProbDuringTraining
                                    )
    tupleWithArgsForValidation = (  log,
                                    "val",
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
                                    [0,0,0] #don't perform reflection-augmentation during validation.
                                    )
    boolItIsTheVeryFirstSubepochOfThisProcess = True #to know so that in the very first I sequentially load the data for it.
    
    # For parallel extraction of samples for next train/val while processing previous iteration.
    threadPool = ThreadPool(processes=1) # Or multiprocessing.Pool.Pool(...), same API.
    
    try:
        model_num_epochs_trained = trainer.get_num_epochs_trained_tfv().eval(session=sessionTf)
        while model_num_epochs_trained < n_epochs :
            epoch = model_num_epochs_trained
            
            trainingAccuracyMonitorForEpoch = AccuracyOfEpochMonitorSegmentation(log, 0, model_num_epochs_trained, cnn3d.num_classes, number_of_subepochs)
            validationAccuracyMonitorForEpoch = None if not performValidationOnSamplesDuringTrainingProcessBool else \
                                            AccuracyOfEpochMonitorSegmentation(log, 1, model_num_epochs_trained, cnn3d.num_classes, number_of_subepochs ) 
                                            
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            log.print3("~~~~~~~~~~~~~~~~~~~~Starting new Epoch! Epoch #"+str(epoch)+"/"+str(n_epochs)+" ~~~~~~~~~~~~~~~~~~~~~~~~~")
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            start_epoch_time = time.time()
            
            for subepoch in range(number_of_subepochs):
                log.print3("**************************************************************************************************")
                log.print3("************* Starting new Subepoch: #"+str(subepoch)+"/"+str(number_of_subepochs)+" *************")
                log.print3("**************************************************************************************************")
                
                #-------------------------GET DATA FOR THIS SUBEPOCH's VALIDATION---------------------------------
                if performValidationOnSamplesDuringTrainingProcessBool :
                    if boolItIsTheVeryFirstSubepochOfThisProcess :
                        [channsOfSegmentsForSubepPerPathwayVal,
                        labelsForCentralOfSegmentsForSubepVal] = getSampledDataAndLabelsForSubepoch( *tupleWithArgsForValidation )
                        boolItIsTheVeryFirstSubepochOfThisProcess = False
                    else : #It was done in parallel with the training of the previous epoch, just grab the results.
                        [channsOfSegmentsForSubepPerPathwayVal,
                        labelsForCentralOfSegmentsForSubepVal] = parallelJobToGetDataForNextValidation.get() # instead of threadpool.join()
                        
                    #------------------------SUBMIT PARALLEL JOB TO GET TRAINING DATA FOR NEXT TRAINING-----------------
                    log.print3("PARALLEL: Before Validation in subepoch #" +str(subepoch) + ", the parallel job for extracting Segments for the next Training is submitted.")
                    parallelJobToGetDataForNextTraining = threadPool.apply_async(getSampledDataAndLabelsForSubepoch, # func to execute.
                                                                            tupleWithArgsForTraining) # tuble with args for func
                    
                    #------------------------------------DO VALIDATION--------------------------------
                    log.print3("-V-V-V-V-V- Now Validating for this subepoch before commencing the training iterations... -V-V-V-V-V-")
                    start_validationForSubepoch_time = time.time()
                    # Compute num of batches from num of extracted samples, in case we did not extract as many as initially requested.
                    numberOfBatchesValidation = len(channsOfSegmentsForSubepPerPathwayVal[0]) // cnn3d.batchSize["val"]
                    
                    trainOrValidateForSubepoch( log,
                                                sessionTf,
                                                "val",
                                                numberOfBatchesValidation, # Computed by the number of extracted samples. So, adapts.
                                                cnn3d,
                                                subepoch,
                                                validationAccuracyMonitorForEpoch,
                                                channsOfSegmentsForSubepPerPathwayVal,
                                                labelsForCentralOfSegmentsForSubepVal)
                    
                    end_validationForSubepoch_time = time.time()
                    log.print3("TIMING: Validating on the batches of this subepoch #"+str(subepoch)+" took time: "+\
                               str(end_validationForSubepoch_time-start_validationForSubepoch_time)+"(s)")
                
                #-------------------------GET DATA FOR THIS SUBEPOCH's TRAINING---------------------------------
                if (not performValidationOnSamplesDuringTrainingProcessBool) and boolItIsTheVeryFirstSubepochOfThisProcess :                    
                    [channsOfSegmentsForSubepPerPathwayTrain,
                    labelsForCentralOfSegmentsForSubepTrain] = getSampledDataAndLabelsForSubepoch( *tupleWithArgsForTraining )
                    boolItIsTheVeryFirstSubepochOfThisProcess = False
                else :
                    #It was done in parallel with the validation (or with previous training iteration, in case I am not performing validation).
                    [channsOfSegmentsForSubepPerPathwayTrain,
                    labelsForCentralOfSegmentsForSubepTrain] = parallelJobToGetDataForNextTraining.get() # From parallel process/thread.
                
                #------------------------SUBMIT PARALLEL JOB TO GET VALIDATION/TRAINING DATA (if val is/not performed) FOR NEXT SUBEPOCH-----------------
                if performValidationOnSamplesDuringTrainingProcessBool :
                    #submit the parallel job
                    log.print3("PARALLEL: Before Training in subepoch #" +str(subepoch) + ", submitting the parallel job for extracting Segments for the next Validation.")
                    parallelJobToGetDataForNextValidation = threadPool.apply_async(getSampledDataAndLabelsForSubepoch, #local function to call and execute in parallel.
                                                                                tupleWithArgsForValidation) #tuple with the arguments required
                else : #extract in parallel the samples for the next subepoch's training.
                    log.print3("PARALLEL: Before Training in subepoch #" +str(subepoch) + ", submitting the parallel job for extracting Segments for the next Training.")
                    parallelJobToGetDataForNextTraining = threadPool.apply_async(getSampledDataAndLabelsForSubepoch, #local function to call and execute in parallel.
                                                                                tupleWithArgsForTraining) #tuple with the arguments required
                    
                #-------------------------------START TRAINING IN BATCHES------------------------------
                log.print3("-T-T-T-T-T- Now Training for this subepoch... This may take a few minutes... -T-T-T-T-T-")
                start_trainingForSubepoch_time = time.time()
                # Compute num of batches from num of extracted samples, in case we did not extract as many as initially requested.
                numberOfBatchesTraining = len(channsOfSegmentsForSubepPerPathwayTrain[0]) // cnn3d.batchSize["train"]
                
                trainOrValidateForSubepoch( log,
                                            sessionTf,
                                            "train",
                                            numberOfBatchesTraining,
                                            cnn3d,
                                            subepoch,
                                            trainingAccuracyMonitorForEpoch,
                                            channsOfSegmentsForSubepPerPathwayTrain,
                                            labelsForCentralOfSegmentsForSubepTrain)
                end_trainingForSubepoch_time = time.time()
                log.print3("TIMING: Training on the batches of this subepoch #"+str(subepoch)+" took time: "+\
                           str(end_trainingForSubepoch_time-start_trainingForSubepoch_time)+"(s)")
                
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
            log.print3("~~~~~~~~~~~~~~~~~~ Epoch #" + str(epoch) + " finished. Reporting Accuracy over whole epoch. ~~~~~~~~~~~~~~~~~~" )
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
            
            if performValidationOnSamplesDuringTrainingProcessBool :
                validationAccuracyMonitorForEpoch.reportMeanAccyracyOfEpoch()
            trainingAccuracyMonitorForEpoch.reportMeanAccyracyOfEpoch()
            
            mean_val_acc_of_ep = validationAccuracyMonitorForEpoch.getMeanEmpiricalAccuracyOfEpoch() if performValidationOnSamplesDuringTrainingProcessBool else None
            trainer.run_updates_end_of_ep(log, sessionTf, mean_val_acc_of_ep) # Updates LR schedule if needed, and increases number of epochs trained.
            model_num_epochs_trained = trainer.get_num_epochs_trained_tfv().eval(session=sessionTf)
            
            del trainingAccuracyMonitorForEpoch; del validationAccuracyMonitorForEpoch;
            #================== Everything for epoch has finished. =======================
            
            log.print3("SAVING: Epoch #"+str(epoch)+" finished. Saving CNN model.")
            filename_to_save_with = fileToSaveTrainedCnnModelTo + "." + datetimeNowAsStr()
            saver_all.save( sessionTf, filename_to_save_with+".model.ckpt", write_meta_graph=False )
            
            end_epoch_time = time.time()
            log.print3("TIMING: The whole Epoch #"+str(epoch)+" took time: "+str(end_epoch_time-start_epoch_time)+"(s)")
            log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of Training Epoch. Model was Saved. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            
            if performFullInferenceOnValidationImagesEveryFewEpochsBool and (model_num_epochs_trained != 0) \
                    and (model_num_epochs_trained % everyThatManyEpochsComputeDiceOnTheFullValidationImages == 0):
                log.print3("***Starting validation with Full Inference / Segmentation on validation subjects for Epoch #"+str(epoch)+"...***")
                
                performInferenceOnWholeVolumes(sessionTf,
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
        end_training_time = time.time()
        log.print3("TIMING: Training process took time: "+str(end_training_time-start_training_time)+"(s)")
        
    except (Exception, KeyboardInterrupt) as e:
        log.print3("")
        log.print3("ERROR: Caught exception in do_training(...): " + str(e))
        log.print3( traceback.format_exc() )
        threadPool.terminate()
    else:
        log.print3("Closing multiprocess pool.")
        threadPool.close()
    finally:
        threadPool.join()
        
    log.print3("The whole do_training() function has finished.")
    

