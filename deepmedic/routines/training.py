# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import sys
import time
import pp
import numpy as np

from deepmedic.logging.accuracyMonitor import AccuracyOfEpochMonitorSegmentation
from deepmedic.neuralnet.wrappers import CnnWrapperForSampling
from deepmedic.dataManagement.sampling import getSampledDataAndLabelsForSubepoch
from deepmedic.routines.testing import performInferenceOnWholeVolumes

from deepmedic.logging.utils import datetimeNowAsStr

TINY_FLOAT = np.finfo(np.float32).tiny


# The main subroutine of do_training, that runs for every batch of validation and training.
def doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(log,
                                                                  sessionTf,
                                                                train_or_val,
                                                                number_of_batches, #This is the integer division of (numb-o-segments/batchSize)
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
    trainedOrValidatedString = "Trained" if train_or_val == "train" else "Validated"
    
    costsOfBatches = []
    #each row in the array below will hold the number of Real Positives, Real Negatives, True Predicted Positives and True Predicted Negatives in the subepoch, in this order.
    arrayWithNumbersOfPerClassRpRnTpTnInSubepoch = np.zeros([ cnn3d.num_classes, 4 ], dtype="int32")
    
    for batch_i in range(number_of_batches):
        printProgressStep = max(1, number_of_batches//5)
        if  batch_i%printProgressStep == 0 :
            log.print3( trainedOrValidatedString + " on "+str(batch_i)+"/"+str(number_of_batches)+" of the batches for this subepoch...")
        if train_or_val=="train" :
            
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
            
            cnn3d.updateTheMatricesOfTheLayersWithTheLastMusAndVarsForTheMovingAverageOfTheBatchNormInference(sessionTf) #I should put this inside the 3dCNN.
            
            costOfThisBatch = listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining[0]
            listWithNumberOfRpRnPpPnForEachClass = listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining[1:]
            
        else : #validation
            
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
            
        #The returned listWithNumberOfRpRnPpPnForEachClass holds Real Positives, Real Negatives, True Predicted Positives and True Predicted Negatives for all classes in this order, flattened. First RpRnTpTn are for WHOLE "class".
        arrayWithNumberOfRpRnPpPnForEachClassForBatch = np.asarray(listWithNumberOfRpRnPpPnForEachClass, dtype="int32").reshape(arrayWithNumbersOfPerClassRpRnTpTnInSubepoch.shape, order='C')
        
        # To later calculate the mean error and cost over the subepoch
        costsOfBatches.append(costOfThisBatch) #only really used in training.
        arrayWithNumbersOfPerClassRpRnTpTnInSubepoch += arrayWithNumberOfRpRnPpPnForEachClassForBatch
    
    #======== Calculate and Report accuracy over subepoch
    # In case of validation, meanCostOfSubepoch is just a placeholder. Cause this does not get calculated and reported in this case.
    meanCostOfSubepoch = accuracyMonitorForEpoch.NA_PATTERN if (train_or_val == "val") else sum(costsOfBatches) / float(number_of_batches)
    # This function does NOT flip the class-0 background to foreground!
    accuracyMonitorForEpoch.updateMonitorAccuraciesWithNewSubepochEntries(meanCostOfSubepoch, arrayWithNumbersOfPerClassRpRnTpTnInSubepoch)
    accuracyMonitorForEpoch.reportAccuracyForLastSubepoch()
    #Done


#------------------------------ MAIN TRAINING ROUTINE -------------------------------------
def do_training(sessionTf,
                saver_all,
                cnn3d,
                trainer,
                log,
                
                fileToSaveTrainedCnnModelTo,
                
                performValidationOnSamplesDuringTrainingProcessBool,
                savePredictionImagesSegmentationAndProbMapsListWhenEvaluatingDiceForValidation,
                
                listOfNamesToGiveToPredictionsValidationIfSavingWhenEvalDice,
                
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
                maxNumSubjectsLoadedPerSubepoch,  # Max num of cases loaded every subepoch for segments extraction. The more, the longer loading.
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
                performFullInferenceOnValidationImagesEveryFewEpochsBool, #Even if not providedGtForValidationBool, inference will be performed if this == True, to save the results, eg for visual.
                everyThatManyEpochsComputeDiceOnTheFullValidationImages, # Should not be == 0, except if performFullInferenceOnValidationImagesEveryFewEpochsBool == False
                
                #--------For FM visualisation---------
                saveIndividualFmImagesForVisualisation,
                saveMultidimensionalImageWithAllFms,
                indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                listOfNamesToGiveToFmVisualisationsIfSaving,
                
                #-------- Others --------
                run_input_checks
                ):
    
    start_training_time = time.time()
    
    # Used because I cannot pass cnn3d to the sampling function.
    #This is because the parallel process used to load theano again. And created problems in the GPU when cnmem is used. Not sure this is needed with Tensorflow. Probably.
    cnn3dWrapper = CnnWrapperForSampling(cnn3d) 
    
    #---------To run PARALLEL the extraction of parts for the next subepoch---
    ppservers = () # tuple of all parallel python servers to connect with
    job_server = pp.Server(ncpus=1, ppservers=ppservers) # Creates jobserver with automatically detected number of workers
    
    tupleWithParametersForTraining = (log,
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
    tupleWithParametersForValidation = (log,
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
                                    [0, -1,-1,-1], #don't perform intensity-augmentation during validation.
                                    [0,0,0] #don't perform reflection-augmentation during validation.
                                    )
    tupleWithLocalFunctionsThatWillBeCalledByTheMainJob = ( )
    tupleWithModulesToImportWhichAreUsedByTheJobFunctions = ( "from __future__ import absolute_import, print_function, division",
                "time", "numpy as np", "from deepmedic.dataManagement.sampling import *" )
    boolItIsTheVeryFirstSubepochOfThisProcess = True #to know so that in the very first I sequencially load the data for it.
    #------End for parallel------
    
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
        
        for subepoch in range(number_of_subepochs): #per subepoch I randomly load some images in the gpu. Random order.
            log.print3("**************************************************************************************************")
            log.print3("************* Starting new Subepoch: #"+str(subepoch)+"/"+str(number_of_subepochs)+" *************")
            log.print3("**************************************************************************************************")
            
            #-------------------------GET DATA FOR THIS SUBEPOCH's VALIDATION---------------------------------
            
            if performValidationOnSamplesDuringTrainingProcessBool :
                if boolItIsTheVeryFirstSubepochOfThisProcess :
                    [channsOfSegmentsForSubepPerPathwayVal,
                    labelsForCentralOfSegmentsForSubepVal] = getSampledDataAndLabelsForSubepoch(log,
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
                                                                        doIntAugm_shiftMuStd_multiMuStd=[False,[],[]],
                                                                        reflectImageWithHalfProbDuringTraining = [0,0,0]
                                                                        )
                    boolItIsTheVeryFirstSubepochOfThisProcess = False
                else : #It was done in parallel with the training of the previous epoch, just grab the results...
                    [channsOfSegmentsForSubepPerPathwayVal,
                    labelsForCentralOfSegmentsForSubepVal] = parallelJobToGetDataForNextValidation() #fromParallelProcessing that had started from last loop when it was submitted.
                    
                # Below is computed with number of extracted samples, in case I dont manage to extract as many as I wanted initially.
                numberOfBatchesValidation = len(channsOfSegmentsForSubepPerPathwayVal[0]) // cnn3d.batchSize["val"]
                
                
                #------------------------SUBMIT PARALLEL JOB TO GET TRAINING DATA FOR NEXT TRAINING-----------------
                #submit the parallel job
                log.print3("PARALLEL: Before Validation in subepoch #" +str(subepoch) + ", the parallel job for extracting Segments for the next Training is submitted.")
                parallelJobToGetDataForNextTraining = job_server.submit(getSampledDataAndLabelsForSubepoch, #local function to call and execute in parallel.
                                                                        tupleWithParametersForTraining, #tuple with the arguments required
                                                                        tupleWithLocalFunctionsThatWillBeCalledByTheMainJob, #tuple of local functions that I need to call
                                                                        tupleWithModulesToImportWhichAreUsedByTheJobFunctions) #tuple of the external modules that I need, of which I am calling functions (not the mods of the ext-functions).
                
                #------------------------------------DO VALIDATION--------------------------------
                log.print3("-V-V-V-V-V- Now Validating for this subepoch before commencing the training iterations... -V-V-V-V-V-")
                start_validationForSubepoch_time = time.time()
                
                doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(log,
                                                                              sessionTf,
                                                                            "val",
                                                                            numberOfBatchesValidation, # Computed by the number of extracted samples. So, adapts.
                                                                            cnn3d,
                                                                            subepoch,
                                                                            validationAccuracyMonitorForEpoch,
                                                                            channsOfSegmentsForSubepPerPathwayVal,
                                                                            labelsForCentralOfSegmentsForSubepVal)
                
                end_validationForSubepoch_time = time.time()
                log.print3("TIMING: Validating on the batches of this subepoch #" + str(subepoch) + " took time: "+str(end_validationForSubepoch_time-start_validationForSubepoch_time)+"(s)")
                
            #-------------------END OF THE VALIDATION-DURING-TRAINING-LOOP-------------------------
            
            
            #-------------------------GET DATA FOR THIS SUBEPOCH's TRAINING---------------------------------
            if (not performValidationOnSamplesDuringTrainingProcessBool) and boolItIsTheVeryFirstSubepochOfThisProcess :                    
                [channsOfSegmentsForSubepPerPathwayTrain,
                labelsForCentralOfSegmentsForSubepTrain] = getSampledDataAndLabelsForSubepoch(log,
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
                boolItIsTheVeryFirstSubepochOfThisProcess = False
            else :
                #It was done in parallel with the validation (or with previous training iteration, in case I am not performing validation).
                [channsOfSegmentsForSubepPerPathwayTrain,
                labelsForCentralOfSegmentsForSubepTrain] = parallelJobToGetDataForNextTraining() #fromParallelProcessing that had started from last loop when it was submitted.
            
            numberOfBatchesTraining = len(channsOfSegmentsForSubepPerPathwayTrain[0]) // cnn3d.batchSize["train"] #Computed with number of extracted samples, in case I dont manage to extract as many as I wanted initially.
            
            
            #------------------------SUBMIT PARALLEL JOB TO GET VALIDATION/TRAINING DATA (if val is/not performed) FOR NEXT SUBEPOCH-----------------
            if performValidationOnSamplesDuringTrainingProcessBool :
                #submit the parallel job
                log.print3("PARALLEL: Before Training in subepoch #" +str(subepoch) + ", submitting the parallel job for extracting Segments for the next Validation.")
                parallelJobToGetDataForNextValidation = job_server.submit(getSampledDataAndLabelsForSubepoch, #local function to call and execute in parallel.
                                                                            tupleWithParametersForValidation, #tuple with the arguments required
                                                                            tupleWithLocalFunctionsThatWillBeCalledByTheMainJob, #tuple of local functions that I need to call
                                                                            tupleWithModulesToImportWhichAreUsedByTheJobFunctions) #tuple of the external modules that I need, of which I am calling functions (not the mods of the ext-functions).
            else : #extract in parallel the samples for the next subepoch's training.
                log.print3("PARALLEL: Before Training in subepoch #" +str(subepoch) + ", submitting the parallel job for extracting Segments for the next Training.")
                parallelJobToGetDataForNextTraining = job_server.submit(getSampledDataAndLabelsForSubepoch, #local function to call and execute in parallel.
                                                                            tupleWithParametersForTraining, #tuple with the arguments required
                                                                            tupleWithLocalFunctionsThatWillBeCalledByTheMainJob, #tuple of local functions that I need to call
                                                                            tupleWithModulesToImportWhichAreUsedByTheJobFunctions) #tuple of the external modules that I need, of which I am calling
                
            #-------------------------------START TRAINING IN BATCHES------------------------------
            log.print3("-T-T-T-T-T- Now Training for this subepoch... This may take a few minutes... -T-T-T-T-T-")
            start_trainingForSubepoch_time = time.time()
            
            doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(log,
                                                                          sessionTf,
                                                                        "train",
                                                                        numberOfBatchesTraining,
                                                                        cnn3d,
                                                                        subepoch,
                                                                        trainingAccuracyMonitorForEpoch,
                                                                        channsOfSegmentsForSubepPerPathwayTrain,
                                                                        labelsForCentralOfSegmentsForSubepTrain)
            
            end_trainingForSubepoch_time = time.time()
            log.print3("TIMING: Training on the batches of this subepoch #" + str(subepoch) + " took time: "+str(end_trainingForSubepoch_time-start_trainingForSubepoch_time)+"(s)")
            
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
        
        
        if performFullInferenceOnValidationImagesEveryFewEpochsBool and (model_num_epochs_trained != 0) and (model_num_epochs_trained % everyThatManyEpochsComputeDiceOnTheFullValidationImages == 0) :
            log.print3("***Starting validation with Full Inference / Segmentation on validation subjects for Epoch #"+str(epoch)+"...***")
            
            performInferenceOnWholeVolumes(sessionTf,
                                    cnn3d,
                                    log,
                                    "test",
                                    savePredictionImagesSegmentationAndProbMapsListWhenEvaluatingDiceForValidation,
                                    
                                    listOfFilepathsToEachChannelOfEachPatientValidation,
                                    
                                    providedGtForValidationBool,
                                    listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc,
                                    
                                    providedRoiMaskForValidationBool,
                                    listOfFilepathsToRoiMaskOfEachPatientValidation,
                                    
                                    listOfNamesToGiveToPredictionsIfSavingResults = "Placeholder" if not savePredictionImagesSegmentationAndProbMapsListWhenEvaluatingDiceForValidation else listOfNamesToGiveToPredictionsValidationIfSavingWhenEvalDice,
                                    
                                    #----Preprocessing------
                                    padInputImagesBool=padInputImagesBool,
                                    
                                    #for the cnn extension
                                    useSameSubChannelsAsSingleScale=useSameSubChannelsAsSingleScale,
                                    
                                    listOfFilepathsToEachSubsampledChannelOfEachPatient=listOfFilepathsToEachSubsampledChannelOfEachPatientValidation,
                                    
                                    #--------For FM visualisation---------
                                    saveIndividualFmImagesForVisualisation=saveIndividualFmImagesForVisualisation,
                                    saveMultidimensionalImageWithAllFms=saveMultidimensionalImageWithAllFms,
                                    indicesOfFmsToVisualisePerPathwayTypeAndPerLayer=indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                                    listOfNamesToGiveToFmVisualisationsIfSaving=listOfNamesToGiveToFmVisualisationsIfSaving
                                    )
        
    end_training_time = time.time()
    log.print3("TIMING: Training process took time: "+str(end_training_time-start_training_time)+"(s)")
    log.print3("The whole do_training() function has finished.")
    
    