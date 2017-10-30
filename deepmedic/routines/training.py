# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange

import sys
import time
import pp
import numpy as np

from deepmedic.loggingAndMonitoring.accuracyMonitor import AccuracyOfEpochMonitorSegmentation
from deepmedic.neuralnet.cnnHelpers import CnnWrapperForSampling
from deepmedic.neuralnet.cnnHelpers import dump_cnn_to_gzip_file_dotSave
from deepmedic.dataManagement.sampling import getSampledDataAndLabelsForSubepoch
from deepmedic.routines.testing import performInferenceOnWholeVolumes

from deepmedic.genericHelpers import datetimeNowAsStr

TINY_FLOAT = np.finfo(np.float32).tiny


# The main subroutine of do_training, that runs for every batch of validation and training.
def doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(myLogger,
                                                                train0orValidation1,
                                                                number_of_batches, #This is the integer division of (numb-o-segments/batchSize)
                                                                cnn3dInstance,
                                                                vectorWithWeightsOfTheClassesForCostFunctionOfTraining,
                                                                subepoch,
                                                                accuracyMonitorForEpoch) :
    """
    Returned array is of dimensions [NumberOfClasses x 6]
    For each class: [meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch, meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, meanCostOfSubepoch]
    In the case of VALIDATION, meanCostOfSubepoch is just a placeholder. Only valid when training.
    """
    trainedOrValidatedString = "Trained" if train0orValidation1 == 0 else "Validated"
    
    costsOfBatches = []
    #each row in the array below will hold the number of Real Positives, Real Negatives, True Predicted Positives and True Predicted Negatives in the subepoch, in this order.
    arrayWithNumbersOfPerClassRpRnTpTnInSubepoch = np.zeros([ cnn3dInstance.numberOfOutputClasses, 4 ], dtype="int32")
    
    for batch_i in xrange(number_of_batches):
        printProgressStep = max(1, number_of_batches//5)
        if  batch_i%printProgressStep == 0 :
            myLogger.print3( trainedOrValidatedString + " on "+str(batch_i)+"/"+str(number_of_batches)+" of the batches for this subepoch...")
        if train0orValidation1==0 : #training
            listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining = cnn3dInstance.cnnTrainModel(batch_i, vectorWithWeightsOfTheClassesForCostFunctionOfTraining)
            cnn3dInstance.updateTheMatricesOfTheLayersWithTheLastMusAndVarsForTheMovingAverageOfTheBatchNormInference() #I should put this inside the 3dCNN.
            
            costOfThisBatch = listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining[0]
            listWithNumberOfRpRnPpPnForEachClass = listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining[1:]
            
        else : #validation
            listWithMeanErrorAndRpRnTpTnForEachClassFromValidation = cnn3dInstance.cnnValidateModel(batch_i)
            costOfThisBatch = 999 #placeholder in case of validation.
            listWithNumberOfRpRnPpPnForEachClass = listWithMeanErrorAndRpRnTpTnForEachClassFromValidation[:]
            
        #The returned listWithNumberOfRpRnPpPnForEachClass holds Real Positives, Real Negatives, True Predicted Positives and True Predicted Negatives for all classes in this order, flattened. First RpRnTpTn are for WHOLE "class".
        arrayWithNumberOfRpRnPpPnForEachClassForBatch = np.asarray(listWithNumberOfRpRnPpPnForEachClass, dtype="int32").reshape(arrayWithNumbersOfPerClassRpRnTpTnInSubepoch.shape, order='C')
        
        # To later calculate the mean error and cost over the subepoch
        costsOfBatches.append(costOfThisBatch) #only really used in training.
        arrayWithNumbersOfPerClassRpRnTpTnInSubepoch += arrayWithNumberOfRpRnPpPnForEachClassForBatch
        
    #======== Calculate and Report accuracy over subepoch
    # In case of validation, meanCostOfSubepoch is just a placeholder. Cause this does not get calculated and reported in this case.
    meanCostOfSubepoch = accuracyMonitorForEpoch.NA_PATTERN if (train0orValidation1 == 1) else sum(costsOfBatches) / float(number_of_batches)
    # This function does NOT flip the class-0 background to foreground!
    accuracyMonitorForEpoch.updateMonitorAccuraciesWithNewSubepochEntries(meanCostOfSubepoch, arrayWithNumbersOfPerClassRpRnTpTnInSubepoch)
    accuracyMonitorForEpoch.reportAccuracyForLastSubepoch()
    #Done


#------------------------------ MAIN TRAINING ROUTINE -------------------------------------
def do_training(myLogger,
                fileToSaveTrainedCnnModelTo,
                cnn3dInstance,
                
                performValidationOnSamplesDuringTrainingProcessBool, #REQUIRED FOR AUTO SCHEDULE.
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
                
                borrowFlag,
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
                smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                #-------Data Augmentation-------
                normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                reflectImageWithHalfProbDuringTraining,
                
                useSameSubChannelsAsSingleScale,
                
                listOfFilepathsToEachSubsampledChannelOfEachPatientTraining, # deprecated, not supported
                listOfFilepathsToEachSubsampledChannelOfEachPatientValidation, # deprecated, not supported
                
                #Learning Rate Schedule:
                lowerLrByStable0orAuto1orPredefined2orExponential3Schedule,
                minIncreaseInValidationAccuracyConsideredForLrSchedule,
                numEpochsToWaitBeforeLowerLR,
                divideLrBy,
                lowerLrAtTheEndOfTheseEpochsPredefinedScheduleList,
                exponentialScheduleForLrAndMom,
                
                #Weighting Classes differently in the CNN's cost function during training:
                numberOfEpochsToWeightTheClassesInTheCostFunction,
                
                performFullInferenceOnValidationImagesEveryFewEpochsBool, #Even if not providedGtForValidationBool, inference will be performed if this == True, to save the results, eg for visual.
                everyThatManyEpochsComputeDiceOnTheFullValidationImages=1, # Should not be == 0, except if performFullInferenceOnValidationImagesEveryFewEpochsBool == False
                
                #--------For FM visualisation---------
                saveIndividualFmImagesForVisualisation=False,
                saveMultidimensionalImageWithAllFms=False,
                indicesOfFmsToVisualisePerPathwayTypeAndPerLayer="placeholder",
                listOfNamesToGiveToFmVisualisationsIfSaving="placeholder"
                ):
    
    start_training_time = time.clock()
    
    # Used because I cannot pass cnn3dInstance to the sampling function.
    #This is because the parallel process then loads theano again. And creates problems in the GPU when cnmem is used.
    cnn3dWrapper = CnnWrapperForSampling(cnn3dInstance) 
    
    #---------To run PARALLEL the extraction of parts for the next subepoch---
    ppservers = () # tuple of all parallel python servers to connect with
    job_server = pp.Server(ncpus=1, ppservers=ppservers) # Creates jobserver with automatically detected number of workers
    
    tupleWithParametersForTraining = (myLogger,
                                    0,
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
                                    smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                    normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                    reflectImageWithHalfProbDuringTraining
                                    )
    tupleWithParametersForValidation = (myLogger,
                                    1,
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
                                    smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                    [0, -1,-1,-1], #don't perform intensity-augmentation during validation.
                                    [0,0,0] #don't perform reflection-augmentation during validation.
                                    )
    tupleWithLocalFunctionsThatWillBeCalledByTheMainJob = ( )
    tupleWithModulesToImportWhichAreUsedByTheJobFunctions = ( "from __future__ import absolute_import, print_function, division", "from six.moves import xrange",
                "time", "numpy as np", "from deepmedic.dataManagement.sampling import *" )
    boolItIsTheVeryFirstSubepochOfThisProcess = True #to know so that in the very first I sequencially load the data for it.
    #------End for parallel------
    
    while cnn3dInstance.numberOfEpochsTrained < n_epochs :
        epoch = cnn3dInstance.numberOfEpochsTrained
        
        trainingAccuracyMonitorForEpoch = AccuracyOfEpochMonitorSegmentation(myLogger, 0, cnn3dInstance.numberOfEpochsTrained, cnn3dInstance.numberOfOutputClasses, number_of_subepochs)
        validationAccuracyMonitorForEpoch = None if not performValidationOnSamplesDuringTrainingProcessBool else \
                                        AccuracyOfEpochMonitorSegmentation(myLogger, 1, cnn3dInstance.numberOfEpochsTrained, cnn3dInstance.numberOfOutputClasses, number_of_subepochs ) 
                                        
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~Starting new Epoch! Epoch #"+str(epoch)+"/"+str(n_epochs)+" ~~~~~~~~~~~~~~~~~~~~~~~~~")
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        start_epoch_time = time.clock()
        
        for subepoch in xrange(number_of_subepochs): #per subepoch I randomly load some images in the gpu. Random order.
            myLogger.print3("**************************************************************************************************")
            myLogger.print3("************* Starting new Subepoch: #"+str(subepoch)+"/"+str(number_of_subepochs)+" *************")
            myLogger.print3("**************************************************************************************************")
            
            #-------------------------GET DATA FOR THIS SUBEPOCH's VALIDATION---------------------------------
            
            if performValidationOnSamplesDuringTrainingProcessBool :
                if boolItIsTheVeryFirstSubepochOfThisProcess :
                    [channsOfSegmentsForSubepPerPathwayVal,
                    labelsForCentralOfSegmentsForSubepVal] = getSampledDataAndLabelsForSubepoch(myLogger,
                                                                        1,
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
                                                                        smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                                        normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc=[0,-1,-1,-1],
                                                                        reflectImageWithHalfProbDuringTraining = [0,0,0]
                                                                        )
                    boolItIsTheVeryFirstSubepochOfThisProcess = False
                else : #It was done in parallel with the training of the previous epoch, just grab the results...
                    [channsOfSegmentsForSubepPerPathwayVal,
                    labelsForCentralOfSegmentsForSubepVal] = parallelJobToGetDataForNextValidation() #fromParallelProcessing that had started from last loop when it was submitted.
                    
                #------------------------------LOAD DATA FOR VALIDATION----------------------
                myLogger.print3("Loading Validation data for subepoch #"+str(subepoch)+" on shared variable...")
                start_loadingToGpu_time = time.clock()
                
                numberOfBatchesValidation = len(channsOfSegmentsForSubepPerPathwayVal[0]) // cnn3dInstance.batchSizeValidation #Computed with number of extracted samples, in case I dont manage to extract as many as I wanted initially.
                
                myLogger.print3("DEBUG: For Validation, loading to shared variable that many Segments: " + str(len(channsOfSegmentsForSubepPerPathwayVal[0])))
                
                cnn3dInstance.sharedInpXVal.set_value(channsOfSegmentsForSubepPerPathwayVal[0], borrow=borrowFlag) # Primary pathway
                for index in xrange(len(channsOfSegmentsForSubepPerPathwayVal[1:])) :
                    cnn3dInstance.sharedInpXPerSubsListVal[index].set_value(channsOfSegmentsForSubepPerPathwayVal[1+index], borrow=borrowFlag)
                cnn3dInstance.sharedLabelsYVal.set_value(labelsForCentralOfSegmentsForSubepVal, borrow=borrowFlag)
                channsOfSegmentsForSubepPerPathwayVal = ""
                labelsForCentralOfSegmentsForSubepVal = ""
                
                end_loadingToGpu_time = time.clock()
                myLogger.print3("TIMING: Loading sharedVariables for Validation in epoch|subepoch="+str(epoch)+"|"+str(subepoch)+" took time: "+str(end_loadingToGpu_time-start_loadingToGpu_time)+"(s)")
                
                
                #------------------------SUBMIT PARALLEL JOB TO GET TRAINING DATA FOR NEXT TRAINING-----------------
                #submit the parallel job
                myLogger.print3("PARALLEL: Before Validation in subepoch #" +str(subepoch) + ", the parallel job for extracting Segments for the next Training is submitted.")
                parallelJobToGetDataForNextTraining = job_server.submit(getSampledDataAndLabelsForSubepoch, #local function to call and execute in parallel.
                                                                        tupleWithParametersForTraining, #tuple with the arguments required
                                                                        tupleWithLocalFunctionsThatWillBeCalledByTheMainJob, #tuple of local functions that I need to call
                                                                        tupleWithModulesToImportWhichAreUsedByTheJobFunctions) #tuple of the external modules that I need, of which I am calling functions (not the mods of the ext-functions).
                
                #------------------------------------DO VALIDATION--------------------------------
                myLogger.print3("-V-V-V-V-V- Now Validating for this subepoch before commencing the training iterations... -V-V-V-V-V-")
                start_validationForSubepoch_time = time.clock()
                
                train0orValidation1 = 1 #validation
                vectorWithWeightsOfTheClassesForCostFunctionOfTraining = 'placeholder' #only used in training
                
                doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(myLogger,
                                                                            train0orValidation1,
                                                                            numberOfBatchesValidation, # Computed by the number of extracted samples. So, adapts.
                                                                            cnn3dInstance,
                                                                            vectorWithWeightsOfTheClassesForCostFunctionOfTraining,
                                                                            subepoch,
                                                                            validationAccuracyMonitorForEpoch)
                cnn3dInstance.freeGpuValidationData()
                
                end_validationForSubepoch_time = time.clock()
                myLogger.print3("TIMING: Validating on the batches of this subepoch #" + str(subepoch) + " took time: "+str(end_validationForSubepoch_time-start_validationForSubepoch_time)+"(s)")
                
                #Update cnn's top achieved validation accuracy if needed: (for the autoReduction of Learning Rate.)
                cnn3dInstance.checkMeanValidationAccOfLastEpochAndUpdateCnnsTopAccAchievedIfNeeded(myLogger,
                                                                                    validationAccuracyMonitorForEpoch.getMeanEmpiricalAccuracyOfEpoch(),
                                                                                    minIncreaseInValidationAccuracyConsideredForLrSchedule)
            #-------------------END OF THE VALIDATION-DURING-TRAINING-LOOP-------------------------
            
            
            #-------------------------GET DATA FOR THIS SUBEPOCH's TRAINING---------------------------------
            if (not performValidationOnSamplesDuringTrainingProcessBool) and boolItIsTheVeryFirstSubepochOfThisProcess :                    
                [channsOfSegmentsForSubepPerPathwayTrain,
                labelsForCentralOfSegmentsForSubepTrain] = getSampledDataAndLabelsForSubepoch(myLogger,
                                                                        0,
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
                                                                        smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                                        normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                                        reflectImageWithHalfProbDuringTraining
                                                                        )
                boolItIsTheVeryFirstSubepochOfThisProcess = False
            else :
                #It was done in parallel with the validation (or with previous training iteration, in case I am not performing validation).
                [channsOfSegmentsForSubepPerPathwayTrain,
                labelsForCentralOfSegmentsForSubepTrain] = parallelJobToGetDataForNextTraining() #fromParallelProcessing that had started from last loop when it was submitted.
                
            #-------------------------COMPUTE CLASS-WEIGHTS, TO WEIGHT COST FUNCTION AND COUNTER CLASS IMBALANCE----------------------
            #Do it for only few epochs, until I get to an ok local minima neighbourhood.
            if cnn3dInstance.numberOfEpochsTrained < numberOfEpochsToWeightTheClassesInTheCostFunction :
                numOfPatchesInTheSubepoch_notParts = np.prod(labelsForCentralOfSegmentsForSubepTrain.shape)
                actualNumOfPatchesPerClassInTheSubepoch_notParts = np.bincount(np.ravel(labelsForCentralOfSegmentsForSubepTrain).astype(int))
                # yx - y1 = (x - x1) * (y2 - y1)/(x2 - x1)
                # yx = the multiplier I currently want, y1 = the multiplier at the begining, y2 = the multiplier at the end
                # x = current epoch, x1 = epoch where linear decrease starts, x2 = epoch where linear decrease ends
                y1 = (1./(actualNumOfPatchesPerClassInTheSubepoch_notParts+TINY_FLOAT)) * (numOfPatchesInTheSubepoch_notParts*1.0/cnn3dInstance.numberOfOutputClasses)
                y2 = 1.
                x1 = 0. * number_of_subepochs # linear decrease starts from epoch=0
                x2 = numberOfEpochsToWeightTheClassesInTheCostFunction * number_of_subepochs
                x = cnn3dInstance.numberOfEpochsTrained * number_of_subepochs + subepoch
                yx = (x - x1) * (y2 - y1)/(x2 - x1) + y1
                vectorWithWeightsOfTheClassesForCostFunctionOfTraining = np.asarray(yx, dtype="float32")
                myLogger.print3("UPDATE: [Weight of Classes] Setting the weights of the classes in the cost function to: " +str(vectorWithWeightsOfTheClassesForCostFunctionOfTraining))
            else :
                vectorWithWeightsOfTheClassesForCostFunctionOfTraining = np.ones(cnn3dInstance.numberOfOutputClasses, dtype='float32')
                
            #------------------- Learning Rate Schedule ------------------------
            # I must make a learning-rate-manager to encapsulate all these... Very ugly currently... All othere LR schedules are at the outer loop, per epoch.
            if (lowerLrByStable0orAuto1orPredefined2orExponential3Schedule == 4) :
                myLogger.print3("DEBUG: Going to change Learning Rate according to POLY schedule:")
                #newLearningRate = initLr * ( 1 - iter/max_iter) ^ power. Power = 0.9 in parsenet, which we validated to behave ok.
                currentIteration = cnn3dInstance.numberOfEpochsTrained * number_of_subepochs + subepoch
                max_iterations = n_epochs * number_of_subepochs
                newLearningRate = cnn3dInstance.initialLearningRate * pow( 1.0 - 1.0*currentIteration/max_iterations , 0.9)
                myLogger.print3("DEBUG: new learning rate was calculated: " +str(newLearningRate))
                cnn3dInstance.change_learning_rate_of_a_cnn(newLearningRate, myLogger)
                
            #----------------------------------LOAD TRAINING DATA ON GPU-------------------------------
            myLogger.print3("Loading Training data for subepoch #"+str(subepoch)+" on shared variable...")
            start_loadingToGpu_time = time.clock()
            
            numberOfBatchesTraining = len(channsOfSegmentsForSubepPerPathwayTrain[0]) // cnn3dInstance.batchSize #Computed with number of extracted samples, in case I dont manage to extract as many as I wanted initially.
            
            cnn3dInstance.sharedInpXTrain.set_value(channsOfSegmentsForSubepPerPathwayTrain[0], borrow=borrowFlag) # Primary pathway
            for index in xrange(len(channsOfSegmentsForSubepPerPathwayTrain[1:])) :
                cnn3dInstance.sharedInpXPerSubsListTrain[index].set_value(channsOfSegmentsForSubepPerPathwayTrain[1+index], borrow=borrowFlag)
            cnn3dInstance.sharedLabelsYTrain.set_value(labelsForCentralOfSegmentsForSubepTrain, borrow=borrowFlag)
            channsOfSegmentsForSubepPerPathwayTrain = ""
            labelsForCentralOfSegmentsForSubepTrain = ""
            
            end_loadingToGpu_time = time.clock()
            myLogger.print3("TIMING: Loading sharedVariables for Training in epoch|subepoch="+str(epoch)+"|"+str(subepoch)+" took time: "+str(end_loadingToGpu_time-start_loadingToGpu_time)+"(s)")
            
            
            #------------------------SUBMIT PARALLEL JOB TO GET VALIDATION/TRAINING DATA (if val is/not performed) FOR NEXT SUBEPOCH-----------------
            if performValidationOnSamplesDuringTrainingProcessBool :
                #submit the parallel job
                myLogger.print3("PARALLEL: Before Training in subepoch #" +str(subepoch) + ", submitting the parallel job for extracting Segments for the next Validation.")
                parallelJobToGetDataForNextValidation = job_server.submit(getSampledDataAndLabelsForSubepoch, #local function to call and execute in parallel.
                                                                            tupleWithParametersForValidation, #tuple with the arguments required
                                                                            tupleWithLocalFunctionsThatWillBeCalledByTheMainJob, #tuple of local functions that I need to call
                                                                            tupleWithModulesToImportWhichAreUsedByTheJobFunctions) #tuple of the external modules that I need, of which I am calling functions (not the mods of the ext-functions).
            else : #extract in parallel the samples for the next subepoch's training.
                myLogger.print3("PARALLEL: Before Training in subepoch #" +str(subepoch) + ", submitting the parallel job for extracting Segments for the next Training.")
                parallelJobToGetDataForNextTraining = job_server.submit(getSampledDataAndLabelsForSubepoch, #local function to call and execute in parallel.
                                                                            tupleWithParametersForTraining, #tuple with the arguments required
                                                                            tupleWithLocalFunctionsThatWillBeCalledByTheMainJob, #tuple of local functions that I need to call
                                                                            tupleWithModulesToImportWhichAreUsedByTheJobFunctions) #tuple of the external modules that I need, of which I am calling
                
            #-------------------------------START TRAINING IN BATCHES------------------------------
            myLogger.print3("-T-T-T-T-T- Now Training for this subepoch... This may take a few minutes... -T-T-T-T-T-")
            start_trainingForSubepoch_time = time.clock()
            
            train0orValidation1 = 0 #training
            doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(myLogger,
                                                                        train0orValidation1,
                                                                        numberOfBatchesTraining,
                                                                        cnn3dInstance,
                                                                        vectorWithWeightsOfTheClassesForCostFunctionOfTraining,
                                                                        subepoch,
                                                                        trainingAccuracyMonitorForEpoch)
            cnn3dInstance.freeGpuTrainingData()
            
            end_trainingForSubepoch_time = time.clock()
            myLogger.print3("TIMING: Training on the batches of this subepoch #" + str(subepoch) + " took time: "+str(end_trainingForSubepoch_time-start_trainingForSubepoch_time)+"(s)")
            
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
        myLogger.print3("~~~~~~~~~~~~~~~~~~ Epoch #" + str(epoch) + " finished. Reporting Accuracy over whole epoch. ~~~~~~~~~~~~~~~~~~" )
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
        
        if performValidationOnSamplesDuringTrainingProcessBool :
            validationAccuracyMonitorForEpoch.reportMeanAccyracyOfEpoch()
        trainingAccuracyMonitorForEpoch.reportMeanAccyracyOfEpoch()
        
        del trainingAccuracyMonitorForEpoch; del validationAccuracyMonitorForEpoch;
        
        #=======================Learning Rate Schedule.=========================
        if (lowerLrByStable0orAuto1orPredefined2orExponential3Schedule == 0) and (numEpochsToWaitBeforeLowerLR > 0) and (cnn3dInstance.numberOfEpochsTrained % numEpochsToWaitBeforeLowerLR)==0 :
            # STABLE LR SCHEDULE"
            myLogger.print3("DEBUG: Going to lower Learning Rate because of STABLE schedule! The CNN has now been trained for: " + str(cnn3dInstance.numberOfEpochsTrained) + " epochs. I need to decrease LR every: " + str(numEpochsToWaitBeforeLowerLR) + " epochs.")
            cnn3dInstance.divide_learning_rate_of_a_cnn_by(divideLrBy, myLogger)
        elif (lowerLrByStable0orAuto1orPredefined2orExponential3Schedule == 1) and (numEpochsToWaitBeforeLowerLR > 0) :
            # AUTO LR SCHEDULE!
            if not performValidationOnSamplesDuringTrainingProcessBool : #This flag should have been set True from the start if training should do Auto-schedule. If we get in here, this is a bug.
                myLogger.print3("ERROR: For Auto-schedule I need to be performing validation-on-samples during the training-process. The flag performValidationOnSamplesDuringTrainingProcessBool should have been set to True. Instead it seems it was False and no validation was performed. This is a bug. Contact the developer, this should not have happened. Try another Learning Rate schedule for now! Exiting.")
                exit(1)
            if (cnn3dInstance.numberOfEpochsTrained >= cnn3dInstance.topMeanValidationAccuracyAchievedInEpoch[1] + numEpochsToWaitBeforeLowerLR) and \
                    (cnn3dInstance.numberOfEpochsTrained >= cnn3dInstance.lastEpochAtTheEndOfWhichLrWasLowered + numEpochsToWaitBeforeLowerLR) :
                myLogger.print3("DEBUG: Going to lower Learning Rate because of AUTO schedule! The CNN has now been trained for: " + str(cnn3dInstance.numberOfEpochsTrained) + " epochs. Epoch with last highest achieved validation accuracy: " + str(cnn3dInstance.topMeanValidationAccuracyAchievedInEpoch[1]) + ", and epoch that Learning Rate was last lowered: " + str(cnn3dInstance.lastEpochAtTheEndOfWhichLrWasLowered) + ". I waited for increase in accuracy for: " +str(numEpochsToWaitBeforeLowerLR) + " epochs. Going to lower Learning Rate...")
                cnn3dInstance.divide_learning_rate_of_a_cnn_by(divideLrBy, myLogger)
        elif (lowerLrByStable0orAuto1orPredefined2orExponential3Schedule == 2) and (cnn3dInstance.numberOfEpochsTrained in lowerLrAtTheEndOfTheseEpochsPredefinedScheduleList) :
            #Predefined Schedule.
            myLogger.print3("DEBUG: Going to lower Learning Rate because of PREDEFINED schedule! The CNN has now been trained for: " + str(cnn3dInstance.numberOfEpochsTrained) + " epochs. I need to decrease after that many epochs: " + str(lowerLrAtTheEndOfTheseEpochsPredefinedScheduleList))
            cnn3dInstance.divide_learning_rate_of_a_cnn_by(divideLrBy, myLogger)
        elif (lowerLrByStable0orAuto1orPredefined2orExponential3Schedule == 3 and cnn3dInstance.numberOfEpochsTrained >= exponentialScheduleForLrAndMom[0]) :
            myLogger.print3("DEBUG: Going to lower Learning Rate and Increase Momentum because of EXPONENTIAL schedule! The CNN has now been trained for: " + str(cnn3dInstance.numberOfEpochsTrained) + " epochs.")
            minEpochToLowerLr = exponentialScheduleForLrAndMom[0]          
            #newLearningRate = initialLearningRate * gamma^t. gamma = {t-th}root(valueIwantLrToHaveAtTimepointT / initialLearningRate)
            gammaForExpSchedule = pow( ( cnn3dInstance.initialLearningRate*exponentialScheduleForLrAndMom[1] * 1.0) / cnn3dInstance.initialLearningRate, 1.0 / (n_epochs-minEpochToLowerLr))
            newLearningRate = cnn3dInstance.initialLearningRate * pow(gammaForExpSchedule, cnn3dInstance.numberOfEpochsTrained-minEpochToLowerLr + 1.0)
            #Momentum increased linearly.
            newMomentum = ((cnn3dInstance.numberOfEpochsTrained - minEpochToLowerLr + 1) - (n_epochs-minEpochToLowerLr))*1.0 / (n_epochs - minEpochToLowerLr) * (exponentialScheduleForLrAndMom[2] - cnn3dInstance.initialMomentum) + exponentialScheduleForLrAndMom[2]
            print("DEBUG: new learning rate was calculated: ", newLearningRate, " and new Momentum: ", newMomentum)
            cnn3dInstance.change_learning_rate_of_a_cnn(newLearningRate, myLogger)
            cnn3dInstance.change_momentum_of_a_cnn(newMomentum, myLogger)
            
        #================== Everything for epoch has finished. =======================
        #Training finished. Update the number of epochs that the cnn was trained.
        cnn3dInstance.increaseNumberOfEpochsTrained()
        
        myLogger.print3("SAVING: Epoch #"+str(epoch)+" finished. Saving CNN model.")
        dump_cnn_to_gzip_file_dotSave(cnn3dInstance, fileToSaveTrainedCnnModelTo+"."+datetimeNowAsStr(), myLogger)
        end_epoch_time = time.clock()
        myLogger.print3("TIMING: The whole Epoch #"+str(epoch)+" took time: "+str(end_epoch_time-start_epoch_time)+"(s)")
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of Training Epoch. Model was Saved. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        if performFullInferenceOnValidationImagesEveryFewEpochsBool and (cnn3dInstance.numberOfEpochsTrained != 0) and (cnn3dInstance.numberOfEpochsTrained % everyThatManyEpochsComputeDiceOnTheFullValidationImages == 0) :
            myLogger.print3("***Starting validation with Full Inference / Segmentation on validation subjects for Epoch #"+str(epoch)+"...***")
            validation0orTesting1 = 0
            #do_validation_or_testing(myLogger,
            performInferenceOnWholeVolumes(myLogger,
                                    validation0orTesting1,
                                    savePredictionImagesSegmentationAndProbMapsListWhenEvaluatingDiceForValidation,
                                    cnn3dInstance,
                                    
                                    listOfFilepathsToEachChannelOfEachPatientValidation,
                                    
                                    providedGtForValidationBool,
                                    listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc,
                                    
                                    providedRoiMaskForValidationBool,
                                    listOfFilepathsToRoiMaskOfEachPatientValidation,
                                    
                                    borrowFlag,
                                    listOfNamesToGiveToPredictionsIfSavingResults = "Placeholder" if not savePredictionImagesSegmentationAndProbMapsListWhenEvaluatingDiceForValidation else listOfNamesToGiveToPredictionsValidationIfSavingWhenEvalDice,
                                    
                                    #----Preprocessing------
                                    padInputImagesBool=padInputImagesBool,
                                    smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage=smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                    
                                    #for the cnn extension
                                    useSameSubChannelsAsSingleScale=useSameSubChannelsAsSingleScale,
                                    
                                    listOfFilepathsToEachSubsampledChannelOfEachPatient=listOfFilepathsToEachSubsampledChannelOfEachPatientValidation,
                                    
                                    #--------For FM visualisation---------
                                    saveIndividualFmImagesForVisualisation=saveIndividualFmImagesForVisualisation,
                                    saveMultidimensionalImageWithAllFms=saveMultidimensionalImageWithAllFms,
                                    indicesOfFmsToVisualisePerPathwayTypeAndPerLayer=indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                                    listOfNamesToGiveToFmVisualisationsIfSaving=listOfNamesToGiveToFmVisualisationsIfSaving
                                    )
            
    dump_cnn_to_gzip_file_dotSave(cnn3dInstance, fileToSaveTrainedCnnModelTo+".final."+datetimeNowAsStr(), myLogger)
    
    end_training_time = time.clock()
    myLogger.print3("TIMING: Training process took time: "+str(end_training_time-start_training_time)+"(s)")
    myLogger.print3("The whole do_training() function has finished.")
    
    