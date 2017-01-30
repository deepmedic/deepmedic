# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import numpy as np
from genericHelpers import *
import os

def dump_cnn_to_gzip_file_dotSave(cnn_instance, filenameWithPathToSaveTo, logger=None) :
    filenameWithPathToSaveToDotSave = os.path.abspath(filenameWithPathToSaveTo + ".save")
    cnn_instance.freeGpuTrainingData(); cnn_instance.freeGpuValidationData(); cnn_instance.freeGpuTestingData();
    
    if logger <> None :
        logger.print3("Saving network to: "+str(filenameWithPathToSaveToDotSave))
    else:
        print("Saving network to: "+str(filenameWithPathToSaveToDotSave))
        
    dump_object_to_gzip_file(cnn_instance, filenameWithPathToSaveToDotSave)
    
    if logger <> None :
        logger.print3("Model saved.")
    else:
        print("Model saved.")
        
    return filenameWithPathToSaveToDotSave

def printParametersOfCnnRun(myLogger,
                        costFunctionLetter,
                        niiDimensions, imagePartDimensions, patchDimensions, nkerns, kernelDimensions,
                        nkernsSubsampled, kernelDimensionsSubsampled, subsampleFactor, subsampledImagePartDimensions,
                        fcLayersFMs,
                        
                        dropoutRatesForAllPathways, convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes, 
                        
                        imagePartDimensionsValidation, subsampledImagePartDimensionsValidation,
                        imagePartDimensionsTesting, subsampledImagePartDimensionsTesting,
                        
                        learning_rate, lowerLrByStable0orAuto1schedule, minIncreaseInValidationAccuracyConsideredForLrSchedule, numEpochsToWaitBeforeLowerLR, divideLrBy, 
                        numberOfEpochsToWeightTheClassesInTheCostFunction, momentum,  momentumTypeNONNormalized0orNormalized1, 
                        L1_reg_constant, L2_reg_constant, softmaxTemperature,
                        n_epochs,  number_of_subepochs, n_images_per_subepoch, batchSize, batchSizeValidation, batchSizeTesting,
                        imagePartsLoadedInGpuPerSubepoch, imagePartsLoadedInGpuPerSubepochValidation,
                        percentThatArePositiveSamplesTraining, percentThatArePositiveSamplesValidation,
                        rollingAverageForBatchNormalizationOverThatManyBatches,
                        
                        listOfThePatientsFilepathNames, listOfThePatientsForValidationFilepathNames, channels_filenames,
                        lesion_filename, maskWhereToGetPositiveSamplesDuringTraining_filename, 
                        brain_mask_filename, maskWhereToGetNegativeSamplesDuringTraining_filename,
                        maskWhereToGetPositiveSamplesDuringValidation_filename, maskWhereToGetNegativeSamplesDuringValidation_filename
                        ) :
    myLogger.print3("=================Printing parameters of this run===================")
    myLogger.print3("*****Main architecture of the CNN*****")
    myLogger.print3("costFunctionLetter="+str(costFunctionLetter))
    myLogger.print3("niiDimensions="+str(niiDimensions)+", imagePartDimensions="+str(imagePartDimensions)+", patchDimensions="+str(patchDimensions))
    myLogger.print3("nkerns="+str(nkerns)+" kernelDimensions="+str(kernelDimensions))
    myLogger.print3("nkernsSubsampled="+str(nkernsSubsampled)+", kernelDimensionsSubsampled="+str(kernelDimensionsSubsampled)+", subsampleFactor="+str(subsampleFactor)+", subsampledImagePartDimensions="+str(subsampledImagePartDimensions))
    myLogger.print3("fcLayersFMs="+str(fcLayersFMs))
    
    myLogger.print3("*****Secondary points of the architecture*****")
    myLogger.print3("dropoutRatesForAllPathways="+str(dropoutRatesForAllPathways)+", convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes="+str(convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes))
    
    myLogger.print3("*****Part-dimesnions for efficient validation and testing*****")
    myLogger.print3("imagePartDimensionsValidation="+str(imagePartDimensionsValidation)+" subsampledImagePartDimensionsValidation="+str(subsampledImagePartDimensionsValidation))
    myLogger.print3("imagePartDimensionsTesting="+str(imagePartDimensionsTesting)+" subsampledImagePartDimensionsTesting="+str(subsampledImagePartDimensionsTesting))
    
    myLogger.print3("*****Training Schedule and relevant Parameters*****")
    myLogger.print3("learning_rate="+str(learning_rate)+ " lowerLrByStable0orAuto1schedule=" +str(lowerLrByStable0orAuto1schedule) +" minIncreaseInValidationAccuracyConsideredForLrSchedule=" + str(minIncreaseInValidationAccuracyConsideredForLrSchedule))
    myLogger.print3("numEpochsToWaitBeforeLowerLR="+str(numEpochsToWaitBeforeLowerLR)+", divideLrBy="+str(divideLrBy))
    myLogger.print3("numberOfEpochsToWeightTheClassesInTheCostFunction="+str(numberOfEpochsToWeightTheClassesInTheCostFunction)+", momentum="+str(momentum) + " momentumTypeNONNormalized0orNormalized1=" + str(momentumTypeNONNormalized0orNormalized1))
    myLogger.print3("L1_reg_constant="+str(L1_reg_constant)+", L2_reg_constant="+str(L2_reg_constant))
    myLogger.print3("softmaxTemperature="+str(softmaxTemperature))
    
    myLogger.print3("n_epochs="+str(n_epochs)+", number_of_subepochs="+str(number_of_subepochs)+", n_images_per_subepoch="+str(n_images_per_subepoch))
    myLogger.print3("batchSize="+str(batchSize)+" batchSizeValidation="+str(batchSizeValidation)+" batchSizeTesting="+str(batchSizeTesting))
    myLogger.print3("imagePartsLoadedInGpuPerSubepoch="+str(imagePartsLoadedInGpuPerSubepoch)+", imagePartsLoadedInGpuPerSubepochValidation="+str(imagePartsLoadedInGpuPerSubepochValidation))
    myLogger.print3("rollingAverageForBatchNormalizationOverThatManyBatches="+str(rollingAverageForBatchNormalizationOverThatManyBatches))
    
    myLogger.print3("***** The image files that were used *****")
    myLogger.print3(">>>Filenames of the patients used for TRAINING in this run:"+str(listOfThePatientsFilepathNames))
    myLogger.print3(">>>Filenames of the patients used for VALIDATION in this run:"+str(listOfThePatientsForValidationFilepathNames))
    myLogger.print3(">>>Channels_filenames:"+str(channels_filenames))
    myLogger.print3(">>>lesion_filename:"+str(lesion_filename)+", maskWhereToGetPositiveSamplesDuringTraining="+str(maskWhereToGetPositiveSamplesDuringTraining_filename))
    myLogger.print3(">>>brain_mask_filename="+str(brain_mask_filename)+" maskWhereToGetNegativeSamplesDuringTraining_filename="+str(maskWhereToGetNegativeSamplesDuringTraining_filename))
    myLogger.print3(">>>maskWhereToGetPositiveSamplesDuringValidation_filename="+str(maskWhereToGetPositiveSamplesDuringValidation_filename))
    myLogger.print3(">>>maskWhereToGetNegativeSamplesDuringValidation_filename="+str(maskWhereToGetNegativeSamplesDuringValidation_filename))
    
    myLogger.print3("=================Finished printing all parameters of this model========")
    
    
def calculateSubsampledImagePartDimensionsFromImagePartSizePatchSizeAndSubsampleFactor(imagePartDimensions, patchDimensions, subsampleFactor) :
    """
    This function gives you how big your subsampled-image-part should be, so that it corresponds to the correct number of central-voxels in the normal-part. Currently, it's coupled with the patch-size of the normal-scale. I.e. the subsampled-patch HAS TO BE THE SAME SIZE as the normal-scale, and corresponds to subFactor*patchsize in context.
    When the central voxels are not a multiple of the subFactor, you get ceil(), so +1 sub-patch. When the CNN repeats the pattern, it is giving dimension higher than the central-voxels of the normal-part, but then they are sliced-down to the correct number (in the cnn_make_model function, right after the repeat).        
    This function works like this because of getImagePartFromSubsampledImageForTraining(), which gets a subsampled-image-part by going 1 normal-patch back from the top-left voxel of a normal-scale-part, and then 3 ahead. If I change it to start from the top-left-CENTRAL-voxel back and front, I will be able to decouple the normal-patch size and the subsampled-patch-size. 
    """
    #if patch is 17x17, a 17x17 subPart is cool for 3 voxels with a subsampleFactor. +2 to be ok for the 9x9 centrally classified voxels, so 19x19 sub-part.
    subsampledImagePartDimensions = []
    for rcz_i in xrange(len(imagePartDimensions)) :
        centralVoxelsInThisDimension = imagePartDimensions[rcz_i] - patchDimensions[rcz_i] + 1
        centralVoxelsInThisDimensionForSubsampledPart = int(ceil(centralVoxelsInThisDimension*1.0/subsampleFactor[rcz_i]))
        sizeOfSubsampledImagePartInThisDimension = patchDimensions[rcz_i] + centralVoxelsInThisDimensionForSubsampledPart - 1
        subsampledImagePartDimensions.append(sizeOfSubsampledImagePartInThisDimension)
    return subsampledImagePartDimensions

def calculateReceptiveFieldDimensionsFromKernelsDimListPerLayerForFullyConvCnnWithStrides1(kernDimPerLayerList) :
    if not kernDimPerLayerList : #list is []
        return 0
    
    numberOfDimensions = len(kernDimPerLayerList[0])
    receptiveField = [1]*numberOfDimensions
    for dimension_i in xrange(numberOfDimensions) :
        for layer_i in xrange(len(kernDimPerLayerList)) :
            receptiveField[dimension_i] += kernDimPerLayerList[layer_i][dimension_i] - 1
    return receptiveField

def checkReceptiveFieldFineInComparisonToSegmentSize(receptiveFieldDim, segmentDim) :
    numberOfRFDim = len(receptiveFieldDim)
    numberOfSegmDim = len(segmentDim)
    if numberOfRFDim <> numberOfSegmDim :
        print "ERROR: [in function checkReceptiveFieldFineInComparisonToSegmentSize()] : Receptive field and image segment have different number of dimensions! (should be 3 for both! Exiting!)"
        exit(1)
    for dim_i in xrange(numberOfRFDim) :
        if receptiveFieldDim[dim_i] > segmentDim[dim_i] :
            print "ERROR: [in function checkReceptiveFieldFineInComparisonToSegmentSize()] : The segment-size (input) should be at least as big as the receptive field of the model! This was not found to hold! Dimensions of Receptive Field:", receptiveFieldDim, ". Dimensions of Segment: ", segmentDim
            return False
    return True

def checkKernDimPerLayerCorrect3dAndNumLayers(kernDimensionsPerLayer, numOfLayers) :
    #kernDimensionsPerLayer : a list with sublists. One sublist per layer. Each sublist should have 3 integers, specifying the dimensions of the kernel at the corresponding layer of the pathway. eg: kernDimensionsPerLayer = [ [3,3,3], [3,3,3], [5,5,5] ] 
    if kernDimensionsPerLayer == None or len(kernDimensionsPerLayer) <> numOfLayers :
        return False
    for kernDimInLayer in kernDimensionsPerLayer :
        if len(kernDimInLayer) <> 3 :
            return False
    return True

def checkSubsampleFactorEven(subFactor) :
    for dim_i in xrange(len(subFactor)) :
        if subFactor[dim_i]%2 <> 1 :
            return False
    return True
