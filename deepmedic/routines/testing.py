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

from deepmedic.loggingAndMonitoring.accuracyMonitor import AccuracyOfEpochMonitorSegmentation
from deepmedic.dataManagement.sampling import load_imgs_of_single_case
from deepmedic.dataManagement.sampling import getCoordsOfAllSegmentsOfAnImage
from deepmedic.dataManagement.sampling import extractDataOfSegmentsUsingSampledSliceCoords
from deepmedic.image.io import savePredictedImageToANewNiiWithHeaderFromOther, saveFmActivationImageToANewNiiWithHeaderFromOther, saveMultidimensionalImageWithAllVisualisedFmsToANewNiiWithHeaderFromOther
from deepmedic.image.processing import unpadCnnOutputs

from deepmedic.pathwayTypes import PathwayTypes as pt
from deepmedic.genericHelpers import strListFl4fNA, getMeanPerColOf2dListExclNA


# Main routine for testing.
def performInferenceForTestingOnWholeVolumes(myLogger,
                            validation0orTesting1,
                            savePredictionImagesSegmentationAndProbMapsList,
                            cnn3dInstance,
                            
                            listOfFilepathsToEachChannelOfEachPatient,
                            
                            providedGtLabelsBool, #boolean. DSC calculation will be performed if this is provided.
                            listOfFilepathsToGtLabelsOfEachPatient,
                            
                            providedRoiMaskForFastInfBool,
                            listOfFilepathsToRoiMaskFastInfOfEachPatient,
                            
                            borrowFlag,
                            listOfNamesToGiveToPredictionsIfSavingResults,
                            
                            #----Preprocessing------
                            padInputImagesBool,
                            smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                            
                            useSameSubChannelsAsSingleScale,
                            listOfFilepathsToEachSubsampledChannelOfEachPatient,
                            
                            #--------For FM visualisation---------
                            saveIndividualFmImagesForVisualisation,
                            saveMultidimensionalImageWithAllFms,
                            indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,#NOTE: saveIndividualFmImagesForVisualisation should contain an entry per pathwayType, even if just []. If not [], the list should contain one entry per layer of the pathway, even if just []. The layer entries, if not [], they should have to integers, lower and upper FM to visualise. Excluding the highest index.
                            listOfNamesToGiveToFmVisualisationsIfSaving
                            ) :
    validationOrTestingString = "Validation" if validation0orTesting1 == 0 else "Testing"
    myLogger.print3("###########################################################################################################")
    myLogger.print3("############################# Starting full Segmentation of " + str(validationOrTestingString) + " subjects ##########################")
    myLogger.print3("###########################################################################################################")
    
    start_validationOrTesting_time = time.clock()
    
    NA_PATTERN = AccuracyOfEpochMonitorSegmentation.NA_PATTERN
    
    NUMBER_OF_CLASSES = cnn3dInstance.numberOfOutputClasses
    
    total_number_of_images = len(listOfFilepathsToEachChannelOfEachPatient)    
    batch_size = cnn3dInstance.batchSizeTesting
    
    #one dice score for whole + for each class)
    # A list of dimensions: total_number_of_images X NUMBER_OF_CLASSES
    diceCoeffs1 = [ [-1] * NUMBER_OF_CLASSES for i in xrange(total_number_of_images) ] #AllpredictedLes/AllLesions
    diceCoeffs2 = [ [-1] * NUMBER_OF_CLASSES for i in xrange(total_number_of_images) ] #predictedInsideRoiMask/AllLesions
    diceCoeffs3 = [ [-1] * NUMBER_OF_CLASSES for i in xrange(total_number_of_images) ] #predictedInsideRoiMask/ LesionsInsideRoiMask (for comparisons)
    
    recFieldCnn = cnn3dInstance.recFieldCnn
    
    #stride is how much I move in each dimension to acquire the next imagePart. 
    #I move exactly the number I segment in the centre of each image part (originally this was 9^3 segmented per imagePart).
    numberOfCentralVoxelsClassified = cnn3dInstance.finalTargetLayer.outputShapeTest[2:]
    strideOfImagePartsPerDimensionInVoxels = numberOfCentralVoxelsClassified
    
    rczHalfRecFieldCnn = [ (recFieldCnn[i]-1)//2 for i in xrange(3) ]
    
    #Find the total number of feature maps that will be created:
    #NOTE: saveIndividualFmImagesForVisualisation should contain an entry per pathwayType, even if just []. If not [], the list should contain one entry per layer of the pathway, even if just []. The layer entries, if not [], they should have to integers, lower and upper FM to visualise.
    if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
        totalNumberOfFMsToProcess = 0
        for pathway in cnn3dInstance.pathways :
            indicesOfFmsToVisualisePerLayerOfCertainPathway = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ]
            if indicesOfFmsToVisualisePerLayerOfCertainPathway!=[] :
                for layer_i in xrange(len(pathway.getLayers())) :
                    indicesOfFmsToVisualiseForCertainLayerOfCertainPathway = indicesOfFmsToVisualisePerLayerOfCertainPathway[layer_i]
                    if indicesOfFmsToVisualiseForCertainLayerOfCertainPathway!=[] :
                        #If the user specifies to grab more feature maps than exist (eg 9999), correct it, replacing it with the number of FMs in the layer.
                        numberOfFeatureMapsInThisLayer = pathway.getLayer(layer_i).getNumberOfFeatureMaps()
                        indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] = min(indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1], numberOfFeatureMapsInThisLayer)
                        totalNumberOfFMsToProcess += indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] - indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[0]
                        
    for image_i in xrange(total_number_of_images) :
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~ Segmenting subject with index #"+str(image_i)+" ~~~~~~~~~~~~~~~~~~~~")
        
        #load the image channels in cpu
        
        [imageChannels, #a nparray(channels,dim0,dim1,dim2)
        gtLabelsImage, #only for accurate/correct DICE1-2 calculation
        roiMask,
        arrayWithWeightMapsWhereToSampleForEachCategory, #only used in training. Placeholder here.
        allSubsampledChannelsOfPatientInNpArray,  #a nparray(channels,dim0,dim1,dim2)
        tupleOfPaddingPerAxesLeftRight #( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). All 0s when no padding.
        ] = load_imgs_of_single_case(
                                    myLogger,
                                    2,
                                    
                                    image_i,
                                    
                                    listOfFilepathsToEachChannelOfEachPatient,
                                    
                                    providedGtLabelsBool,
                                    listOfFilepathsToGtLabelsOfEachPatient,
                                    
                                    providedWeightMapsToSampleForEachCategory = False, # Says if weightMaps are provided. If true, must provide all. Placeholder in testing.
                                    forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient = "placeholder", # Placeholder in testing.
                                    
                                    providedRoiMaskBool = providedRoiMaskForFastInfBool,
                                    listOfFilepathsToRoiMaskOfEachPatient = listOfFilepathsToRoiMaskFastInfOfEachPatient,
                                    
                                    useSameSubChannelsAsSingleScale = useSameSubChannelsAsSingleScale,
                                    usingSubsampledPathways = cnn3dInstance.numSubsPaths > 0,
                                    listOfFilepathsToEachSubsampledChannelOfEachPatient = listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                    
                                    padInputImagesBool = padInputImagesBool,
                                    cnnReceptiveField = recFieldCnn, # only used if padInputsBool
                                    dimsOfPrimeSegmentRcz = cnn3dInstance.pathways[0].getShapeOfInput()[2][2:], # only used if padInputsBool
                                    
                                    smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage = smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                    normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc = [0, -1,-1,-1],
                                    reflectImageWithHalfProb = [0,0,0]
                                    )
        niiDimensions = list(imageChannels[0].shape)
        #The predicted probability-maps for the whole volume, one per class. Will be constructed by stitching together the predictions from each segment.
        predProbMapsPerClass = np.zeros([NUMBER_OF_CLASSES]+niiDimensions, dtype = "float32")
        #create the big array that will hold all the fms (for feature extraction, to save as a big multi-dim image).
        if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
            multidimensionalImageWithAllToBeVisualisedFmsArray =  np.zeros([totalNumberOfFMsToProcess] + niiDimensions, dtype = "float32")
            
        # Tile the image and get all slices of the segments that it fully breaks down to.
        [sliceCoordsOfSegmentsInImage] = getCoordsOfAllSegmentsOfAnImage(myLogger=myLogger,
                                                                        dimsOfPrimarySegment=cnn3dInstance.pathways[0].getShapeOfInput()[2][2:],
                                                                        strideOfSegmentsPerDimInVoxels=strideOfImagePartsPerDimensionInVoxels,
                                                                        batch_size = batch_size,
                                                                        channelsOfImageNpArray = imageChannels,#chans,niiDims
                                                                        roiMask = roiMask
                                                                        )
        myLogger.print3("Starting to segment each image-part by calling the cnn.cnnTestModel(i). This part takes a few mins per volume...")
        
        #In the next part, for each imagePart in a batch I get from the cnn a vector with labels for the central voxels of the imagepart (9^3 originally).
        #I will reshape the 9^3 vector to a cube and "put it" in the new-segmentation-image, where it corresponds.
        #I have to find exactly to which voxels these labels correspond to. Consider that the image part is bigger than the 9^3 label box...
        #by half-patch at the top and half-patch at the bottom of each dimension.
        
        #Here I calculate how many imageParts can fit in each r-c-z direction/dimension.
        #It is how many times the stride (originally 9^3) can fit in the niiDimension-1patch (half up, half bottom)
        imagePartsPerRdirection = (niiDimensions[0]-recFieldCnn[0]+1) // strideOfImagePartsPerDimensionInVoxels[0]
        imagePartsPerCdirection = (niiDimensions[1]-recFieldCnn[1]+1) // strideOfImagePartsPerDimensionInVoxels[1]
        imagePartsPerZdirection = (niiDimensions[2]-recFieldCnn[2]+1) // strideOfImagePartsPerDimensionInVoxels[2]
        imagePartsPerZSlice = imagePartsPerRdirection*imagePartsPerCdirection
        
        totalNumberOfImagePartsToProcessForThisImage = len(sliceCoordsOfSegmentsInImage)
        myLogger.print3("Total number of Segments to process:"+str(totalNumberOfImagePartsToProcessForThisImage))
        
        imagePartOfConstructedProbMap_i = 0
        imagePartOfConstructedFeatureMaps_i = 0
        number_of_batches = totalNumberOfImagePartsToProcessForThisImage//batch_size
        extractTimePerSubject = 0; loadingTimePerSubject = 0; fwdPassTimePerSubject = 0
        for batch_i in xrange(number_of_batches) : #batch_size = how many image parts in one batch. Has to be the same with the batch_size it was created with. This is no problem for testing. Could do all at once, or just 1 image part at time.
            
            printProgressStep = max(1, number_of_batches//5)
            if batch_i%printProgressStep == 0:
                myLogger.print3("Processed "+str(batch_i*batch_size)+"/"+str(number_of_batches*batch_size)+" Segments.")
                
            # Extract the data for the segments of this batch. ( I could modularize extractDataOfASegmentFromImagesUsingSampledSliceCoords() of training and use it here as well. )
            start_extract_time = time.clock()
            sliceCoordsOfSegmentsInBatch = sliceCoordsOfSegmentsInImage[ batch_i*batch_size : (batch_i+1)*batch_size ]
            [channsOfSegmentsPerPath] = extractDataOfSegmentsUsingSampledSliceCoords(cnn3dInstance=cnn3dInstance,
                                                                                    sliceCoordsOfSegmentsToExtract=sliceCoordsOfSegmentsInBatch,
                                                                                    channelsOfImageNpArray=imageChannels,#chans,niiDims
                                                                                    channelsOfSubsampledImageNpArray=allSubsampledChannelsOfPatientInNpArray,
                                                                                    recFieldCnn=recFieldCnn
                                                                                    )
            end_extract_time = time.clock()
            extractTimePerSubject += end_extract_time - start_extract_time
            
            # Load the data of the batch on the GPU
            start_loading_time = time.clock()
            cnn3dInstance.sharedInpXTest.set_value(np.asarray(channsOfSegmentsPerPath[0], dtype='float32'), borrow=borrowFlag)
            for index in xrange(len(channsOfSegmentsPerPath[1:])) :
                cnn3dInstance.sharedInpXPerSubsListTest[index].set_value(np.asarray(channsOfSegmentsPerPath[1+index], dtype='float32'), borrow=borrowFlag)
            end_loading_time = time.clock()
            loadingTimePerSubject += end_loading_time - start_loading_time
            
            # Do the inference
            start_training_time = time.clock()
            featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = cnn3dInstance.cnnTestAndVisualiseAllFmsFunction(0)
            end_training_time = time.clock()
            fwdPassTimePerSubject += end_training_time - start_training_time
            
            predictionForATestBatch = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[-1]
            listWithTheFmsOfAllLayersSortedByPathwayTypeForTheBatch = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[:-1]
            #No reshape needed, cause I now do it internally. But to dimensions (batchSize, FMs, R,C,Z).
            
            #~~~~~~~~~~~~~~~~CONSTRUCT THE PREDICTED PROBABILITY MAPS~~~~~~~~~~~~~~
            #From the results of this batch, create the prediction image by putting the predictions to the correct place in the image.
            for imagePart_in_this_batch_i in xrange(batch_size) :
                #Now put the label-cube in the new-label-segmentation-image, at the correct position. 
                #The very first label goes not in index 0,0,0 but half-patch further away! At the position of the central voxel of the top-left patch!
                sliceCoordsOfThisSegment = sliceCoordsOfSegmentsInImage[imagePartOfConstructedProbMap_i]
                coordsOfTopLeftVoxelForThisPart = [ sliceCoordsOfThisSegment[0][0], sliceCoordsOfThisSegment[1][0], sliceCoordsOfThisSegment[2][0] ]
                predProbMapsPerClass[
                        :,
                        coordsOfTopLeftVoxelForThisPart[0] + rczHalfRecFieldCnn[0] : coordsOfTopLeftVoxelForThisPart[0] + rczHalfRecFieldCnn[0] + strideOfImagePartsPerDimensionInVoxels[0],
                        coordsOfTopLeftVoxelForThisPart[1] + rczHalfRecFieldCnn[1] : coordsOfTopLeftVoxelForThisPart[1] + rczHalfRecFieldCnn[1] + strideOfImagePartsPerDimensionInVoxels[1],
                        coordsOfTopLeftVoxelForThisPart[2] + rczHalfRecFieldCnn[2] : coordsOfTopLeftVoxelForThisPart[2] + rczHalfRecFieldCnn[2] + strideOfImagePartsPerDimensionInVoxels[2],
                        ] = predictionForATestBatch[imagePart_in_this_batch_i]
                imagePartOfConstructedProbMap_i += 1
            #~~~~~~~~~~~~~FINISHED CONSTRUCTING THE PREDICTED PROBABILITY MAPS~~~~~~~
            
            #~~~~~~~~~~~~~~CONSTRUCT THE FEATURE MAPS FOR VISUALISATION~~~~~~~~~~~~~~~~~
            if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
                fmsReturnedForATestBatchForCertainLayer = None
                #currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray is the index in the multidimensional array that holds all the to-be-visualised-fms. It is the one that corresponds to the next to-be-visualised indexOfTheLayerInTheReturnedListByTheBatchTraining.
                currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray = 0
                #indexOfTheLayerInTheReturnedListByTheBatchTraining is the index over all the layers in the returned list. I will work only with the ones specified to visualise.
                indexOfTheLayerInTheReturnedListByTheBatchTraining = -1
                
                for pathway in cnn3dInstance.pathways :
                    for layer_i in xrange(len(pathway.getLayers())) :
                        indexOfTheLayerInTheReturnedListByTheBatchTraining += 1
                        if indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ]==[] or indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ][layer_i]==[] :
                            continue
                        indicesOfFmsToExtractFromThisLayer = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ][layer_i]
                        
                        fmsReturnedForATestBatchForCertainLayer = listWithTheFmsOfAllLayersSortedByPathwayTypeForTheBatch[indexOfTheLayerInTheReturnedListByTheBatchTraining][:, indicesOfFmsToExtractFromThisLayer[0]:indicesOfFmsToExtractFromThisLayer[1],:,:,:]
                        #We specify a range of fms to visualise from a layer. currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray : highIndexOfFmsInTheMultidimensionalImageToFillInThisIterationExcluding defines were to put them in the multidimensional-image-array.
                        highIndexOfFmsInTheMultidimensionalImageToFillInThisIterationExcluding = currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray + indicesOfFmsToExtractFromThisLayer[1] - indicesOfFmsToExtractFromThisLayer[0]
                        fmImageInMultidimArrayToReconstructInThisIteration = multidimensionalImageWithAllToBeVisualisedFmsArray[currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray: highIndexOfFmsInTheMultidimensionalImageToFillInThisIterationExcluding]
                        
                        #=========================================================================================================================================
                        #====the following calculations could be move OUTSIDE THE FOR LOOPS, by using the kernel-size parameter (from the cnn instance) instead of the shape of the returned value.
                        #====fmsReturnedForATestBatchForCertainLayer.shape[2] - (numberOfCentralVoxelsClassified[0]-1) is essentially the width of the patch left after the convolutions.
                        #====These calculations are pathway and layer-specific. So they could be done once, prior to image processing, and results cached in a list to be accessed during the loop.
                        numberOfVoxToSubtrToGetPatchWidthAtThisFm_R =  numberOfCentralVoxelsClassified[0]-1 if pathway.pType() != pt.SUBS else int(math.ceil((numberOfCentralVoxelsClassified[0]*1.0)/pathway.subsFactor()[0]) -1)
                        numberOfVoxToSubtrToGetPatchWidthAtThisFm_C =  numberOfCentralVoxelsClassified[1]-1 if pathway.pType() != pt.SUBS else int(math.ceil((numberOfCentralVoxelsClassified[1]*1.0)/pathway.subsFactor()[1]) -1)
                        numberOfVoxToSubtrToGetPatchWidthAtThisFm_Z =  numberOfCentralVoxelsClassified[2]-1 if pathway.pType() != pt.SUBS else int(math.ceil((numberOfCentralVoxelsClassified[2]*1.0)/pathway.subsFactor()[2]) -1)
                        rPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions = fmsReturnedForATestBatchForCertainLayer.shape[2] - numberOfVoxToSubtrToGetPatchWidthAtThisFm_R
                        cPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions = fmsReturnedForATestBatchForCertainLayer.shape[3] - numberOfVoxToSubtrToGetPatchWidthAtThisFm_C
                        zPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions = fmsReturnedForATestBatchForCertainLayer.shape[4] - numberOfVoxToSubtrToGetPatchWidthAtThisFm_Z
                        rOfTopLeftCentralVoxelAtTheFm = (rPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions-1)//2 #-1 so that if width is even, I'll get the left voxel from the centre as 1st, which I THINK is how I am getting the patches from the original image.
                        cOfTopLeftCentralVoxelAtTheFm = (cPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions-1)//2
                        zOfTopLeftCentralVoxelAtTheFm = (zPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions-1)//2
                        
                        #the math.ceil / subsamplingFactor is a trick to make it work for even subsamplingFactor too. Eg 9/2=4.5 => Get 5. Combined with the trick at repeat, I get my correct number of central voxels hopefully.
                        numberOfCentralVoxelsToGetInDirectionR = int(math.ceil((numberOfCentralVoxelsClassified[0]*1.0)/pathway.subsFactor()[0])) if pathway.pType() == pt.SUBS else numberOfCentralVoxelsClassified[0]
                        numberOfCentralVoxelsToGetInDirectionC = int(math.ceil((numberOfCentralVoxelsClassified[1]*1.0)/pathway.subsFactor()[1])) if pathway.pType() == pt.SUBS else numberOfCentralVoxelsClassified[1]
                        numberOfCentralVoxelsToGetInDirectionZ = int(math.ceil((numberOfCentralVoxelsClassified[2]*1.0)/pathway.subsFactor()[2])) if pathway.pType() == pt.SUBS else numberOfCentralVoxelsClassified[2]
                        #=========================================================================================================================================
                        
                        #Grab the central voxels of the predicted fms from the cnn in this batch.
                        centralVoxelsOfAllFmsInLayer = fmsReturnedForATestBatchForCertainLayer[:, #batchsize
                                                            :, #number of featuremaps
                                                            rOfTopLeftCentralVoxelAtTheFm:rOfTopLeftCentralVoxelAtTheFm+numberOfCentralVoxelsToGetInDirectionR,
                                                            cOfTopLeftCentralVoxelAtTheFm:cOfTopLeftCentralVoxelAtTheFm+numberOfCentralVoxelsToGetInDirectionC,
                                                            zOfTopLeftCentralVoxelAtTheFm:zOfTopLeftCentralVoxelAtTheFm+numberOfCentralVoxelsToGetInDirectionZ
                                                            ]
                        #If the pathway that is visualised currently is the subsampled, I need to upsample the central voxels to the normal resolution, before reconstructing the image-fm.
                        if pathway.pType() == pt.SUBS : #subsampled layer. Remember that this returns smaller dimension outputs, cause it works in the subsampled space. I need to repeat it, to bring it to the dimensions of the normal-voxel-space.
                            expandedOutputOfFmsR = np.repeat(centralVoxelsOfAllFmsInLayer, pathway.subsFactor()[0],axis = 2)
                            expandedOutputOfFmsRC = np.repeat(expandedOutputOfFmsR, pathway.subsFactor()[1],axis = 3)
                            expandedOutputOfFmsRCZ = np.repeat(expandedOutputOfFmsRC, pathway.subsFactor()[2],axis = 4)
                            #The below is a trick to get correct number of voxels even when subsampling factor is even or not exact divisor of the number of central voxels.
                            #...This trick is coupled with the ceil() when getting the numberOfCentralVoxelsToGetInDirectionR above.
                            centralVoxelsOfAllFmsToBeVisualisedForWholeBatch = expandedOutputOfFmsRCZ[:,
                                                                                                    :,
                                                                                                    0:numberOfCentralVoxelsClassified[0],
                                                                                                    0:numberOfCentralVoxelsClassified[1],
                                                                                                    0:numberOfCentralVoxelsClassified[2]
                                                                                                    ]
                        else :
                            centralVoxelsOfAllFmsToBeVisualisedForWholeBatch = centralVoxelsOfAllFmsInLayer
                            
                        #----For every image part within this batch, reconstruct the corresponding part of the feature maps of the layer we are currently visualising in this loop.
                        for imagePart_in_this_batch_i in xrange(batch_size) :
                            #Now put the label-cube in the new-label-segmentation-image, at the correct position. 
                            #The very first label goes not in index 0,0,0 but half-patch further away! At the position of the central voxel of the top-left patch!
                            sliceCoordsOfThisSegment = sliceCoordsOfSegmentsInImage[imagePartOfConstructedFeatureMaps_i + imagePart_in_this_batch_i]
                            coordsOfTopLeftVoxelForThisPart = [ sliceCoordsOfThisSegment[0][0], sliceCoordsOfThisSegment[1][0], sliceCoordsOfThisSegment[2][0] ]
                            fmImageInMultidimArrayToReconstructInThisIteration[ # I put the central-predicted-voxels of all FMs to the corresponding, newly created images all at once.
                                    :, #last dimension is the number-of-Fms, I create an image for each.        
                                    coordsOfTopLeftVoxelForThisPart[0] + rczHalfRecFieldCnn[0] : coordsOfTopLeftVoxelForThisPart[0] + rczHalfRecFieldCnn[0] + strideOfImagePartsPerDimensionInVoxels[0],
                                    coordsOfTopLeftVoxelForThisPart[1] + rczHalfRecFieldCnn[1] : coordsOfTopLeftVoxelForThisPart[1] + rczHalfRecFieldCnn[1] + strideOfImagePartsPerDimensionInVoxels[1],
                                    coordsOfTopLeftVoxelForThisPart[2] + rczHalfRecFieldCnn[2] : coordsOfTopLeftVoxelForThisPart[2] + rczHalfRecFieldCnn[2] + strideOfImagePartsPerDimensionInVoxels[2]
                                    ] = centralVoxelsOfAllFmsToBeVisualisedForWholeBatch[imagePart_in_this_batch_i]
                        currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray = highIndexOfFmsInTheMultidimensionalImageToFillInThisIterationExcluding
                imagePartOfConstructedFeatureMaps_i += batch_size #all the image parts before this were reconstructed for all layers and feature maps. Next batch-iteration should start from this 

            #~~~~~~~~~~~~~~~~~~FINISHED CONSTRUCTING THE FEATURE MAPS FOR VISUALISATION~~~~~~~~~~
            
        #Clear GPU from testing data.
        cnn3dInstance.freeGpuTestingData()
        
        myLogger.print3("TIMING: Segmentation of this subject: [Extracting:] "+ str(extractTimePerSubject) +\
                                                            " [Loading:] " + str(loadingTimePerSubject) +\
                                                            " [ForwardPass:] " + str(fwdPassTimePerSubject) +\
                                                            " [Total:] " + str(extractTimePerSubject+loadingTimePerSubject+fwdPassTimePerSubject) + "(s)")
        
        # ================ SAVE PREDICTIONS =====================
        #== saving predicted segmentations ==
        predSegmentation = np.argmax(predProbMapsPerClass, axis=0) #The segmentation.
        unpaddedPredSegmentation = predSegmentation if not padInputImagesBool else unpadCnnOutputs(predSegmentation, tupleOfPaddingPerAxesLeftRight)
        # Multiply with the below to zero-out anything outside the RoiMask if given. Provided that RoiMask is binary [0,1].
        unpaddedRoiMaskIfGivenElse1 = 1
        if isinstance(roiMask, (np.ndarray)) : #If roiMask was given:
            unpaddedRoiMaskIfGivenElse1 = roiMask if not padInputImagesBool else unpadCnnOutputs(roiMask, tupleOfPaddingPerAxesLeftRight)
            
        if savePredictionImagesSegmentationAndProbMapsList[0] == True : #save predicted segmentation
            npDtypeForPredictedImage = np.dtype(np.int16)
            suffixToAdd = "_Segm"
            #Save the image. Pass the filename paths of the normal image so that I can dublicate the header info, eg RAS transformation.
            unpaddedPredSegmentationWithinRoi = unpaddedPredSegmentation * unpaddedRoiMaskIfGivenElse1
            savePredictedImageToANewNiiWithHeaderFromOther( unpaddedPredSegmentationWithinRoi,
                                                            listOfNamesToGiveToPredictionsIfSavingResults,
                                                            listOfFilepathsToEachChannelOfEachPatient,
                                                            image_i,
                                                            suffixToAdd,
                                                            npDtypeForPredictedImage,
                                                            myLogger
                                                            )
        #== saving probability maps ==
        for class_i in xrange(0, NUMBER_OF_CLASSES) :
            if (len(savePredictionImagesSegmentationAndProbMapsList[1]) >= class_i + 1) and (savePredictionImagesSegmentationAndProbMapsList[1][class_i] == True) : #save predicted probMap for class
                npDtypeForPredictedImage = np.dtype(np.float32)
                suffixToAdd = "_ProbMapClass" + str(class_i)
                #Save the image. Pass the filename paths of the normal image so that I can dublicate the header info, eg RAS transformation.
                predProbMapClassI = predProbMapsPerClass[class_i,:,:,:]
                unpaddedPredProbMapClassI = predProbMapClassI if not padInputImagesBool else unpadCnnOutputs(predProbMapClassI, tupleOfPaddingPerAxesLeftRight)
                unpaddedPredProbMapClassIWithinRoi = unpaddedPredProbMapClassI * unpaddedRoiMaskIfGivenElse1
                savePredictedImageToANewNiiWithHeaderFromOther( unpaddedPredProbMapClassIWithinRoi,
                                                                listOfNamesToGiveToPredictionsIfSavingResults,
                                                                listOfFilepathsToEachChannelOfEachPatient,
                                                                image_i,
                                                                suffixToAdd,
                                                                npDtypeForPredictedImage,
                                                                myLogger
                                                                )
        #== saving feature maps ==
        if saveIndividualFmImagesForVisualisation :
            currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray = 0
            for pathway_i in xrange( len(cnn3dInstance.pathways) ) :
                pathway = cnn3dInstance.pathways[pathway_i]
                indicesOfFmsToVisualisePerLayerOfCertainPathway = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ]
                if indicesOfFmsToVisualisePerLayerOfCertainPathway!=[] :
                    for layer_i in xrange( len(pathway.getLayers()) ) :
                        indicesOfFmsToVisualiseForCertainLayerOfCertainPathway = indicesOfFmsToVisualisePerLayerOfCertainPathway[layer_i]
                        if indicesOfFmsToVisualiseForCertainLayerOfCertainPathway!=[] :
                            #If the user specifies to grab more feature maps than exist (eg 9999), correct it, replacing it with the number of FMs in the layer.
                            for fmActualNumber in xrange(indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[0], indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1]) :
                                fmToSave = multidimensionalImageWithAllToBeVisualisedFmsArray[currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray]
                                unpaddedFmToSave = fmToSave if not padInputImagesBool else unpadCnnOutputs(fmToSave, tupleOfPaddingPerAxesLeftRight)
                                saveFmActivationImageToANewNiiWithHeaderFromOther(  unpaddedFmToSave,
                                                                                    listOfNamesToGiveToFmVisualisationsIfSaving,
                                                                                    listOfFilepathsToEachChannelOfEachPatient,
                                                                                    image_i,
                                                                                    pathway_i,
                                                                                    layer_i,
                                                                                    fmActualNumber,
                                                                                    myLogger
                                                                                    ) 
                                currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray += 1
        if saveMultidimensionalImageWithAllFms :
            multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms =  np.transpose(multidimensionalImageWithAllToBeVisualisedFmsArray, (1,2,3, 0) )
            unpaddedMultidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms = multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms if not padInputImagesBool else \
                unpadCnnOutputs(multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms, tupleOfPaddingPerAxesLeftRight)
            #Save a multidimensional Nii image. 3D Image, with the 4th dimension being all the Fms...
            saveMultidimensionalImageWithAllVisualisedFmsToANewNiiWithHeaderFromOther(  unpaddedMultidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms,
                                                                                        listOfNamesToGiveToFmVisualisationsIfSaving,
                                                                                        listOfFilepathsToEachChannelOfEachPatient,
                                                                                        image_i,
                                                                                        myLogger )
        #================= FINISHED SAVING RESULTS ====================
        
        #================= EVALUATE DSC FOR EACH SUBJECT ========================
        if providedGtLabelsBool : # Ground Truth was provided for calculation of DSC. Do DSC calculation.
            myLogger.print3("+++++++++++++++++++++ Reporting Segmentation Metrics for the subject #" + str(image_i) + " ++++++++++++++++++++++++++")
            #Unpad whatever needed.
            unpaddedGtLabelsImage = gtLabelsImage if not padInputImagesBool else unpadCnnOutputs(gtLabelsImage, tupleOfPaddingPerAxesLeftRight)
            #calculate DSC per class.
            for class_i in xrange(0, NUMBER_OF_CLASSES) :
                if class_i == 0 : #in this case, do the evaluation for the segmentation of the WHOLE FOREGROUND (ie, all classes merged except background)
                    binaryPredSegmClassI = unpaddedPredSegmentation > 0 # Merge every class except the background (assumed to be label == 0 )
                    binaryGtLabelClassI = unpaddedGtLabelsImage > 0
                else :
                    binaryPredSegmClassI = unpaddedPredSegmentation == class_i
                    binaryGtLabelClassI = unpaddedGtLabelsImage == class_i
                    
                binaryPredSegmClassIWithinRoi = binaryPredSegmClassI * unpaddedRoiMaskIfGivenElse1
                
                #Calculate the 3 Dices. Dice1 = Allpredicted/allLesions, Dice2 = PredictedWithinRoiMask / AllLesions , Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask.
                #Dice1 = Allpredicted/allLesions
                diceCoeff1 = calculateDiceCoefficient(binaryPredSegmClassI, binaryGtLabelClassI)
                diceCoeffs1[image_i][class_i] = diceCoeff1 if diceCoeff1 != -1 else NA_PATTERN
                #Dice2 = PredictedWithinRoiMask / AllLesions
                diceCoeff2 = calculateDiceCoefficient(binaryPredSegmClassIWithinRoi, binaryGtLabelClassI)
                diceCoeffs2[image_i][class_i] = diceCoeff2 if diceCoeff2 != -1 else NA_PATTERN
                #Dice3 = PredictedWithinRoiMask / LesionsInsideRoiMask
                diceCoeff3 = calculateDiceCoefficient(binaryPredSegmClassIWithinRoi, binaryGtLabelClassI * unpaddedRoiMaskIfGivenElse1)
                diceCoeffs3[image_i][class_i] = diceCoeff3 if diceCoeff3 != -1 else NA_PATTERN
                
            myLogger.print3("ACCURACY: (" + str(validationOrTestingString) + ") The Per-Class DICE Coefficients for subject with index #"+str(image_i)+" equal: DICE1="+strListFl4fNA(diceCoeffs1[image_i],NA_PATTERN)+" DICE2="+strListFl4fNA(diceCoeffs2[image_i],NA_PATTERN)+" DICE3="+strListFl4fNA(diceCoeffs3[image_i],NA_PATTERN))
            printExplanationsAboutDice(myLogger)
            
    #================= Loops for all patients have finished. Now lets just report the average DSC over all the processed patients. ====================
    if providedGtLabelsBool and total_number_of_images>0 : # Ground Truth was provided for calculation of DSC. Do DSC calculation.
        myLogger.print3("+++++++++++++++++++++++++++++++ Segmentation of all subjects finished +++++++++++++++++++++++++++++++++++")
        myLogger.print3("+++++++++++++++++++++ Reporting Average Segmentation Metrics over all subjects ++++++++++++++++++++++++++")
        meanDiceCoeffs1 = getMeanPerColOf2dListExclNA(diceCoeffs1, NA_PATTERN)
        meanDiceCoeffs2 = getMeanPerColOf2dListExclNA(diceCoeffs2, NA_PATTERN)
        meanDiceCoeffs3 = getMeanPerColOf2dListExclNA(diceCoeffs3, NA_PATTERN)
        myLogger.print3("ACCURACY: (" + str(validationOrTestingString) + ") The Per-Class average DICE Coefficients over all subjects are: DICE1=" + strListFl4fNA(meanDiceCoeffs1, NA_PATTERN) + " DICE2="+strListFl4fNA(meanDiceCoeffs2, NA_PATTERN)+" DICE3="+strListFl4fNA(meanDiceCoeffs3, NA_PATTERN))
        printExplanationsAboutDice(myLogger)
        
    end_validationOrTesting_time = time.clock()
    myLogger.print3("TIMING: "+validationOrTestingString+" process took time: "+str(end_validationOrTesting_time-start_validationOrTesting_time)+"(s)")
    
    myLogger.print3("###########################################################################################################")
    myLogger.print3("############################# Finished full Segmentation of " + str(validationOrTestingString) + " subjects ##########################")
    myLogger.print3("###########################################################################################################")


def calculateDiceCoefficient(predictedBinaryLabels, groundTruthBinaryLabels) :
    unionCorrectlyPredicted = predictedBinaryLabels * groundTruthBinaryLabels
    numberOfTruePositives = np.sum(unionCorrectlyPredicted)
    numberOfGtPositives = np.sum(groundTruthBinaryLabels)
    diceCoeff = (2.0 * numberOfTruePositives) / (np.sum(predictedBinaryLabels) + numberOfGtPositives) if numberOfGtPositives!=0 else -1
    return diceCoeff

def printExplanationsAboutDice(myLogger) :
    myLogger.print3("EXPLANATION: DICE1/2/3 are lists with the DICE per class. For Class-0, we calculate DICE for whole foreground, i.e all labels merged, except the background label=0. Useful for multi-class problems.")
    myLogger.print3("EXPLANATION: DICE1 is calculated as segmentation over whole volume VS whole Ground Truth (GT). DICE2 is the segmentation within the ROI vs GT. DICE3 is segmentation within the ROI vs the GT within the ROI.")
    myLogger.print3("EXPLANATION: If an ROI mask has been provided, you should be consulting DICE2 or DICE3.")
    
    