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
from deepmedic.dataManagement.sampling import extractDataOfSegmentsUsingSampledSliceCoords
from deepmedic.image.io import savePredImgToNiiWithOriginalHdr, saveFmImgToNiiWithOriginalHdr, save4DImgWithAllFmsToNiiWithOriginalHdr
from deepmedic.image.processing import unpadCnnOutputs

from deepmedic.neuralnet.pathwayTypes import PathwayTypes as pt
from deepmedic.logging.utils import strListFl4fNA, getMeanPerColOf2dListExclNA


# Main routine for testing.
def inferenceWholeVolumes(  sessionTf,
                            cnn3d,
                            log,
                            val_or_test,
                            savePredictedSegmAndProbsDict,
                            listOfFilepathsToEachChannelOfEachPatient,
                            providedGtLabelsBool, #boolean. DSC calculation will be performed if this is provided.
                            listOfFilepathsToGtLabelsOfEachPatient,
                            providedRoiMaskForFastInfBool,
                            listOfFilepathsToRoiMaskFastInfOfEachPatient,
                            namesForSavingSegmAndProbs,
                            suffixForSegmAndProbsDict,
                            
                            #----Preprocessing------
                            padInputImagesBool,
                            
                            useSameSubChannelsAsSingleScale,
                            listOfFilepathsToEachSubsampledChannelOfEachPatient,
                            
                            #--------For FM visualisation---------
                            saveIndividualFmImagesForVisualisation,
                            saveMultidimensionalImageWithAllFms,
                            indicesOfFmsToVisualisePerPathwayTypeAndPerLayer,
                            namesForSavingFms ) :
    # saveIndividualFmImagesForVisualisation: should contain an entry per pathwayType, even if just []...
    #       ... If not [], the list should contain one entry per layer of the pathway, even if just [].
    #       ... The layer entries, if not [], they should have to integers, lower and upper FM to visualise. Excluding the highest index.
    
    validation_or_testing_str = "Validation" if val_or_test == "val" else "Testing"
    log.print3("###########################################################################################################")
    log.print3("############################# Starting full Segmentation of " + str(validation_or_testing_str) + " subjects ##########################")
    log.print3("###########################################################################################################")
    
    start_time = time.time()
    
    NA_PATTERN = AccuracyOfEpochMonitorSegmentation.NA_PATTERN
    
    NUMBER_OF_CLASSES = cnn3d.num_classes
    
    total_number_of_images = len(listOfFilepathsToEachChannelOfEachPatient)    
    batch_size = cnn3d.batchSize["test"]
    
    #one dice score for whole + for each class)
    # A list of dimensions: total_number_of_images X NUMBER_OF_CLASSES
    diceCoeffs1 = [ [-1] * NUMBER_OF_CLASSES for i in range(total_number_of_images) ] #AllpredictedLes/AllLesions
    diceCoeffs2 = [ [-1] * NUMBER_OF_CLASSES for i in range(total_number_of_images) ] #predictedInsideRoiMask/AllLesions
    diceCoeffs3 = [ [-1] * NUMBER_OF_CLASSES for i in range(total_number_of_images) ] #predictedInsideRoiMask/ LesionsInsideRoiMask (for comparisons)
    
    recFieldCnn = cnn3d.recFieldCnn
    
    #stride is how much I move in each dimension to acquire the next imagePart. 
    #I move exactly the number I segment in the centre of each image part (originally this was 9^3 segmented per imagePart).
    numberOfCentralVoxelsClassified = cnn3d.finalTargetLayer.outputShape["test"][2:]
    strideOfImagePartsPerDimensionInVoxels = numberOfCentralVoxelsClassified
    
    rczHalfRecFieldCnn = [ (recFieldCnn[i]-1)//2 for i in range(3) ]
    
    #Find the total number of feature maps that will be created:
    #NOTE: saveIndividualFmImagesForVisualisation should contain an entry per pathwayType, even if just []. If not [], the list should contain one entry per layer of the pathway, even if just []. The layer entries, if not [], they should have to integers, lower and upper FM to visualise.
    if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
        totalNumberOfFMsToProcess = 0
        for pathway in cnn3d.pathways :
            indicesOfFmsToVisualisePerLayerOfCertainPathway = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ]
            if indicesOfFmsToVisualisePerLayerOfCertainPathway != [] :
                for layer_i in range(len(pathway.getLayers())) :
                    indicesOfFmsToVisualiseForCertainLayerOfCertainPathway = indicesOfFmsToVisualisePerLayerOfCertainPathway[layer_i]
                    if indicesOfFmsToVisualiseForCertainLayerOfCertainPathway!=[] :
                        #If the user specifies to grab more feature maps than exist (eg 9999), correct it, replacing it with the number of FMs in the layer.
                        numberOfFeatureMapsInThisLayer = pathway.getLayer(layer_i).getNumberOfFeatureMaps()
                        indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] = min(indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1], numberOfFeatureMapsInThisLayer)
                        totalNumberOfFMsToProcess += indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] - indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[0]
                        
    for image_i in range(total_number_of_images) :
        log.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        log.print3("~~~~~~~~~~~~~~~~~~~~ Segmenting subject with index #"+str(image_i)+" ~~~~~~~~~~~~~~~~~~~~")
        
        #load the image channels in cpu
        
        [imageChannels, #a nparray(channels,dim0,dim1,dim2)
        gtLabelsImage, #only for accurate/correct DICE1-2 calculation
        roiMask,
        arrayWithWeightMapsWhereToSampleForEachCategory, #only used in training. Placeholder here.
        allSubsampledChannelsOfPatientInNpArray,  #a nparray(channels,dim0,dim1,dim2)
        tupleOfPaddingPerAxesLeftRight #( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). All 0s when no padding.
        ] = load_imgs_of_subject(
                                log,
                                None,
                                "test",
                                False, # run_input_checks.
                                image_i,
                                
                                listOfFilepathsToEachChannelOfEachPatient,
                                
                                providedGtLabelsBool,
                                listOfFilepathsToGtLabelsOfEachPatient,
                                num_classes = cnn3d.num_classes,
                                
                                providedWeightMapsToSampleForEachCategory = False, # Says if weightMaps are provided. If true, must provide all. Placeholder in testing.
                                forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient = "placeholder", # Placeholder in testing.
                                
                                providedRoiMaskBool = providedRoiMaskForFastInfBool,
                                listOfFilepathsToRoiMaskOfEachPatient = listOfFilepathsToRoiMaskFastInfOfEachPatient,
                                
                                useSameSubChannelsAsSingleScale = useSameSubChannelsAsSingleScale,
                                usingSubsampledPathways = cnn3d.numSubsPaths > 0,
                                listOfFilepathsToEachSubsampledChannelOfEachPatient = listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                
                                padInputImagesBool = padInputImagesBool,
                                cnnReceptiveField = recFieldCnn, # only used if padInputsBool
                                dimsOfPrimeSegmentRcz = cnn3d.pathways[0].getShapeOfInput("test")[2:], # only used if padInputsBool
                                
                                reflectImageWithHalfProb = [0,0,0]
                                )
        niiDimensions = list(imageChannels[0].shape)
        #The predicted probability-maps for the whole volume, one per class. Will be constructed by stitching together the predictions from each segment.
        predProbMapsPerClass = np.zeros([NUMBER_OF_CLASSES]+niiDimensions, dtype = "float32")
        #create the big array that will hold all the fms (for feature extraction, to save as a big multi-dim image).
        if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
            multidimensionalImageWithAllToBeVisualisedFmsArray =  np.zeros([totalNumberOfFMsToProcess] + niiDimensions, dtype = "float32")
            
        # Tile the image and get all slices of the segments that it fully breaks down to.
        [sliceCoordsOfSegmentsInImage] = getCoordsOfAllSegmentsOfAnImage(log=log,
                                                                        dimsOfPrimarySegment=cnn3d.pathways[0].getShapeOfInput("test")[2:],
                                                                        strideOfSegmentsPerDimInVoxels=strideOfImagePartsPerDimensionInVoxels,
                                                                        batch_size = batch_size,
                                                                        channelsOfImageNpArray = imageChannels,#chans,niiDims
                                                                        roiMask = roiMask )
        
        log.print3("Starting to segment each image-part by calling the cnn.cnnTestModel(i). This part takes a few mins per volume...")
        
        num_segments_for_case = len(sliceCoordsOfSegmentsInImage)
        log.print3("Total number of Segments to process:"+str(num_segments_for_case))
        
        imagePartOfConstructedProbMap_i = 0
        imagePartOfConstructedFeatureMaps_i = 0
        num_batches = num_segments_for_case//batch_size
        extractTimePerSubject = 0; loadingTimePerSubject = 0; fwdPassTimePerSubject = 0
        for batch_i in range(num_batches) :
            
            print_progress_step = max(1, num_batches//5)
            if batch_i == 0 or ((batch_i+1) % print_progress_step) == 0 or (batch_i+1) == num_batches :
                log.print3("Processed "+str((batch_i+1)*batch_size)+"/"+str(num_segments_for_case)+" segments.")
                
            # Extract the data for the segments of this batch. ( I could modularize extractDataOfASegmentFromImagesUsingSampledSliceCoords() of training and use it here as well. )
            start_extract_time = time.time()
            sliceCoordsOfSegmentsInBatch = sliceCoordsOfSegmentsInImage[ batch_i*batch_size : (batch_i+1)*batch_size ]
            [channsOfSegmentsPerPath] = extractDataOfSegmentsUsingSampledSliceCoords(cnn3d=cnn3d,
                                                                                    sliceCoordsOfSegmentsToExtract=sliceCoordsOfSegmentsInBatch,
                                                                                    channelsOfImageNpArray=imageChannels,#chans,niiDims
                                                                                    channelsOfSubsampledImageNpArray=allSubsampledChannelsOfPatientInNpArray,
                                                                                    recFieldCnn=recFieldCnn )
            end_extract_time = time.time()
            extractTimePerSubject += end_extract_time - start_extract_time
            
            # ======= Run the inference ============
            
            ops_to_fetch = cnn3d.get_main_ops('test')
            list_of_ops = [ ops_to_fetch['pred_probs'] ] + ops_to_fetch['list_of_fms_per_layer']
            
            # No loading of data in bulk as in training, cause here it's only 1 batch per iteration.
            start_loading_time = time.time()
            feeds = cnn3d.get_main_feeds('test')
            feeds_dict = { feeds['x'] : np.asarray(channsOfSegmentsPerPath[0], dtype='float32') }
            for path_i in range(len(channsOfSegmentsPerPath[1:])) :
                feeds_dict.update( { feeds['x_sub_'+str(path_i)]: np.asarray(channsOfSegmentsPerPath[1+path_i], dtype='float32') } )
            end_loading_time = time.time()
            loadingTimePerSubject += end_loading_time - start_loading_time
            
            start_testing_time = time.time()
            # Forward pass
            # featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = cnn3d.cnnTestAndVisualiseAllFmsFunction( *input_args_to_net )
            featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = sessionTf.run( fetches=list_of_ops, feed_dict=feeds_dict )
            end_testing_time = time.time()
            fwdPassTimePerSubject += end_testing_time - start_testing_time
            
            predictionForATestBatch = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[0]
            listWithTheFmsOfAllLayersSortedByPathwayTypeForTheBatch = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[1:] # If no FMs visualised, this should return []
            #No reshape needed, cause I now do it internally. But to dimensions (batchSize, FMs, R,C,Z).
            
            #~~~~~~~~~~~~~~~~CONSTRUCT THE PREDICTED PROBABILITY MAPS~~~~~~~~~~~~~~
            #From the results of this batch, create the prediction image by putting the predictions to the correct place in the image.
            for imagePart_in_this_batch_i in range(batch_size) :
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
                indexOfTheLayerInTheReturnedListByTheBatchTraining = 0
                
                for pathway in cnn3d.pathways :
                    for layer_i in range(len(pathway.getLayers())) :
                        if indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ]==[] or indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ][layer_i]==[] :
                            continue
                        indicesOfFmsToExtractFromThisLayer = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ][layer_i]
                        fmsReturnedForATestBatchForCertainLayer = listWithTheFmsOfAllLayersSortedByPathwayTypeForTheBatch[indexOfTheLayerInTheReturnedListByTheBatchTraining]
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
                        for imagePart_in_this_batch_i in range(batch_size) :
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
                        
                        indexOfTheLayerInTheReturnedListByTheBatchTraining += 1
                        
                imagePartOfConstructedFeatureMaps_i += batch_size #all the image parts before this were reconstructed for all layers and feature maps. Next batch-iteration should start from this 
                
            #~~~~~~~~~~~~~~~~~~FINISHED CONSTRUCTING THE FEATURE MAPS FOR VISUALISATION~~~~~~~~~~
        
        log.print3("TIMING: Segmentation of subject: [Extracting:] {0:.2f}".format(extractTimePerSubject) +\
                                                    " [Loading:] {0:.2f}".format(loadingTimePerSubject) +\
                                                    " [ForwardPass:] {0:.2f}".format(fwdPassTimePerSubject) +\
                                                    " [Total:] {0:.2f}".format(extractTimePerSubject+loadingTimePerSubject+fwdPassTimePerSubject) + " secs.")
        
        # ================ SAVE PREDICTIONS =====================
        #== saving predicted segmentations ==
        predSegmentation = np.argmax(predProbMapsPerClass, axis=0) #The segmentation.
        unpaddedPredSegmentation = predSegmentation if not padInputImagesBool else unpadCnnOutputs(predSegmentation, tupleOfPaddingPerAxesLeftRight)
        # Multiply with the below to zero-out anything outside the RoiMask if given. Provided that RoiMask is binary [0,1].
        unpaddedRoiMaskIfGivenElse1 = 1
        if isinstance(roiMask, (np.ndarray)) : #If roiMask was given:
            unpaddedRoiMaskIfGivenElse1 = roiMask if not padInputImagesBool else unpadCnnOutputs(roiMask, tupleOfPaddingPerAxesLeftRight)
            
        if savePredictedSegmAndProbsDict["segm"] == True : #save predicted segmentation
            suffixToAdd = suffixForSegmAndProbsDict["segm"]
            #Save the image. Pass the filename paths of the normal image so that I can dublicate the header info, eg RAS transformation.
            unpaddedPredSegmentationWithinRoi = unpaddedPredSegmentation * unpaddedRoiMaskIfGivenElse1
            savePredImgToNiiWithOriginalHdr( unpaddedPredSegmentationWithinRoi,
                                            namesForSavingSegmAndProbs,
                                            listOfFilepathsToEachChannelOfEachPatient,
                                            image_i,
                                            suffixToAdd,
                                            np.dtype(np.int16),
                                            log )
            
        #== saving probability maps ==
        for class_i in range(0, NUMBER_OF_CLASSES) :
            if (len(savePredictedSegmAndProbsDict["prob"]) >= class_i + 1) and (savePredictedSegmAndProbsDict["prob"][class_i] == True) : #save predicted probMap for class
                suffixToAdd = suffixForSegmAndProbsDict["prob"] + str(class_i)
                #Save the image. Pass the filename paths of the normal image so that I can dublicate the header info, eg RAS transformation.
                predProbMapClassI = predProbMapsPerClass[class_i,:,:,:]
                unpaddedPredProbMapClassI = predProbMapClassI if not padInputImagesBool else unpadCnnOutputs(predProbMapClassI, tupleOfPaddingPerAxesLeftRight)
                unpaddedPredProbMapClassIWithinRoi = unpaddedPredProbMapClassI * unpaddedRoiMaskIfGivenElse1
                savePredImgToNiiWithOriginalHdr( unpaddedPredProbMapClassIWithinRoi,
                                                namesForSavingSegmAndProbs,
                                                listOfFilepathsToEachChannelOfEachPatient,
                                                image_i,
                                                suffixToAdd,
                                                np.dtype(np.float32),
                                                log )
                
        #== saving feature maps ==
        if saveIndividualFmImagesForVisualisation :
            currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray = 0
            for pathway_i in range( len(cnn3d.pathways) ) :
                pathway = cnn3d.pathways[pathway_i]
                indicesOfFmsToVisualisePerLayerOfCertainPathway = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[ pathway.pType() ]
                if indicesOfFmsToVisualisePerLayerOfCertainPathway!=[] :
                    for layer_i in range( len(pathway.getLayers()) ) :
                        indicesOfFmsToVisualiseForCertainLayerOfCertainPathway = indicesOfFmsToVisualisePerLayerOfCertainPathway[layer_i]
                        if indicesOfFmsToVisualiseForCertainLayerOfCertainPathway!=[] :
                            #If the user specifies to grab more feature maps than exist (eg 9999), correct it, replacing it with the number of FMs in the layer.
                            for fmActualNumber in range(indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[0], indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1]) :
                                fmToSave = multidimensionalImageWithAllToBeVisualisedFmsArray[currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray]
                                unpaddedFmToSave = fmToSave if not padInputImagesBool else unpadCnnOutputs(fmToSave, tupleOfPaddingPerAxesLeftRight)
                                saveFmImgToNiiWithOriginalHdr(  unpaddedFmToSave,
                                                                namesForSavingFms,
                                                                listOfFilepathsToEachChannelOfEachPatient,
                                                                image_i,
                                                                pathway_i,
                                                                layer_i,
                                                                fmActualNumber,
                                                                log )
                                
                                currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray += 1
        if saveMultidimensionalImageWithAllFms :
            multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms =  np.transpose(multidimensionalImageWithAllToBeVisualisedFmsArray, (1,2,3, 0) )
            unpaddedMultidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms = multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms if not padInputImagesBool else \
                unpadCnnOutputs(multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms, tupleOfPaddingPerAxesLeftRight)
            #Save a multidimensional Nii image. 3D Image, with the 4th dimension being all the Fms...
            save4DImgWithAllFmsToNiiWithOriginalHdr( unpaddedMultidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms,
                                                    namesForSavingFms,
                                                    listOfFilepathsToEachChannelOfEachPatient,
                                                    image_i,
                                                    log )
        #================= FINISHED SAVING RESULTS ====================
        
        #================= EVALUATE DSC FOR EACH SUBJECT ========================
        if providedGtLabelsBool : # Ground Truth was provided for calculation of DSC. Do DSC calculation.
            log.print3("+++++++++++++++++++++ Reporting Segmentation Metrics for the subject #" + str(image_i) + " ++++++++++++++++++++++++++")
            #Unpad whatever needed.
            unpaddedGtLabelsImage = gtLabelsImage if not padInputImagesBool else unpadCnnOutputs(gtLabelsImage, tupleOfPaddingPerAxesLeftRight)
            #calculate DSC per class.
            for class_i in range(0, NUMBER_OF_CLASSES) :
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
                
            log.print3("ACCURACY: (" + str(validation_or_testing_str) + ") The Per-Class DICE Coefficients for subject with index #"+str(image_i)+" equal: DICE1="+strListFl4fNA(diceCoeffs1[image_i],NA_PATTERN)+" DICE2="+strListFl4fNA(diceCoeffs2[image_i],NA_PATTERN)+" DICE3="+strListFl4fNA(diceCoeffs3[image_i],NA_PATTERN))
            printExplanationsAboutDice(log)
            
    #================= Loops for all patients have finished. Now lets just report the average DSC over all the processed patients. ====================
    if providedGtLabelsBool and total_number_of_images>0 : # Ground Truth was provided for calculation of DSC. Do DSC calculation.
        log.print3("+++++++++++++++++++++++++++++++ Segmentation of all subjects finished +++++++++++++++++++++++++++++++++++")
        log.print3("+++++++++++++++++++++ Reporting Average Segmentation Metrics over all subjects ++++++++++++++++++++++++++")
        meanDiceCoeffs1 = getMeanPerColOf2dListExclNA(diceCoeffs1, NA_PATTERN)
        meanDiceCoeffs2 = getMeanPerColOf2dListExclNA(diceCoeffs2, NA_PATTERN)
        meanDiceCoeffs3 = getMeanPerColOf2dListExclNA(diceCoeffs3, NA_PATTERN)
        log.print3("ACCURACY: (" + str(validation_or_testing_str) + ") The Per-Class average DICE Coefficients over all subjects are: DICE1=" + strListFl4fNA(meanDiceCoeffs1, NA_PATTERN) + " DICE2="+strListFl4fNA(meanDiceCoeffs2, NA_PATTERN)+" DICE3="+strListFl4fNA(meanDiceCoeffs3, NA_PATTERN))
        printExplanationsAboutDice(log)
        
    end_time = time.time()
    log.print3("TIMING: "+validation_or_testing_str+" process lasted: {0:.2f}".format(end_time-start_time)+" secs.")
    
    log.print3("###########################################################################################################")
    log.print3("############################# Finished full Segmentation of " + str(validation_or_testing_str) + " subjects ##########################")
    log.print3("###########################################################################################################")
    return 0


def calculateDiceCoefficient(predictedBinaryLabels, groundTruthBinaryLabels) :
    unionCorrectlyPredicted = predictedBinaryLabels * groundTruthBinaryLabels
    numberOfTruePositives = np.sum(unionCorrectlyPredicted)
    numberOfGtPositives = np.sum(groundTruthBinaryLabels)
    diceCoeff = (2.0 * numberOfTruePositives) / (np.sum(predictedBinaryLabels) + numberOfGtPositives) if numberOfGtPositives!=0 else -1
    return diceCoeff

def printExplanationsAboutDice(log) :
    log.print3("EXPLANATION: DICE1/2/3 are lists with the DICE per class. For Class-0, we calculate DICE for whole foreground, i.e all labels merged, except the background label=0. Useful for multi-class problems.")
    log.print3("EXPLANATION: DICE1 is calculated as segmentation over whole volume VS whole Ground Truth (GT). DICE2 is the segmentation within the ROI vs GT. DICE3 is segmentation within the ROI vs the GT within the ROI.")
    log.print3("EXPLANATION: If an ROI mask has been provided, you should be consulting DICE2 or DICE3.")

    
