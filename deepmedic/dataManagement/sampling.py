# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange

import time
import numpy as np
import math
import random

from deepmedic.image.io import loadVolume
from deepmedic.image.processing import reflectImageArrayIfNeeded, smoothImageWithGaussianFilterIfNeeded, calculateTheZeroIntensityOf3dImage, padCnnInputs
from deepmedic.pathwayTypes import PathwayTypes as pt


# Order of calls:
# getSampledDataAndLabelsForSubepoch
#    get_random_ind_of_cases_to_train_subep
#    getNumberOfSegmentsToExtractPerCategoryFromEachSubject
#    load_imgs_of_single_case
#    sampleImageParts
#    extractDataOfASegmentFromImagesUsingSampledSliceCoords
#        getImagePartFromSubsampledImageForTraining
#    shuffleTheSegmentsForThisSubepoch

# Main sampling process during training. Executed in parallel while training on a batch on the GPU.
# Called from training.do_training()
def getSampledDataAndLabelsForSubepoch(myLogger,
                                        training0orValidation1,
                                        cnn3d,
                                        maxNumSubjectsLoadedPerSubepoch,
                                        numberOfImagePartsToLoadInGpuPerSubepoch,
                                        samplingTypeInstance,
                                        
                                        listOfFilepathsToEachChannelOfEachPatient,
                                        listOfFilepathsToGtLabelsOfEachPatientTrainOrVal,
                                        
                                        providedRoiMaskBool,
                                        listOfFilepathsToRoiMaskOfEachPatient,
                                        
                                        providedWeightMapsToSampleForEachCategory,
                                        forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient,
                                        
                                        useSameSubChannelsAsSingleScale,
                                        listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                        
                                        padInputImagesBool,
                                        smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                        normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                        reflectImageWithHalfProbDuringTraining
                                        ):
    start_getAllImageParts_time = time.clock()
    
    trainingOrValidationString = "Training" if training0orValidation1 == 0 else "Validation"
    
    myLogger.print3(":=:=:=:=:=:=:=:=: Starting to extract Segments from the images for next " + trainingOrValidationString + "... :=:=:=:=:=:=:=:=:")
    
    total_number_of_subjects = len(listOfFilepathsToEachChannelOfEachPatient)
    randomIndicesList_for_gpu = get_random_ind_of_cases_to_train_subep(total_number_of_subjects = total_number_of_subjects,
                                                                        max_subjects_on_gpu_for_subepoch = maxNumSubjectsLoadedPerSubepoch,
                                                                        get_max_subjects_for_gpu_even_if_total_less = False,
                                                                        myLogger=myLogger)
    myLogger.print3("Out of [" + str(total_number_of_subjects) + "] subjects given for [" + trainingOrValidationString + "], it was specified to extract Segments from maximum [" + str(maxNumSubjectsLoadedPerSubepoch) + "] per subepoch.")
    myLogger.print3("Shuffled indices of subjects that were randomly chosen: "+str(randomIndicesList_for_gpu))
    
    #This is x. Will end up with dimensions: numberOfPathwaysThatTakeInput, partImagesLoadedPerSubepoch, channels, r,c,z, but flattened.
    imagePartsChannelsToLoadOnGpuForSubepochPerPathway = [ [] for i in xrange(cnn3d.getNumPathwaysThatRequireInput()) ]
    gtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch = [] # Labels only for the central/predicted part of segments.
    numOfSubjectsLoadingThisSubepochForSampling = len(randomIndicesList_for_gpu) #Can be different than maxNumSubjectsLoadedPerSubepoch, cause of available images number.
    
    dimsOfPrimeSegmentRcz=cnn3d.pathways[0].getShapeOfInput()[training0orValidation1][2:]
    
    # This is to separate each sampling category (fore/background, uniform, full-image, weighted-classes)
    stringsPerCategoryToSample = samplingTypeInstance.getStringsPerCategoryToSample()
    numberOfCategoriesToSample = samplingTypeInstance.getNumberOfCategoriesToSample()
    percentOfSamplesPerCategoryToSample = samplingTypeInstance.getPercentOfSamplesPerCategoryToSample()
    arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject = getNumberOfSegmentsToExtractPerCategoryFromEachSubject(numberOfImagePartsToLoadInGpuPerSubepoch,
                                                                                                                        percentOfSamplesPerCategoryToSample,
                                                                                                                        numOfSubjectsLoadingThisSubepochForSampling)
    numOfInpChannelsForPrimaryPath = len(listOfFilepathsToEachChannelOfEachPatient[0])
    
    myLogger.print3("SAMPLING: Starting iterations to extract Segments from each subject for next " + trainingOrValidationString + "...")
    
    for index_for_vector_with_images_on_gpu in xrange(0, numOfSubjectsLoadingThisSubepochForSampling) :
        myLogger.print3("SAMPLING: Going to load the images and extract segments from the subject #" + str(index_for_vector_with_images_on_gpu + 1) + "/" +str(numOfSubjectsLoadingThisSubepochForSampling))
        
        [allChannelsOfPatientInNpArray, #a nparray(channels,dim0,dim1,dim2)
        gtLabelsImage,
        roiMask,
        arrayWithWeightMapsWhereToSampleForEachCategory, #can be returned "placeholderNothing" if it's testing phase or not "provided weighted maps". In this case, I will sample from GT/ROI.
        allSubsampledChannelsOfPatientInNpArray,  #a nparray(channels,dim0,dim1,dim2)
        tupleOfPaddingPerAxesLeftRight #( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). All 0s when no padding.
        ] = load_imgs_of_single_case(
                                                myLogger,
                                                training0orValidation1,
                                                
                                                randomIndicesList_for_gpu[index_for_vector_with_images_on_gpu],
                                                
                                                listOfFilepathsToEachChannelOfEachPatient,
                                                
                                                providedGtLabelsBool=True, # If this getTheArr function is called (training), gtLabels should already been provided.
                                                listOfFilepathsToGtLabelsOfEachPatient=listOfFilepathsToGtLabelsOfEachPatientTrainOrVal, 
                                                
                                                providedWeightMapsToSampleForEachCategory = providedWeightMapsToSampleForEachCategory, # Says if weightMaps are provided. If true, must provide all. Placeholder in testing.
                                                forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient = forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient, # Placeholder in testing.
                                                
                                                providedRoiMaskBool = providedRoiMaskBool,
                                                listOfFilepathsToRoiMaskOfEachPatient = listOfFilepathsToRoiMaskOfEachPatient,
                                                
                                                useSameSubChannelsAsSingleScale=useSameSubChannelsAsSingleScale,
                                                
                                                usingSubsampledPathways=cnn3d.numSubsPaths > 0,
                                                listOfFilepathsToEachSubsampledChannelOfEachPatient=listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                                
                                                padInputImagesBool=padInputImagesBool,
                                                cnnReceptiveField=cnn3d.recFieldCnn, # only used if padInputsBool
                                                dimsOfPrimeSegmentRcz=dimsOfPrimeSegmentRcz, # only used if padInputsBool
                                                
                                                smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage = smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc=normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                reflectImageWithHalfProb = reflectImageWithHalfProbDuringTraining
                                                )
        myLogger.print3("DEBUG: Index of this case in the original user-defined list of subjects: " + str(randomIndicesList_for_gpu[index_for_vector_with_images_on_gpu]))
        myLogger.print3("Images for subject loaded.")
        ########################
        #For normalization-augmentation: Get channels' stds if needed:
        stdsOfTheChannsOfThisImage = np.ones(numOfInpChannelsForPrimaryPath, dtype="float32")
        if training0orValidation1 == 0 and normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 2 and\
            normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[1] == 0 : #intensity-augm is to be done, but images are not normalized.
            if providedRoiMaskBool == True :
                stdsOfTheChannsOfThisImage = np.std(allChannelsOfPatientInNpArray[:, roiMask>0], axis=(1,2,3)) #We'll use this for the downsampled version too.
            else : #no brain mask provided:
                stdsOfTheChannsOfThisImage = np.std(allChannelsOfPatientInNpArray, axis=(1,2,3))
        #######################
        
        dimensionsOfImageChannel = allChannelsOfPatientInNpArray[0].shape
        finalWeightMapsToSampleFromPerCategoryForSubject = samplingTypeInstance.logicDecidingAndGivingFinalSamplingMapsForEachCategory(
                                                                                                providedWeightMapsToSampleForEachCategory,
                                                                                                arrayWithWeightMapsWhereToSampleForEachCategory,
                                                                                                
                                                                                                True, #providedGtLabelsBool. True both for training and for validation. Prerequisite from user-interface.
                                                                                                gtLabelsImage,
                                                                                                
                                                                                                providedRoiMaskBool,
                                                                                                roiMask,
                                                                                                
                                                                                                dimensionsOfImageChannel)
        #THE number of imageParts in memory per subepoch does not need to be constant. The batch_size does.
        #But I could have less batches per subepoch if some images dont have lesions I guess. Anyway.
        
        for cat_i in xrange(numberOfCategoriesToSample) :
            catString = stringsPerCategoryToSample[cat_i]
            numOfSegmsToExtractForThisCatFromThisSubject = arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject[cat_i][index_for_vector_with_images_on_gpu]
            finalWeightMapToSampleFromForThisCat = finalWeightMapsToSampleFromPerCategoryForSubject[cat_i]
            
            # Check if the weight map is fully-zeros. In this case, don't call the sampling function, just continue.
            # Note that this way, the data loaded on GPU will not be as much as I initially wanted. Thus calculate number-of-batches from this actual number of extracted segments.
            if np.sum(finalWeightMapToSampleFromForThisCat>0) == 0 :
                myLogger.print3("WARN: The sampling mask/map was found just zeros! No [" + catString + "] image parts were sampled for this subject!")
                continue
            
            myLogger.print3("From subject #"+str(index_for_vector_with_images_on_gpu)+", sampling that many segments of Category [" + catString + "] : " + str(numOfSegmsToExtractForThisCatFromThisSubject) )
            imagePartsSampled = sampleImageParts(myLogger = myLogger,
                                                numOfSegmentsToExtractForThisSubject = numOfSegmsToExtractForThisCatFromThisSubject,
                                                dimsOfSegmentRcz = dimsOfPrimeSegmentRcz,
                                                dimensionsOfImageChannel = dimensionsOfImageChannel, #image dimensions for this subject. All images should have the same.
                                                weightMapToSampleFrom=finalWeightMapToSampleFromForThisCat)
            myLogger.print3("Finished sampling segments of Category [" + catString + "]. Number sampled: " + str( len(imagePartsSampled[0][0]) ) )
            
            # Use the just sampled coordinates of slices to actually extract the segments (data) from the subject's images. 
            for image_part_i in xrange(len(imagePartsSampled[0][0])) :
                coordsOfCentralVoxelOfThisImPart = imagePartsSampled[0][:,image_part_i]
                #sliceCoordsOfThisImagePart = imagePartsSampled[1][:,image_part_i,:] #[0] is the central voxel coords.
                
                [ channelsForThisImagePartPerPathway,
                gtLabelsForTheCentralClassifiedPartOfThisImagePart # used to be gtLabelsForThisImagePart, before extracting only for the central voxels.
                ] = extractDataOfASegmentFromImagesUsingSampledSliceCoords(
                                                                        training0orValidation1,
                                                                        
                                                                        cnn3d,
                                                                        
                                                                        coordsOfCentralVoxelOfThisImPart,
                                                                        numOfInpChannelsForPrimaryPath,
                                                                        
                                                                        allChannelsOfPatientInNpArray,
                                                                        allSubsampledChannelsOfPatientInNpArray,
                                                                        gtLabelsImage,
                                                                        
                                                                        # Intensity Augmentation
                                                                        normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                                        stdsOfTheChannsOfThisImage
                                                                        )
                for pathway_i in xrange(cnn3d.getNumPathwaysThatRequireInput()) :
                    imagePartsChannelsToLoadOnGpuForSubepochPerPathway[pathway_i].append(channelsForThisImagePartPerPathway[pathway_i])
                gtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch.append(gtLabelsForTheCentralClassifiedPartOfThisImagePart)
                
    #I need to shuffle them, together imageParts and lesionParts!
    [imagePartsChannelsToLoadOnGpuForSubepochPerPathway,
    gtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch ] = shuffleTheSegmentsForThisSubepoch( imagePartsChannelsToLoadOnGpuForSubepochPerPathway,
                                                                                                gtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch )
    
    end_getAllImageParts_time = time.clock()
    myLogger.print3("TIMING: Extracting all the Segments for next " + trainingOrValidationString + " took time: "+str(end_getAllImageParts_time-start_getAllImageParts_time)+"(s)")
    
    myLogger.print3(":=:=:=:=:=:=:=:=: Finished extracting Segments from the images for next " + trainingOrValidationString + ". :=:=:=:=:=:=:=:=:")
    
    imagePartsChannelsToLoadOnGpuForSubepochPerPathwayArrays = [ np.asarray(imPartsForPathwayi, dtype="float32") for imPartsForPathwayi in imagePartsChannelsToLoadOnGpuForSubepochPerPathway ]
    return [imagePartsChannelsToLoadOnGpuForSubepochPerPathwayArrays,
            np.asarray(gtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch, dtype="float32") ]
    
    
    
def get_random_ind_of_cases_to_train_subep(total_number_of_subjects, 
                                            max_subjects_on_gpu_for_subepoch, 
                                            get_max_subjects_for_gpu_even_if_total_less=False,
                                            myLogger=None):
    
    subjects_indices = list(range(total_number_of_subjects)) #list() for python3 compatibility, as range cannot get assignment in shuffle()
    random_order_chosen_subjects=[]
    
    random.shuffle(subjects_indices) #does it in place. Now they are shuffled
    
    if max_subjects_on_gpu_for_subepoch>=total_number_of_subjects:
        random_order_chosen_subjects += subjects_indices
        
        if get_max_subjects_for_gpu_even_if_total_less : #This is if I want to have a certain amount on GPU, even if total subjects are less.
            while (len(random_order_chosen_subjects)<max_subjects_on_gpu_for_subepoch):
                random.shuffle(subjects_indices)
                number_of_extra_subjects_to_get_to_fill_gpu = min(max_subjects_on_gpu_for_subepoch - len(random_order_chosen_subjects), total_number_of_subjects)
                random_order_chosen_subjects += (subjects_indices[:number_of_extra_subjects_to_get_to_fill_gpu])
            if len(random_order_chosen_subjects)!=max_subjects_on_gpu_for_subepoch :
                if myLogger!=None :
                    myLogger.print3("ERROR: in get_random_subjects_indices_to_load_on_GPU(), something is wrong!")
                else :
                    print("ERROR: in get_random_subjects_indices_to_load_on_GPU(), something is wrong!")
                exit(1)
    else:
        random_order_chosen_subjects += subjects_indices[:max_subjects_on_gpu_for_subepoch]
        
    return random_order_chosen_subjects



def getNumberOfSegmentsToExtractPerCategoryFromEachSubject( numberOfImagePartsToLoadInGpuPerSubepoch,
                                                            percentOfSamplesPerCategoryToSample, # list with a percentage for each type of category to sample
                                                            numOfSubjectsLoadingThisSubepochForSampling ) :
    numberOfSamplingCategories = len(percentOfSamplesPerCategoryToSample)
    # [numForCat1,..., numForCatN]
    arrayNumberOfSegmentsToExtractPerSamplingCategory = np.zeros( numberOfSamplingCategories, dtype="int32" )
    # [arrayForCat1,..., arrayForCatN] : arrayForCat1 = [ numbOfSegmsToExtrFromSubject1, ...,  numbOfSegmsToExtrFromSubjectK]
    arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject = np.zeros( [ numberOfSamplingCategories, numOfSubjectsLoadingThisSubepochForSampling ] , dtype="int32" )
    
    numberOfSamplesDistributedInTheCategories = 0
    for cat_i in xrange(numberOfSamplingCategories) :
        numberOfSamplesFromThisCategoryPerSubepoch = int(numberOfImagePartsToLoadInGpuPerSubepoch*percentOfSamplesPerCategoryToSample[cat_i])
        arrayNumberOfSegmentsToExtractPerSamplingCategory[cat_i] += numberOfSamplesFromThisCategoryPerSubepoch
        numberOfSamplesDistributedInTheCategories += numberOfSamplesFromThisCategoryPerSubepoch
    # Distribute samples that were left from the rounding error of integer division.
    numOfUndistributedSamples = numberOfImagePartsToLoadInGpuPerSubepoch - numberOfSamplesDistributedInTheCategories
    indicesOfCategoriesToGiveUndistrSamples = np.random.choice(numberOfSamplingCategories, size=numOfUndistributedSamples, replace=True, p=percentOfSamplesPerCategoryToSample)
    for cat_i in indicesOfCategoriesToGiveUndistrSamples : # they will be as many as the undistributed samples
        arrayNumberOfSegmentsToExtractPerSamplingCategory[cat_i] += 1
        
    for cat_i in xrange(numberOfSamplingCategories) :
        numberOfSamplesFromThisCategoryPerSubepochPerImage = arrayNumberOfSegmentsToExtractPerSamplingCategory[cat_i] // numOfSubjectsLoadingThisSubepochForSampling
        arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject[cat_i] += numberOfSamplesFromThisCategoryPerSubepochPerImage
        numberOfSamplesFromThisCategoryPerSubepochLeftUnevenly = arrayNumberOfSegmentsToExtractPerSamplingCategory[cat_i] % numOfSubjectsLoadingThisSubepochForSampling
        for i_unevenSampleFromThisCat in xrange(numberOfSamplesFromThisCategoryPerSubepochLeftUnevenly):
            arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject[cat_i, random.randint(0, numOfSubjectsLoadingThisSubepochForSampling-1)] += 1
            
    return arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject

    
    
# roi_mask_filename and roiMinusLesion_mask_filename can be passed "no". In this case, the corresponding return result is nothing.
# This is so because: the do_training() function only needs the roiMinusLesion_mask, whereas the do_testing() only needs the roi_mask.        
def load_imgs_of_single_case(myLogger,
                             training0orValidation1orTest2,
                             
                             index_of_wanted_image, #THIS IS THE CASE's index!
                             
                             listOfFilepathsToEachChannelOfEachPatient,
                             
                             providedGtLabelsBool,
                             listOfFilepathsToGtLabelsOfEachPatient,
                             
                             providedWeightMapsToSampleForEachCategory, # Says if weightMaps are provided. If true, must provide all. Placeholder in testing.
                             forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient, # Placeholder in testing.
                             
                             providedRoiMaskBool,
                             listOfFilepathsToRoiMaskOfEachPatient,
                             
                             useSameSubChannelsAsSingleScale,
                             usingSubsampledPathways,
                             listOfFilepathsToEachSubsampledChannelOfEachPatient,
                             
                             padInputImagesBool,
                             cnnReceptiveField, # only used if padInputImagesBool
                             dimsOfPrimeSegmentRcz,
                             
                             smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                             normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                             reflectImageWithHalfProb
                             ):
    #listOfNiiFilepathNames: should be a list of lists. Each sublist corresponds to one certain patient-case.
    #...Each sublist should have as many elements(strings-filenamePaths) as numberOfChannels, point to the channels of this patient.
    
    if index_of_wanted_image >= len(listOfFilepathsToEachChannelOfEachPatient) :
        myLogger.print3("ERROR : Function 'load_imgs_of_single_case'-")
        myLogger.print3("------- The argument 'index_of_wanted_image' given is greater than the filenames given for the .nii folders! Exiting.")
        exit(1)
    
    myLogger.print3("Loading subject with 1st channel at: "+str(listOfFilepathsToEachChannelOfEachPatient[index_of_wanted_image][0]))
    
    numberOfNormalScaleChannels = len(listOfFilepathsToEachChannelOfEachPatient[0])
    
    #reflect Image with 50% prob, for each axis:
    reflectFlags = []
    for reflectImageWithHalfProb_dimi in xrange(0, len(reflectImageWithHalfProb)) :
        reflectFlags.append(reflectImageWithHalfProb[reflectImageWithHalfProb_dimi] * random.randint(0,1))
    
    tupleOfPaddingPerAxesLeftRight = ((0,0), (0,0), (0,0)) #This will be given a proper value if padding is performed.
    
    if providedRoiMaskBool :
        fullFilenamePathOfRoiMask = listOfFilepathsToRoiMaskOfEachPatient[index_of_wanted_image]
        roiMask = loadVolume(fullFilenamePathOfRoiMask)
        
        roiMask = reflectImageArrayIfNeeded(reflectFlags, roiMask)
        [roiMask, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(roiMask, cnnReceptiveField, dimsOfPrimeSegmentRcz) if padInputImagesBool else [roiMask, tupleOfPaddingPerAxesLeftRight]
    else :
        roiMask = "placeholderNothing"
        
    #Load the channels of the patient.
    niiDimensions = None
    allChannelsOfPatientInNpArray = None
    #The below has dimensions (channels, 2). Holds per channel: [value to add per voxel for mean norm, value to multiply for std renorm]
    howMuchToAddAndMultiplyForNormalizationAugmentationForEachChannel = np.ones( (numberOfNormalScaleChannels, 2), dtype="float32")
    for channel_i in xrange(numberOfNormalScaleChannels):
        fullFilenamePathOfChannel = listOfFilepathsToEachChannelOfEachPatient[index_of_wanted_image][channel_i]
        if fullFilenamePathOfChannel != "-" : #normal case, filepath was given.
            channelData = loadVolume(fullFilenamePathOfChannel)
                
            channelData = smoothImageWithGaussianFilterIfNeeded(smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage[0], channelData)
            channelData = reflectImageArrayIfNeeded(reflectFlags, channelData) #reflect if flag ==1 .
            [channelData, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(channelData, cnnReceptiveField, dimsOfPrimeSegmentRcz) if padInputImagesBool else [channelData, tupleOfPaddingPerAxesLeftRight]
            
            if not isinstance(allChannelsOfPatientInNpArray, (np.ndarray)) :
                #Initialize the array in which all the channels for the patient will be placed.
                niiDimensions = list(channelData.shape)
                allChannelsOfPatientInNpArray = np.zeros( (numberOfNormalScaleChannels, niiDimensions[0], niiDimensions[1], niiDimensions[2]))
                
            allChannelsOfPatientInNpArray[channel_i] = channelData
        else : # "-" was given in the config-listing file. Do Min-fill!
            myLogger.print3("DEBUG: Zero-filling modality with index [" + str(channel_i) +"].")
            allChannelsOfPatientInNpArray[channel_i] = -4.0
            
        
        
        #-------For Data Augmentation when it comes to normalisation values--------------
        #The normalization-augmentation variable should be [0]==0 for no normAug, eg in the case of validation. [0] == 1 if I want it to be applied on per-image basis.
        if training0orValidation1orTest2 == 0 and normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 1 : #[0] should be ==0 for no normAug, eg in the case of validation
            stdOfChannel = 1.
            if normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[1] == 0 : #[0]==0 means it is not already normalized. Else just use std=1.
                if providedRoiMaskBool :
                    stdOfChannel = np.std(allChannelsOfPatientInNpArray[channel_i][roiMask>0]) #We'll use this for the downsampled version too.
                else : #no roi mask provided:
                    stdOfChannel = np.std(allChannelsOfPatientInNpArray[channel_i])
            #Get parameters by how much to renormalize-augment mean and std.
            #Draw from gaussian
            howMuchToAddAndMultiplyForNormalizationAugmentationForEachChannel[channel_i][0] = random.normalvariate(normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[1][0], normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[1][1]) * stdOfChannel
            howMuchToAddAndMultiplyForNormalizationAugmentationForEachChannel[channel_i][1] = random.normalvariate(normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[2][0], normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[2][1])
            #Renormalize-Augmentation
            valueToAddToEachVoxel = howMuchToAddAndMultiplyForNormalizationAugmentationForEachChannel[channel_i][0]
            valueToMultiplyEachVoxel = howMuchToAddAndMultiplyForNormalizationAugmentationForEachChannel[channel_i][1]
            allChannelsOfPatientInNpArray[channel_i] = (allChannelsOfPatientInNpArray[channel_i] + valueToAddToEachVoxel)*valueToMultiplyEachVoxel
            
    #Load the class labels.
    if providedGtLabelsBool : #For training (exact target labels) or validation on samples labels.
        fullFilenamePathOfGtLabels = listOfFilepathsToGtLabelsOfEachPatient[index_of_wanted_image]
        imageGtLabels = loadVolume(fullFilenamePathOfGtLabels)
        
        imageGtLabels = imageGtLabels if np.issubdtype( imageGtLabels.dtype, np.int ) else np.rint(imageGtLabels).astype("int32")
        imageGtLabels = reflectImageArrayIfNeeded(reflectFlags, imageGtLabels) #reflect if flag ==1 .
        
        [imageGtLabels, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(imageGtLabels, cnnReceptiveField, dimsOfPrimeSegmentRcz) if padInputImagesBool else [imageGtLabels, tupleOfPaddingPerAxesLeftRight]
    else : 
        imageGtLabels = "placeholderNothing" #For validation and testing
        
    if training0orValidation1orTest2 != 2 and providedWeightMapsToSampleForEachCategory==True : # in testing these weightedMaps are never provided, they are for training/validation only.
        numberOfSamplingCategories = len(forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient)
        arrayWithWeightMapsWhereToSampleForEachCategory = np.zeros( [numberOfSamplingCategories] + list(allChannelsOfPatientInNpArray[0].shape), dtype="float32" ) 
        for cat_i in xrange( numberOfSamplingCategories ) :
            filepathsToTheWeightMapsOfAllPatientsForThisCategory = forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient[cat_i]
            filepathToTheWeightMapOfThisPatientForThisCategory = filepathsToTheWeightMapsOfAllPatientsForThisCategory[index_of_wanted_image]
            weightedMapForThisCatData = loadVolume(filepathToTheWeightMapOfThisPatientForThisCategory)
            
            weightedMapForThisCatData = reflectImageArrayIfNeeded(reflectFlags, weightedMapForThisCatData)
            [weightedMapForThisCatData, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(weightedMapForThisCatData, cnnReceptiveField, dimsOfPrimeSegmentRcz) if padInputImagesBool else [weightedMapForThisCatData, tupleOfPaddingPerAxesLeftRight]
            
            arrayWithWeightMapsWhereToSampleForEachCategory[cat_i] = weightedMapForThisCatData
    else :
        arrayWithWeightMapsWhereToSampleForEachCategory = "placeholderNothing"
        
    # The second CNN pathway...
    if not usingSubsampledPathways :
        allSubsampledChannelsOfPatientInNpArray = "placeholderNothing"
    elif useSameSubChannelsAsSingleScale : #Pass this in the configuration file, instead of a list of channel names, to use the same channels as the normal res.
        allSubsampledChannelsOfPatientInNpArray = allChannelsOfPatientInNpArray #np.asarray(allChannelsOfPatientInNpArray, dtype="float32") #Hope this works, to win time in loading. Without copying it did not work.
    else :
        numberOfSubsampledScaleChannels = len(listOfFilepathsToEachSubsampledChannelOfEachPatient[0])
        allSubsampledChannelsOfPatientInNpArray = np.zeros( (numberOfSubsampledScaleChannels, niiDimensions[0], niiDimensions[1], niiDimensions[2]))
        for channel_i in xrange(numberOfSubsampledScaleChannels):
            fullFilenamePathOfChannel = listOfFilepathsToEachSubsampledChannelOfEachPatient[index_of_wanted_image][channel_i]
            channelData = loadVolume(fullFilenamePathOfChannel)
            
            channelData = smoothImageWithGaussianFilterIfNeeded(smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage[1], channelData)
            channelData = reflectImageArrayIfNeeded(reflectFlags, channelData)
            [channelData, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(channelData, cnnReceptiveField, dimsOfPrimeSegmentRcz) if padInputImagesBool else [channelData, tupleOfPaddingPerAxesLeftRight]
            
            allSubsampledChannelsOfPatientInNpArray[channel_i] = channelData
            
            #-------For Data Augmentation when it comes to normalisation values--------------
            if training0orValidation1orTest2 == 0 and normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 1 :
                #Use values  computed on the normal-resolution images. BUT PREREQUISITE IS TO HAVE THE SAME CHANNELS IN THE TWO PATHWAYS. Else need to recompute!
                valueToAddToEachVoxel = howMuchToAddAndMultiplyForNormalizationAugmentationForEachChannel[channel_i][0]
                valueToMultiplyEachVoxel = howMuchToAddAndMultiplyForNormalizationAugmentationForEachChannel[channel_i][1]
                allSubsampledChannelsOfPatientInNpArray[channel_i] = (allSubsampledChannelsOfPatientInNpArray[channel_i] + valueToAddToEachVoxel)*valueToMultiplyEachVoxel
                
    return [allChannelsOfPatientInNpArray, imageGtLabels, roiMask, arrayWithWeightMapsWhereToSampleForEachCategory, allSubsampledChannelsOfPatientInNpArray, tupleOfPaddingPerAxesLeftRight]



#made for 3d
def sampleImageParts(   myLogger,
                        numOfSegmentsToExtractForThisSubject,
                        dimsOfSegmentRcz,
                        dimensionsOfImageChannel,# the dimensions of the images of this subject. All channels etc should have the same dimensions
                        weightMapToSampleFrom
                        ) :
    """
    This function returns the coordinates (index) of the "central" voxel of sampled image parts (1voxel to the left if even part-dimension).
    It also returns the indices of the image parts, left and right indices, INCLUSIVE BOTH SIDES.
    
    Return value: [ rcz-coordsOfCentralVoxelsOfPartsSampled, rcz-sliceCoordsOfImagePartsSampled ]
    > coordsOfCentralVoxelsOfPartsSampled : an array with shape: 3(rcz) x numOfSegmentsToExtractForThisSubject. 
        Example: [ rCoordsForCentralVoxelOfEachPart, cCoordsForCentralVoxelOfEachPart, zCoordsForCentralVoxelOfEachPart ]
        >> r/c/z-CoordsForCentralVoxelOfEachPart : A 1-dim array with numOfSegmentsToExtractForThisSubject, that holds the r-index within the image of each sampled part.
    > sliceCoordsOfImagePartsSampled : 3(rcz) x NumberOfImagePartSamples x 2. The last dimension has [0] for the lower boundary of the slice, and [1] for the higher boundary. INCLUSIVE BOTH SIDES.
        Example: [ r-sliceCoordsOfImagePart, c-sliceCoordsOfImagePart, z-sliceCoordsOfImagePart ]
    """
    # Check if the weight map is fully-zeros. In this case, return no element.
    # Note: Currently, the caller function is checking this case already and does not let this being called. Which is still fine.
    if np.sum(weightMapToSampleFrom>0) == 0 :
        myLogger.print3("WARN: The sampling mask/map was found just zeros! No image parts were sampled for this subject!")
        return [ [[],[],[]], [[],[],[]] ]
    
    imagePartsSampled = []
    
    #Now out of these, I need to randomly select one, which will be an ImagePart's central voxel.
    #But I need to be CAREFUL and get one that IS NOT closer to the image boundaries than the dimensions of the ImagePart permit.
    
    #I look for lesions that are not closer to the image boundaries than the ImagePart dimensions allow.
    #KernelDim is always odd. BUT ImagePart dimensions can be odd or even.
    #If odd, ok, floor(dim/2) from central.
    #If even, dim/2-1 voxels towards the begining of the axis and dim/2 towards the end. Ie, "central" imagePart voxel is 1 closer to begining.
    #BTW imagePartDim takes kernel into account (ie if I want 9^3 voxels classified per imagePart with kernel 5x5, I want 13 dim ImagePart)
    
    halfImagePartBoundaries = np.zeros( (len(dimsOfSegmentRcz), 2) , dtype='int32') #dim1: 1 row per r,c,z. Dim2: left/right width not to sample from (=half segment).
    
    #The below starts all zero. Will be Multiplied by other true-false arrays expressing if the relevant voxels are within boundaries.
    #In the end, the final vector will be true only for the indices of lesions that are within all boundaries.
    booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries = np.zeros(weightMapToSampleFrom.shape, dtype="int32")
    
    #The following loop leads to booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries to be true for the indices that allow you to get an image part CENTERED on them, and be safely within image boundaries. Note that if the imagePart is of even dimension, the "central" voxel is one voxel to the left.
    for rcz_i in xrange( len(dimsOfSegmentRcz) ) :
        if dimsOfSegmentRcz[rcz_i]%2 == 0: #even
            dimensionDividedByTwo = dimsOfSegmentRcz[rcz_i]//2
            halfImagePartBoundaries[rcz_i] = [dimensionDividedByTwo - 1, dimensionDividedByTwo] #central of ImagePart is 1 vox closer to begining of axes.
        else: #odd
            dimensionDividedByTwoFloor = math.floor(dimsOfSegmentRcz[rcz_i]//2) #eg 5/2 = 2, with the 3rd voxel being the "central"
            halfImagePartBoundaries[rcz_i] = [dimensionDividedByTwoFloor, dimensionDividedByTwoFloor] 
    #used to be [halfImagePartBoundaries[0][0]: -halfImagePartBoundaries[0][1]], but in 2D case halfImagePartBoundaries might be ==0, causes problem and you get a null slice.
    booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries[halfImagePartBoundaries[0][0]: dimensionsOfImageChannel[0] - halfImagePartBoundaries[0][1],
                                                            halfImagePartBoundaries[1][0]: dimensionsOfImageChannel[1] - halfImagePartBoundaries[1][1],
                                                            halfImagePartBoundaries[2][0]: dimensionsOfImageChannel[2] - halfImagePartBoundaries[2][1]] = 1
                                                            
    constrainedWithImageBoundariesMaskToSample = weightMapToSampleFrom * booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries
    #normalize the probabilities to sum to 1, cause the function needs it as so.
    constrainedWithImageBoundariesMaskToSample = constrainedWithImageBoundariesMaskToSample / (1.0* np.sum(constrainedWithImageBoundariesMaskToSample))
    
    flattenedConstrainedWithImageBoundariesMaskToSample = constrainedWithImageBoundariesMaskToSample.flatten()
    
    #This is going to be a 3xNumberOfImagePartSamples array.
    indicesInTheFlattenArrayThatWereSampledAsCentralVoxelsOfImageParts = np.random.choice(  constrainedWithImageBoundariesMaskToSample.size,
                                                                                            size = numOfSegmentsToExtractForThisSubject,
                                                                                            replace=True,
                                                                                            p=flattenedConstrainedWithImageBoundariesMaskToSample)
    #np.unravel_index([listOfIndicesInFlattened], dims) returns a tuple of arrays (eg 3 of them if 3 dimImage), where each of the array in the tuple has the same shape as the listOfIndices. They have the r/c/z coords that correspond to the index of the flattened version.
    #So, coordsOfCentralVoxelsOfPartsSampled will end up being an array with shape: 3(rcz) x numOfSegmentsToExtractForThisSubject.
    coordsOfCentralVoxelsOfPartsSampled = np.asarray(np.unravel_index(indicesInTheFlattenArrayThatWereSampledAsCentralVoxelsOfImageParts,
                                                                    constrainedWithImageBoundariesMaskToSample.shape #the shape of the roiMask/scan.
                                                                    )
                                                    )
    #Array with shape: 3(rcz) x NumberOfImagePartSamples x 2. The last dimension has [0] for the lower boundary of the slice, and [1] for the higher boundary. INCLUSIVE BOTH SIDES.
    sliceCoordsOfImagePartsSampled = np.zeros(list(coordsOfCentralVoxelsOfPartsSampled.shape) + [2], dtype="int32")
    sliceCoordsOfImagePartsSampled[:,:,0] = coordsOfCentralVoxelsOfPartsSampled - halfImagePartBoundaries[ :, np.newaxis, 0 ] #np.newaxis broadcasts. To broadcast the -+.
    sliceCoordsOfImagePartsSampled[:,:,1] = coordsOfCentralVoxelsOfPartsSampled + halfImagePartBoundaries[ :, np.newaxis, 1 ]
    """
    The slice coordinates returned are INCLUSIVE BOTH sides.
    """
    #coordsOfCentralVoxelsOfPartsSampled: Array of dimensions 3(rcz) x NumberOfImagePartSamples.
    #sliceCoordsOfImagePartsSampled: Array of dimensions 3(rcz) x NumberOfImagePartSamples x 2. The last dim has [0] for the lower boundary of the slice, and [1] for the higher boundary. INCLUSIVE BOTH SIDES.
    imagePartsSampled = [coordsOfCentralVoxelsOfPartsSampled, sliceCoordsOfImagePartsSampled]
    return imagePartsSampled



def getImagePartFromSubsampledImageForTraining( dimsOfPrimarySegment,
                                                recFieldCnn,
                                                subsampledImageChannels,
                                                image_part_slices_coords,
                                                subSamplingFactor,
                                                subsampledImagePartDimensions
                                                ) :
    """
    This returns an image part from the sampled data, given the image_part_slices_coords, which has the coordinates where the normal-scale image part starts and ends (inclusive).
    Actually, in this case, the right (end) part of image_part_slices_coords is not used.
    
    The way it works is NOT optimal. From the begining of the normal-resolution part, it goes further to the left 1 image PATCH (depending on subsampling factor) and then forward 3 PATCHES. This stops it from being used with arbitrary size of subsampled-image-part (decoupled by the normal-patch). Now, the subsampled patch has to be of the same size as the normal-scale. In order to change this, I should find where THE FIRST TOP LEFT CENTRAL (predicted) VOXEL is, and do the back-one-(sub)patch + front-3-(sub)patches from there, not from the begining of the patch.
    
    Current way it works (correct):
    If I have eg subsample factor=3 and 9 central-pred-voxels, I get 3 "central" voxels/patches for the subsampled-part. Straightforward. If I have a number of central voxels that is not an exact multiple of the subfactor, eg 10 central-voxels, I get 3+1 central voxels in the subsampled-part. When the cnn is convolving them, they will get repeated to 4(last-layer-neurons)*3(factor) = 12, and will get sliced down to 10, in order to have same dimension with the 1st pathway.
    """
    subsampledImageDimensions = subsampledImageChannels[0].shape
    
    subsampledChannelsForThisImagePart = np.ones(   (len(subsampledImageChannels), 
                                                    subsampledImagePartDimensions[0],
                                                    subsampledImagePartDimensions[1],
                                                    subsampledImagePartDimensions[2]), 
                                                dtype = 'float32')
    
    numberOfCentralVoxelsClassifiedForEachImagePart_rDim = dimsOfPrimarySegment[0] - recFieldCnn[0] + 1
    numberOfCentralVoxelsClassifiedForEachImagePart_cDim = dimsOfPrimarySegment[1] - recFieldCnn[1] + 1
    numberOfCentralVoxelsClassifiedForEachImagePart_zDim = dimsOfPrimarySegment[2] - recFieldCnn[2] + 1
    
    #Calculate the slice that I should get, and where I should put it in the imagePart (eg if near the borders, and I cant grab a whole slice-imagePart).
    rSlotsPreviously = ((subSamplingFactor[0]-1)//2)*recFieldCnn[0] if subSamplingFactor[0]%2==1 \
                                                else (subSamplingFactor[0]-2)//2*recFieldCnn[0] + recFieldCnn[0]//2
    cSlotsPreviously = ((subSamplingFactor[1]-1)//2)*recFieldCnn[1] if subSamplingFactor[1]%2==1 \
                                                else (subSamplingFactor[1]-2)//2*recFieldCnn[1] + recFieldCnn[1]//2
    zSlotsPreviously = ((subSamplingFactor[2]-1)//2)*recFieldCnn[2] if subSamplingFactor[2]%2==1 \
                                                else (subSamplingFactor[2]-2)//2*recFieldCnn[2] + recFieldCnn[2]//2
    #1*17
    rToCentralVoxelOfAnAveragedArea = subSamplingFactor[0]//2 if subSamplingFactor[0]%2==1 else (subSamplingFactor[0]//2 - 1) #one closer to the beginning of the dim. Same happens when I get parts of image.
    cToCentralVoxelOfAnAveragedArea = subSamplingFactor[1]//2 if subSamplingFactor[1]%2==1 else (subSamplingFactor[1]//2 - 1)
    zToCentralVoxelOfAnAveragedArea =  subSamplingFactor[2]//2 if subSamplingFactor[2]%2==1 else (subSamplingFactor[2]//2 - 1)
    #This is where to start taking voxels from the subsampled image. From the beginning of the imagePart(1 st patch)...
    #... go forward a few steps to the voxel that is like the "central" in this subsampled (eg 3x3) area. 
    #...Then go backwards -Patchsize to find the first voxel of the subsampled. 
    rlow = image_part_slices_coords[0][0] + rToCentralVoxelOfAnAveragedArea - rSlotsPreviously#These indices can run out of image boundaries. I ll correct them afterwards.
    #If the patch is 17x17, I want a 17x17 subsampled Patch. BUT if the imgPART is 25x25 (9voxClass), I want 3 subsampledPatches in my subsampPart to cover this area!
    #That is what the last term below is taking care of.
    #CAST TO INT because ceil returns a float, and later on when computing rHighNonInclToPutTheNotPaddedInSubsampledImPart I need to do INTEGER DIVISION.
    rhighNonIncl = int(rlow + subSamplingFactor[0]*recFieldCnn[0] + (math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_rDim*1.0)/subSamplingFactor[0]) - 1) * subSamplingFactor[0]) #not including this index in the image-part
    clow = image_part_slices_coords[1][0] + cToCentralVoxelOfAnAveragedArea - cSlotsPreviously
    chighNonIncl = int(clow + subSamplingFactor[1]*recFieldCnn[1] + (math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_cDim*1.0)/subSamplingFactor[1]) - 1) * subSamplingFactor[1])
    zlow = image_part_slices_coords[2][0] + zToCentralVoxelOfAnAveragedArea - zSlotsPreviously
    zhighNonIncl = int(zlow + subSamplingFactor[2]*recFieldCnn[2] + (math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_zDim*1.0)/subSamplingFactor[2]) - 1) * subSamplingFactor[2])
        
    rlowCorrected = max(rlow, 0)
    clowCorrected = max(clow, 0)
    zlowCorrected = max(zlow, 0)
    rhighNonInclCorrected = min(rhighNonIncl, subsampledImageDimensions[0])
    chighNonInclCorrected = min(chighNonIncl, subsampledImageDimensions[1])
    zhighNonInclCorrected = min(zhighNonIncl, subsampledImageDimensions[2]) #This gave 7
    
    rLowToPutTheNotPaddedInSubsampledImPart = 0 if rlow >= 0 else abs(rlow)//subSamplingFactor[0]
    cLowToPutTheNotPaddedInSubsampledImPart = 0 if clow >= 0 else abs(clow)//subSamplingFactor[1]
    zLowToPutTheNotPaddedInSubsampledImPart = 0 if zlow >= 0 else abs(zlow)//subSamplingFactor[2]
    
    dimensionsOfTheSliceOfSubsampledImageNotPadded = [  int(math.ceil((rhighNonInclCorrected - rlowCorrected)*1.0/subSamplingFactor[0])),
                                                        int(math.ceil((chighNonInclCorrected - clowCorrected)*1.0/subSamplingFactor[1])),
                                                        int(math.ceil((zhighNonInclCorrected - zlowCorrected)*1.0/subSamplingFactor[2]))
                                                        ]
    
    #I now have exactly where to get the slice from and where to put it in the new array.
    for channel_i in xrange(len(subsampledImageChannels)) :
        intensityZeroOfChannel = calculateTheZeroIntensityOf3dImage(subsampledImageChannels[channel_i])        
        subsampledChannelsForThisImagePart[channel_i] *= intensityZeroOfChannel
        
        sliceOfSubsampledImageNotPadded = subsampledImageChannels[channel_i][   rlowCorrected : rhighNonInclCorrected : subSamplingFactor[0],
                                                                                clowCorrected : chighNonInclCorrected : subSamplingFactor[1],
                                                                                zlowCorrected : zhighNonInclCorrected : subSamplingFactor[2]
                                                                            ]
        subsampledChannelsForThisImagePart[
            channel_i,
            rLowToPutTheNotPaddedInSubsampledImPart : rLowToPutTheNotPaddedInSubsampledImPart+dimensionsOfTheSliceOfSubsampledImageNotPadded[0],
            cLowToPutTheNotPaddedInSubsampledImPart : cLowToPutTheNotPaddedInSubsampledImPart+dimensionsOfTheSliceOfSubsampledImageNotPadded[1],
            zLowToPutTheNotPaddedInSubsampledImPart : zLowToPutTheNotPaddedInSubsampledImPart+dimensionsOfTheSliceOfSubsampledImageNotPadded[2]] = sliceOfSubsampledImageNotPadded
            
    #placeholderReturn = np.ones([3,19,19,19], dtype="float32") #channel, dims 
    return subsampledChannelsForThisImagePart



def shuffleTheSegmentsForThisSubepoch(  imagePartsChannelsToLoadOnGpuForSubepochPerPathway,
                                        gtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch ) :
    numOfPathwayWithInput = len(imagePartsChannelsToLoadOnGpuForSubepochPerPathway)
    inputToZip = [ sublistForPathway for sublistForPathway in imagePartsChannelsToLoadOnGpuForSubepochPerPathway ]
    inputToZip += [ gtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch ]
    
    combined = list(zip(*inputToZip)) #list() for python3 compatibility, as range cannot get assignment in shuffle()
    random.shuffle(combined)
    shuffledInputListsToZip = list(zip(*combined))
    
    shuffledImagePartsChannelsToLoadOnGpuForSubepochPerPathway = [ sublistForPathway for sublistForPathway in shuffledInputListsToZip[:numOfPathwayWithInput] ]
    shuffledGtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch = shuffledInputListsToZip[numOfPathwayWithInput]
    
    return [shuffledImagePartsChannelsToLoadOnGpuForSubepochPerPathway, shuffledGtLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch]



# I must merge this with function: extractDataOfSegmentsUsingSampledSliceCoords() that is used for Testing! Should be easy!
# This is used in training/val only.
def extractDataOfASegmentFromImagesUsingSampledSliceCoords(
                                                        training0orValidation1,
                                                        
                                                        cnn3d,
                                                        
                                                        coordsOfCentralVoxelOfThisImPart,
                                                        numOfInpChannelsForPrimaryPath,
                                                        
                                                        allChannelsOfPatientInNpArray,
                                                        allSubsampledChannelsOfPatientInNpArray,
                                                        gtLabelsImage,
                                                        
                                                        # Intensity Augmentation
                                                        normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                        stdsOfTheChannsOfThisImage
                                                        ) :
    channelsForThisImagePartPerPathway = []
    
    howMuchToAddForEachChannel = None
    howMuchToMultiplyForEachChannel = None
    
    for pathway in cnn3d.pathways[:1] : #Hack. The rest of this loop can work for the whole .pathways...
        # ... BUT the loop does not check what happens if boundaries are out of limits, to fill with zeros. This is done in getImagePartFromSubsampledImageForTraining().
        #... Update it in a nice way to be done here, and then take getImagePartFromSubsampledImageForTraining() out and make loop go for every pathway.
        
        if pathway.pType() == pt.FC :
            continue
        subSamplingFactor = pathway.subsFactor()
        pathwayInputShapeRcz = pathway.getShapeOfInput()[0][2:] if training0orValidation1 == 0 else pathway.getShapeOfInput()[1][2:]
        leftBoundaryRcz = [ coordsOfCentralVoxelOfThisImPart[0] - subSamplingFactor[0]*(pathwayInputShapeRcz[0]-1)//2,
                            coordsOfCentralVoxelOfThisImPart[1] - subSamplingFactor[1]*(pathwayInputShapeRcz[1]-1)//2,
                            coordsOfCentralVoxelOfThisImPart[2] - subSamplingFactor[2]*(pathwayInputShapeRcz[2]-1)//2]
        rightBoundaryRcz = [leftBoundaryRcz[0] + subSamplingFactor[0]*pathwayInputShapeRcz[0],
                            leftBoundaryRcz[1] + subSamplingFactor[1]*pathwayInputShapeRcz[1],
                            leftBoundaryRcz[2] + subSamplingFactor[2]*pathwayInputShapeRcz[2]]
        channelsForThisImagePart = allChannelsOfPatientInNpArray[:,
                                                                leftBoundaryRcz[0] : rightBoundaryRcz[0] : subSamplingFactor[0],
                                                                leftBoundaryRcz[1] : rightBoundaryRcz[1] : subSamplingFactor[1],
                                                                leftBoundaryRcz[2] : rightBoundaryRcz[2] : subSamplingFactor[2]]
        
        #############################
        #Normalization Augmentation of the Patches! For more randomness.
        #Get parameters by how much to renormalize-augment mean and std.
        if training0orValidation1 == 0 and normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 2 : #[0] == 2 means augment the intensities of the segments.
            if howMuchToAddForEachChannel == None or howMuchToMultiplyForEachChannel == None :
                muOfGaussToAdd = normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[2][0]
                stdOfGaussToAdd = normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[2][1]
                if stdOfGaussToAdd != 0 : #np.random.normal does not work for an std==0.
                    howMuchToAddForEachChannel = np.random.normal(muOfGaussToAdd, stdOfGaussToAdd, [numOfInpChannelsForPrimaryPath, 1,1,1])
                else :
                    howMuchToAddForEachChannel = np.ones([numOfInpChannelsForPrimaryPath, 1,1,1], dtype="float32")*muOfGaussToAdd
                howMuchToAddForEachChannel = howMuchToAddForEachChannel * np.reshape(stdsOfTheChannsOfThisImage, [numOfInpChannelsForPrimaryPath, 1,1,1])
                
                muOfGaussToMultiply = normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[3][0]
                stdOfGaussToMultiply = normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[3][1]
                if stdOfGaussToMultiply != 0 :
                    howMuchToMultiplyForEachChannel = np.random.normal(muOfGaussToMultiply, stdOfGaussToMultiply, [numOfInpChannelsForPrimaryPath, 1,1,1])
                else :
                    howMuchToMultiplyForEachChannel = np.ones([numOfInpChannelsForPrimaryPath, 1,1,1], dtype="float32")*muOfGaussToMultiply
            channelsForThisImagePart = (channelsForThisImagePart + howMuchToAddForEachChannel)*howMuchToMultiplyForEachChannel
        ##############################
        
        channelsForThisImagePartPerPathway.append(channelsForThisImagePart)
        
    # Extract the samples for secondary pathways. This whole for can go away, if I update above code to check to slices out of limits.
    for pathway_i in xrange(len(cnn3d.pathways)) : # Except Normal 1st, cause that was done already.
        if cnn3d.pathways[pathway_i].pType() == pt.FC or cnn3d.pathways[pathway_i].pType() == pt.NORM:
            continue
        #this datastructure is similar to channelsForThisImagePart, but contains voxels from the subsampled image.
        dimsOfPrimarySegment = cnn3d.pathways[pathway_i].getShapeOfInput()[training0orValidation1][2:]
        slicesCoordsOfSegmForPrimaryPathway = [ [leftBoundaryRcz[0], rightBoundaryRcz[0]-1], [leftBoundaryRcz[1], rightBoundaryRcz[1]-1], [leftBoundaryRcz[2], rightBoundaryRcz[2]-1] ] #the right hand values are placeholders in this case.
        channsForThisSubsampledPartAndPathway = getImagePartFromSubsampledImageForTraining(dimsOfPrimarySegment=dimsOfPrimarySegment,
                                                                                        recFieldCnn=cnn3d.recFieldCnn,
                                                                                        subsampledImageChannels=allSubsampledChannelsOfPatientInNpArray,
                                                                                        image_part_slices_coords=slicesCoordsOfSegmForPrimaryPathway,
                                                                                        subSamplingFactor=cnn3d.pathways[pathway_i].subsFactor(),
                                                                                        subsampledImagePartDimensions=cnn3d.pathways[pathway_i].getShapeOfInput()[training0orValidation1][2:]
                                                                                        )
        #############################
        #Normalization-Augmentation of the Patches! For more randomness.
        #Get parameters by how much to renormalize-augment mean and std.
        if training0orValidation1 == 0 and normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 2:
            #Use values  computed on the normal-resolution images. BUT PREREQUISITE IS TO HAVE THE SAME CHANNELS IN THE TWO PATHWAYS. Else need to recompute!
            channsForThisSubsampledPartAndPathway = (channsForThisSubsampledPartAndPathway + howMuchToAddForEachChannel)*howMuchToMultiplyForEachChannel
        elif normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 2 : 
            #Need to recompute. NOT IMPLEMENTED YET.
            myLogger.print3("ERROR: The system uses different channels for normal and subsampled pathway. And was asked to use Data Augmentation with intensity-noise. Not implemented yet. Exiting.")
            exit(1)
        ##############################
        channelsForThisImagePartPerPathway.append(channsForThisSubsampledPartAndPathway)
        
    # Get ground truth labels for training.
    numOfCentralVoxelsClassifRcz = cnn3d.finalTargetLayer_outputShapeTrainValTest[training0orValidation1][2:]
    leftBoundaryRcz = [ coordsOfCentralVoxelOfThisImPart[0] - (numOfCentralVoxelsClassifRcz[0]-1)//2,
                        coordsOfCentralVoxelOfThisImPart[1] - (numOfCentralVoxelsClassifRcz[1]-1)//2,
                        coordsOfCentralVoxelOfThisImPart[2] - (numOfCentralVoxelsClassifRcz[2]-1)//2]
    rightBoundaryRcz = [leftBoundaryRcz[0] + numOfCentralVoxelsClassifRcz[0],
                        leftBoundaryRcz[1] + numOfCentralVoxelsClassifRcz[1],
                        leftBoundaryRcz[2] + numOfCentralVoxelsClassifRcz[2]]
    gtLabelsForTheCentralClassifiedPartOfThisImagePart = gtLabelsImage[ leftBoundaryRcz[0] : rightBoundaryRcz[0],
                                                                        leftBoundaryRcz[1] : rightBoundaryRcz[1],
                                                                        leftBoundaryRcz[2] : rightBoundaryRcz[2] ]
    
    return [ channelsForThisImagePartPerPathway, gtLabelsForTheCentralClassifiedPartOfThisImagePart ]




#################################################################################################################################
#                                                                                                                               #
#       Below are functions for testing only. There is duplication with training. They are not the same, but could be merged.   #
#                                                                                                                               #
#################################################################################################################################

# This is very similar to sampleImageParts() I believe, which is used for training. Consider way to merge them.
def getCoordsOfAllSegmentsOfAnImage(myLogger,
                                    dimsOfPrimarySegment, # RCZ dims of input to primary pathway (NORMAL). Which should be the first one in .pathways.
                                    strideOfSegmentsPerDimInVoxels,
                                    batch_size,
                                    channelsOfImageNpArray,#chans,niiDims
                                    roiMask
                                    ) :
    myLogger.print3("Starting to (tile) extract Segments from the images of the subject for Segmentation...")
    
    sliceCoordsOfSegmentsToReturn = []
    
    niiDimensions = list(channelsOfImageNpArray[0].shape) # Dims of the volumes
    
    zLowBoundaryNext=0; zAxisCentralPartPredicted = False;
    while not zAxisCentralPartPredicted :
        zFarBoundary = min(zLowBoundaryNext+dimsOfPrimarySegment[2], niiDimensions[2]) #Excluding.
        zLowBoundary = zFarBoundary - dimsOfPrimarySegment[2]
        zLowBoundaryNext = zLowBoundaryNext + strideOfSegmentsPerDimInVoxels[2]
        zAxisCentralPartPredicted = False if zFarBoundary < niiDimensions[2] else True #THIS IS THE IMPORTANT CRITERION.
        
        cLowBoundaryNext=0; cAxisCentralPartPredicted = False;
        while not cAxisCentralPartPredicted :
            cFarBoundary = min(cLowBoundaryNext+dimsOfPrimarySegment[1], niiDimensions[1]) #Excluding.
            cLowBoundary = cFarBoundary - dimsOfPrimarySegment[1]
            cLowBoundaryNext = cLowBoundaryNext + strideOfSegmentsPerDimInVoxels[1]
            cAxisCentralPartPredicted = False if cFarBoundary < niiDimensions[1] else True
            
            rLowBoundaryNext=0; rAxisCentralPartPredicted = False;
            while not rAxisCentralPartPredicted :
                rFarBoundary = min(rLowBoundaryNext+dimsOfPrimarySegment[0], niiDimensions[0]) #Excluding.
                rLowBoundary = rFarBoundary - dimsOfPrimarySegment[0]
                rLowBoundaryNext = rLowBoundaryNext + strideOfSegmentsPerDimInVoxels[0]
                rAxisCentralPartPredicted = False if rFarBoundary < niiDimensions[0] else True
                
                if isinstance(roiMask, (np.ndarray)) : #In case I pass a brain-mask, I ll use it to only predict inside it. Otherwise, whole image.
                    if not np.any(roiMask[rLowBoundary:rFarBoundary,
                                            cLowBoundary:cFarBoundary,
                                            zLowBoundary:zFarBoundary
                                            ]) : #all of it is out of the brain so skip it.
                        continue
                    
                sliceCoordsOfSegmentsToReturn.append([ [rLowBoundary, rFarBoundary-1], [cLowBoundary, cFarBoundary-1], [zLowBoundary, zFarBoundary-1] ])
                
    #I need to have a total number of image-parts that can be exactly-divided by the 'batch_size'. For this reason, I add in the far end of the list multiple copies of the last element. I NEED THIS IN THEANO. I TRIED WITHOUT. NO.
    total_number_of_image_parts = len(sliceCoordsOfSegmentsToReturn)
    number_of_imageParts_missing_for_exact_division =  batch_size - total_number_of_image_parts%batch_size if total_number_of_image_parts%batch_size != 0 else 0
    for extra_useless_image_part_i in xrange(number_of_imageParts_missing_for_exact_division) :
        sliceCoordsOfSegmentsToReturn.append(sliceCoordsOfSegmentsToReturn[-1])
        
    #I think that since the parts are acquired in a certain order and are sorted this way in the list, it is easy
    #to know which part of the image they came from, as it depends only on the stride-size and the imagePart size.
    
    myLogger.print3("Finished (tiling) extracting Segments from the images of the subject for Segmentation.")
    
    # sliceCoordsOfSegmentsToReturn: list with 3 dimensions. numberOfSegments x 3(rcz) x 2 (lower and upper limit of the segment, INCLUSIVE both sides)
    return [sliceCoordsOfSegmentsToReturn]



# I must merge this with function: extractDataOfASegmentFromImagesUsingSampledSliceCoords() that is used for Training/Validation! Should be easy!
# This is used in testing only.
def extractDataOfSegmentsUsingSampledSliceCoords(cnn3dInstance,
                                                sliceCoordsOfSegmentsToExtract,
                                                channelsOfImageNpArray,#chans,niiDims
                                                channelsOfSubsampledImageNpArray, #chans,niiDims
                                                recFieldCnn
                                                ) :
    numberOfSegmentsToExtract = len(sliceCoordsOfSegmentsToExtract)
    channsForSegmentsPerPathToReturn = [ [] for i in xrange(cnn3dInstance.getNumPathwaysThatRequireInput()) ] # [pathway, image parts, channels, r, c, z]
    dimsOfPrimarySegment = cnn3dInstance.pathways[0].getShapeOfInput()[2][2:] # RCZ dims of input to primary pathway (NORMAL). Which should be the first one in .pathways.
    
    for segment_i in xrange(numberOfSegmentsToExtract) :
        rLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][0][0]; rFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][0][1]
        cLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][1][0]; cFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][1][1]
        zLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][2][0]; zFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][2][1]
        # segment for primary pathway
        channsForPrimaryPathForThisSegm = channelsOfImageNpArray[:,
                                                                rLowBoundary:rFarBoundary+1,
                                                                cLowBoundary:cFarBoundary+1,
                                                                zLowBoundary:zFarBoundary+1
                                                                ]
        channsForSegmentsPerPathToReturn[0].append(channsForPrimaryPathForThisSegm)
        
        #Subsampled pathways
        for pathway_i in xrange(len(cnn3dInstance.pathways)) : # Except Normal 1st, cause that was done already.
            if cnn3dInstance.pathways[pathway_i].pType() == pt.FC or cnn3dInstance.pathways[pathway_i].pType() == pt.NORM:
                continue
            slicesCoordsOfSegmForPrimaryPathway = [ [rLowBoundary, rFarBoundary-1], [cLowBoundary, cFarBoundary-1], [zLowBoundary, zFarBoundary-1] ] #the right hand values are placeholders in this case.
            channsForThisSubsPathForThisSegm = getImagePartFromSubsampledImageForTraining(  dimsOfPrimarySegment=dimsOfPrimarySegment,
                                                                                            recFieldCnn=recFieldCnn,
                                                                                            subsampledImageChannels=channelsOfSubsampledImageNpArray,
                                                                                            image_part_slices_coords=slicesCoordsOfSegmForPrimaryPathway,
                                                                                            subSamplingFactor=cnn3dInstance.pathways[pathway_i].subsFactor(),
                                                                                            subsampledImagePartDimensions=cnn3dInstance.pathways[pathway_i].getShapeOfInput()[2][2:]
                                                                                            )
            channsForSegmentsPerPathToReturn[pathway_i].append(channsForThisSubsPathForThisSegm)
            
    return [channsForSegmentsPerPathToReturn]





