# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import sys
import time
import numpy as np
import nibabel as nib
import random
import math

from scipy.ndimage.filters import gaussian_filter

import pp

from deepmedic.cnnHelpers import dump_cnn_to_gzip_file_dotSave
from deepmedic.genericHelpers import *

from deepmedic.accuracyMonitor import AccuracyOfEpochMonitorSegmentation

TINY_FLOAT = np.finfo(np.float32).tiny 

#These two pad/unpad should have their own class, and an instance should be created per subject. So that unpad gets how much to unpad from the pad.
def padCnnInputs(array1, cnnReceptiveField, imagePartDimensions) : #Works for 2D as well I think.
    cnnReceptiveFieldArray = np.asarray(cnnReceptiveField, dtype="int16")
    array1Dimensions = np.asarray(array1.shape,dtype="int16")
    if len(array1.shape) <> 3 :
        print("ERROR! Given array in padCnnInputs() was expected of 3-dimensions, but was passed an array of dimensions: ", array1.shape,", Exiting!")
        exit(1)
    #paddingValue = (array1[0,0,0] + array1[-1,0,0] + array1[0,-1,0] + array1[-1,-1,0] + array1[0,0,-1] + array1[-1,0,-1] + array1[0,-1,-1] + array1[-1,-1,-1]) / 8.0
    #Calculate how much padding needed to fully infer the original array1, taking only the receptive field in account.
    paddingAtLeftPerAxis = (cnnReceptiveFieldArray - 1) / 2
    paddingAtRightPerAxis = cnnReceptiveFieldArray - 1 - paddingAtLeftPerAxis
    #Now, to cover the case that the specified image-segment of the CNN is larger than the image (eg full-image inference and current image is smaller), pad further to right.
    paddingFurtherToTheRightNeededForSegment = np.maximum(0, np.asarray(imagePartDimensions,dtype="int16")-(array1Dimensions+paddingAtLeftPerAxis+paddingAtRightPerAxis))
    paddingAtRightPerAxis += paddingFurtherToTheRightNeededForSegment
    
    tupleOfPaddingPerAxes = ( (paddingAtLeftPerAxis[0],paddingAtRightPerAxis[0]), (paddingAtLeftPerAxis[1],paddingAtRightPerAxis[1]), (paddingAtLeftPerAxis[2],paddingAtRightPerAxis[2]))
    #Very poor design because channels/gt/bmask etc are all getting back a different padding? tupleOfPaddingPerAxes is returned in order for unpad to know.
    return [np.lib.pad(array1, tupleOfPaddingPerAxes, 'reflect' ), tupleOfPaddingPerAxes]

#In the 3 first axes. Which means it can take a 4-dim image.
def unpadCnnOutputs(array1, tupleOfPaddingPerAxesLeftRight) :
    #tupleOfPaddingPerAxesLeftRight : ( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)).
    unpaddedArray1 = array1[tupleOfPaddingPerAxesLeftRight[0][0]:, tupleOfPaddingPerAxesLeftRight[1][0]:, tupleOfPaddingPerAxesLeftRight[2][0]:]
    #The checks below are to make it work if padding == 0, which may happen for 2D on the 3rd axis.
    unpaddedArray1 = unpaddedArray1[:-tupleOfPaddingPerAxesLeftRight[0][1],:,:] if tupleOfPaddingPerAxesLeftRight[0][1] > 0 else unpaddedArray1 
    unpaddedArray1 = unpaddedArray1[:,:-tupleOfPaddingPerAxesLeftRight[1][1],:] if tupleOfPaddingPerAxesLeftRight[1][1] > 0 else unpaddedArray1
    unpaddedArray1 = unpaddedArray1[:,:,:-tupleOfPaddingPerAxesLeftRight[2][1]] if tupleOfPaddingPerAxesLeftRight[2][1] > 0 else unpaddedArray1
    return unpaddedArray1

def reflectImageArrayIfNeeded(reflectFlags, imageArray) :
    stepsForReflectionPerDimension = [-1 if reflectFlags[0] else 1, -1 if reflectFlags[1] else 1, -1 if reflectFlags[2] else 1]
    
    reflImageArray = imageArray[::stepsForReflectionPerDimension[0], ::stepsForReflectionPerDimension[1], ::stepsForReflectionPerDimension[2]]
    return reflImageArray

def smoothImageWithGaussianFilterIfNeeded(smoothImageWithGaussianFilterStds, imageArray) :
    # Either None for no smoothing, or a List with 3 elements, [std-r, std-c, std-z] of the gaussian kernel to smooth with.
    #If I do not want to smooth at all a certain axis, pass std=0 for it. Works, I tried. Returns the actual voxel value.
    if smoothImageWithGaussianFilterStds == None :
        return imageArray
    else :
        return gaussian_filter(imageArray, smoothImageWithGaussianFilterStds)
    
# roi_mask_filename and roiMinusLesion_mask_filename can be passed "no". In this case, the corresponding return result is nothing.
# This is so because: the do_training() function only needs the roiMinusLesion_mask, whereas the do_testing() only needs the roi_mask.        
def actual_load_patient_images_from_filepath_and_return_nparrays(myLogger,
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
                                                                usingSubsampledWaypath,
                                                                listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                                                
                                                                padInputImagesBool,
                                                                cnnReceptiveField, # only used if padInputImagesBool
                                                                imagePartDimensions,
                                                                
                                                                smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                                normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                                reflectImageWithHalfProb
                                                                ):
    #listOfNiiFilepathNames: should be a list of lists. Each sublist corresponds to one certain patient-case.
    #...Each sublist should have as many elements(strings-filenamePaths) as numberOfChannels, point to the channels of this patient.
    
    if index_of_wanted_image >= len(listOfFilepathsToEachChannelOfEachPatient) :
        myLogger.print3("ERROR : Function 'ACTUAL_load_patient_images_from_filepath_and_return_nparrays'-")
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
        img_proxy = nib.load(fullFilenamePathOfRoiMask)
        roiMaskData = img_proxy.get_data()
        roiMaskData = reflectImageArrayIfNeeded(reflectFlags, roiMaskData)
        roiMask = roiMaskData #np.asarray(roiMaskData, dtype="float32") #the .get_data returns a nparray but it is not a float64
        img_proxy.uncache()
        [roiMask, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(roiMask, cnnReceptiveField, imagePartDimensions) if padInputImagesBool else [roiMask, tupleOfPaddingPerAxesLeftRight]
    else :
        roiMask = "placeholderNothing"
        
    #Load the channels of the patient.
    niiDimensions = None
    allChannelsOfPatientInNpArray = None
    #The below has dimensions (channels, 2). Holds per channel: [value to add per voxel for mean norm, value to multiply for std renorm]
    howMuchToAddAndMultiplyForNormalizationAugmentationForEachChannel = np.ones( (numberOfNormalScaleChannels, 2), dtype="float32")
    for channel_i in xrange(numberOfNormalScaleChannels):
        fullFilenamePathOfChannel = listOfFilepathsToEachChannelOfEachPatient[index_of_wanted_image][channel_i]
        img_proxy = nib.load(fullFilenamePathOfChannel)
        channelData = img_proxy.get_data()
        if len(channelData.shape) > 3 :
            channelData = channelData[:,:,:,0]
            
        channelData = smoothImageWithGaussianFilterIfNeeded(smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage[0], channelData)
        channelData = reflectImageArrayIfNeeded(reflectFlags, channelData) #reflect if flag ==1 .
        [channelData, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(channelData, cnnReceptiveField, imagePartDimensions) if padInputImagesBool else [channelData, tupleOfPaddingPerAxesLeftRight]
        
        if not isinstance(allChannelsOfPatientInNpArray, (np.ndarray)) :
            #Initialize the array in which all the channels for the patient will be placed.
            niiDimensions = list(channelData.shape)
            allChannelsOfPatientInNpArray = np.zeros( (numberOfNormalScaleChannels, niiDimensions[0], niiDimensions[1], niiDimensions[2]))
            
        allChannelsOfPatientInNpArray[channel_i] = channelData
        """
        if len(channelData.shape) <= 3 :
            allChannelsOfPatientInNpArray[channel_i] = channelData #np.asarray(channelData, dtype="float32")
        else : #In many cases the image is of 4 dimensions, with last being 'time'
            allChannelsOfPatientInNpArray[channel_i] = channelData[:,:,:,0] #np.asarray(channelData[:,:,:,0], dtype="float32") #[:,:,:,0] because the nii image actually is of 4 dims, with 4th being time.
        """
        img_proxy.uncache()
        
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
            
    #LOAD the class-labels.
    if providedGtLabelsBool : #For training (exact target labels) or validation on samples labels.
        fullFilenamePathOfGtLabels = listOfFilepathsToGtLabelsOfEachPatient[index_of_wanted_image]
        imgGtLabels_proxy = nib.load(fullFilenamePathOfGtLabels)
        #If the gt file was not type "int" (eg it was float), convert it to int. Because later I m doing some == int comparisons.
        gtLabelsData = imgGtLabels_proxy.get_data()
        gtLabelsData = gtLabelsData if np.issubdtype( gtLabelsData.dtype, np.int ) else np.rint(gtLabelsData).astype("int32")
        gtLabelsData = reflectImageArrayIfNeeded(reflectFlags, gtLabelsData) #reflect if flag ==1 .
        imageGtLabels = gtLabelsData
        imgGtLabels_proxy.uncache()
        [imageGtLabels, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(imageGtLabels, cnnReceptiveField, imagePartDimensions) if padInputImagesBool else [imageGtLabels, tupleOfPaddingPerAxesLeftRight]
    else : 
        imageGtLabels = "placeholderNothing" #For validation and testing
        
    if training0orValidation1orTest2 <> 2 and providedWeightMapsToSampleForEachCategory==True : # in testing these weightedMaps are never provided, they are for training/validation only.
        numberOfSamplingCategories = len(forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient)
        arrayWithWeightMapsWhereToSampleForEachCategory = np.zeros( [numberOfSamplingCategories] + list(allChannelsOfPatientInNpArray[0].shape), dtype="float32" ) 
        for cat_i in xrange( numberOfSamplingCategories ) :
            filepathsToTheWeightMapsOfAllPatientsForThisCategory = forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient[cat_i]
            filepathToTheWeightMapOfThisPatientForThisCategory = filepathsToTheWeightMapsOfAllPatientsForThisCategory[index_of_wanted_image]
            
            img_proxy = nib.load(filepathToTheWeightMapOfThisPatientForThisCategory)
            weightedMapForThisCatData = img_proxy.get_data()
            weightedMapForThisCatData = reflectImageArrayIfNeeded(reflectFlags, weightedMapForThisCatData)
            img_proxy.uncache()
            [weightedMapForThisCatData, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(weightedMapForThisCatData, cnnReceptiveField, imagePartDimensions) if padInputImagesBool else [weightedMapForThisCatData, tupleOfPaddingPerAxesLeftRight]
            
            arrayWithWeightMapsWhereToSampleForEachCategory[cat_i] = weightedMapForThisCatData
    else :
        arrayWithWeightMapsWhereToSampleForEachCategory = "placeholderNothing"
        
    # The second CNN pathway...
    if not usingSubsampledWaypath :
        allSubsampledChannelsOfPatientInNpArray = "placeholderNothing"
    elif useSameSubChannelsAsSingleScale : #Pass this in the configuration file, instead of a list of channel names, to use the same channels as the normal res.
        allSubsampledChannelsOfPatientInNpArray = allChannelsOfPatientInNpArray #np.asarray(allChannelsOfPatientInNpArray, dtype="float32") #Hope this works, to win time in loading. Without copying it did not work.
    else :
        numberOfSubsampledScaleChannels = len(listOfFilepathsToEachSubsampledChannelOfEachPatient[0])
        allSubsampledChannelsOfPatientInNpArray = np.zeros( (numberOfSubsampledScaleChannels, niiDimensions[0], niiDimensions[1], niiDimensions[2]))
        for channel_i in xrange(numberOfSubsampledScaleChannels):
            fullFilenamePathOfChannel = listOfFilepathsToEachSubsampledChannelOfEachPatient[index_of_wanted_image][channel_i]
            img_proxy = nib.load(fullFilenamePathOfChannel)
            channelData = img_proxy.get_data()
            if len(channelData.shape) > 3 :
                channelData = channelData[:,:,:,0]
            channelData = smoothImageWithGaussianFilterIfNeeded(smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage[1], channelData)
            channelData = reflectImageArrayIfNeeded(reflectFlags, channelData)
            [channelData, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(channelData, cnnReceptiveField, imagePartDimensions) if padInputImagesBool else [channelData, tupleOfPaddingPerAxesLeftRight]
            
            allSubsampledChannelsOfPatientInNpArray[channel_i] = channelData
            """
            if len(channelData.shape) <= 3 :
                allSubsampledChannelsOfPatientInNpArray[channel_i] = channelData #np.asarray(channelData, dtype="float32")
            else : #In many cases the image is of 4 dimensions, with last being 'time'
                allSubsampledChannelsOfPatientInNpArray[channel_i] = channelData[:,:,:,0] #np.asarray(channelData[:,:,:,0], dtype="float32") #[:,:,:,0] because the nii image actually is of 4 dims, with 4th being time.
            """
            img_proxy.uncache()
            
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
                        imagePartDimensions,
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
    
    halfImagePartBoundaries = np.zeros( (len(imagePartDimensions), 2) , dtype='int32') #dim1: 1 row per r,c,z. Dim2: left/right width not to sample from (=half segment).
    
    #The below starts all zero. Will be Multiplied by other true-false arrays expressing if the relevant voxels are within boundaries.
    #In the end, the final vector will be true only for the indices of lesions that are within all boundaries.
    booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries = np.zeros(weightMapToSampleFrom.shape, dtype="int32")
    
    #The following loop leads to booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries to be true for the indices that allow you to get an image part CENTERED on them, and be safely within image boundaries. Note that if the imagePart is of even dimension, the "central" voxel is one voxel to the left.
    for rcz_i in xrange( len(imagePartDimensions) ) :
        if imagePartDimensions[rcz_i]%2 == 0: #even
            dimensionDividedByTwo = imagePartDimensions[rcz_i]/2
            halfImagePartBoundaries[rcz_i] = [dimensionDividedByTwo - 1, dimensionDividedByTwo] #central of ImagePart is 1 vox closer to begining of axes.
        else: #odd
            dimensionDividedByTwoFloor = math.floor(imagePartDimensions[rcz_i]/2) #eg 5/2 = 2, with the 3rd voxel being the "central"
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
    indicesInTheFlattenArrayThatWereSampledAsCentralVoxelsOfImageParts = np.random.choice(constrainedWithImageBoundariesMaskToSample.size,
                                                                                            size = numOfSegmentsToExtractForThisSubject,
                                                                                            replace=True,
                                                                                            p=flattenedConstrainedWithImageBoundariesMaskToSample)
    #np.unravel_index([listOfIndicesInFlattened], dims) returns a tuple of arrays (eg 3 of them if 3 dimImage), where each of the array in the tuple has the same shape as the listOfIndices. They have the r/c/z coords that correspond to the index of the flattened version.
    #So, coordsOfCentralVoxelsOfPartsSampled will end up being an array with shape: 3(rcz) x numOfSegmentsToExtractForThisSubject.
    coordsOfCentralVoxelsOfPartsSampled = np.asarray(np.unravel_index(indicesInTheFlattenArrayThatWereSampledAsCentralVoxelsOfImageParts,
                                                                    constrainedWithImageBoundariesMaskToSample.shape #the shape of the brainmask/scan.
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
    # Currently, coordsOfCentralVoxelsOfPartsSampled is useless. I could remove it entirely.
    return imagePartsSampled


def getImagePartFromSubsampledImageForTraining( imagePartDimensions,
                                                patchDimensions,
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
    
    numberOfCentralVoxelsClassifiedForEachImagePart_rDim = imagePartDimensions[0] - patchDimensions[0] + 1
    numberOfCentralVoxelsClassifiedForEachImagePart_cDim = imagePartDimensions[1] - patchDimensions[1] + 1
    numberOfCentralVoxelsClassifiedForEachImagePart_zDim = imagePartDimensions[2] - patchDimensions[2] + 1
    
    #Calculate the slice that I should get, and where I should put it in the imagePart (eg if near the borders, and I cant grab a whole slice-imagePart).
    rSlotsPreviously = ((subSamplingFactor[0]-1)/2)*patchDimensions[0] if subSamplingFactor[0]%2==1 \
                                                else (subSamplingFactor[0]-2)/2*patchDimensions[0] + patchDimensions[0]/2
    cSlotsPreviously = ((subSamplingFactor[1]-1)/2)*patchDimensions[1] if subSamplingFactor[1]%2==1 \
                                                else (subSamplingFactor[1]-2)/2*patchDimensions[1] + patchDimensions[1]/2
    zSlotsPreviously = ((subSamplingFactor[2]-1)/2)*patchDimensions[2] if subSamplingFactor[2]%2==1 \
                                                else (subSamplingFactor[2]-2)/2*patchDimensions[2] + patchDimensions[2]/2
    #1*17
    rToCentralVoxelOfAnAveragedArea = subSamplingFactor[0]/2 if subSamplingFactor[0]%2==1 else (subSamplingFactor[0]/2 - 1) #one closer to the beginning of the dim. Same happens when I get parts of image.
    cToCentralVoxelOfAnAveragedArea = subSamplingFactor[1]/2 if subSamplingFactor[1]%2==1 else (subSamplingFactor[1]/2 - 1)
    zToCentralVoxelOfAnAveragedArea =  subSamplingFactor[2]/2 if subSamplingFactor[2]%2==1 else (subSamplingFactor[2]/2 - 1)
    #This is where to start taking voxels from the subsampled image. From the beginning of the imagePart(1 st patch)...
    #... go forward a few steps to the voxel that is like the "central" in this subsampled (eg 3x3) area. 
    #...Then go backwards -Patchsize to find the first voxel of the subsampled. 
    rlow = image_part_slices_coords[0][0] + rToCentralVoxelOfAnAveragedArea - rSlotsPreviously#These indices can run out of image boundaries. I ll correct them afterwards.
    #If the patch is 17x17, I want a 17x17 subsampled Patch. BUT if the imgPART is 25x25 (9voxClass), I want 3 subsampledPatches in my subsampPart to cover this area!
    #That is what the last term below is taking care of.
    #CAST TO INT because ceil returns a float, and later on when computing rHighNonInclToPutTheNotPaddedInSubsampledImPart I need to do INTEGER DIVISION.
    rhighNonIncl = int(rlow + subSamplingFactor[0]*patchDimensions[0] + (math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_rDim*1.0)/subSamplingFactor[0]) - 1) * subSamplingFactor[0]) #not including this index in the image-part
    clow = image_part_slices_coords[1][0] + cToCentralVoxelOfAnAveragedArea - cSlotsPreviously
    chighNonIncl = int(clow + subSamplingFactor[1]*patchDimensions[1] + (math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_cDim*1.0)/subSamplingFactor[1]) - 1) * subSamplingFactor[1])
    zlow = image_part_slices_coords[2][0] + zToCentralVoxelOfAnAveragedArea - zSlotsPreviously
    zhighNonIncl = int(zlow + subSamplingFactor[2]*patchDimensions[2] + (math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_zDim*1.0)/subSamplingFactor[2]) - 1) * subSamplingFactor[2])
    
    rlowCorrected = max(rlow, 0)
    clowCorrected = max(clow, 0)
    zlowCorrected = max(zlow, 0)
    rhighNonInclCorrected = min(rhighNonIncl, subsampledImageDimensions[0])
    chighNonInclCorrected = min(chighNonIncl, subsampledImageDimensions[1])
    zhighNonInclCorrected = min(zhighNonIncl, subsampledImageDimensions[2]) #This gave 7
    
    rLowToPutTheNotPaddedInSubsampledImPart = 0 if rlow >= 0 else abs(rlow)/subSamplingFactor[0]
    cLowToPutTheNotPaddedInSubsampledImPart = 0 if clow >= 0 else abs(clow)/subSamplingFactor[1]
    zLowToPutTheNotPaddedInSubsampledImPart = 0 if zlow >= 0 else abs(zlow)/subSamplingFactor[2]
    
    #print "DEBUG: rlow=",rlow, " rhighNonIncl=",rhighNonIncl," rlowCorrected=",rlowCorrected," rhighNonInclCorrected=",rhighNonInclCorrected," rLowToPutTheNotPaddedInSubsampledImPart=", rLowToPutTheNotPaddedInSubsampledImPart, " rHighNonInclToPutTheNotPaddedInSubsampledImPart=",rHighNonInclToPutTheNotPaddedInSubsampledImPart
    
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


def getCoordsOfAllSegmentsOfAnImage(myLogger,
                                    strideOfImagePartsPerDimensionInVoxels,
                                    imagePartDimensions,
                                    batch_size,
                                    channelsOfImageNpArray,#chans,niiDims
                                    brainMask
                                    ) :
    myLogger.print3("Starting to (tile) extract Segments from the images of the subject for Segmentation...")
    
    sliceCoordsOfSegmentsToReturn = []
    
    niiDimensions = list(channelsOfImageNpArray[0].shape)
    
    zLowBoundaryNext=0; zAxisCentralPartPredicted = False;
    while not zAxisCentralPartPredicted :
        zFarBoundary = min(zLowBoundaryNext+imagePartDimensions[2], niiDimensions[2]) #Excluding.
        zLowBoundary = zFarBoundary - imagePartDimensions[2]
        zLowBoundaryNext = zLowBoundaryNext + strideOfImagePartsPerDimensionInVoxels[2]
        zAxisCentralPartPredicted = False if zFarBoundary < niiDimensions[2] else True #THIS IS THE IMPORTANT CRITERIO.
        
        cLowBoundaryNext=0; cAxisCentralPartPredicted = False;
        while not cAxisCentralPartPredicted :
            cFarBoundary = min(cLowBoundaryNext+imagePartDimensions[1], niiDimensions[1]) #Excluding.
            cLowBoundary = cFarBoundary - imagePartDimensions[1]
            cLowBoundaryNext = cLowBoundaryNext + strideOfImagePartsPerDimensionInVoxels[1]
            cAxisCentralPartPredicted = False if cFarBoundary < niiDimensions[1] else True
            
            rLowBoundaryNext=0; rAxisCentralPartPredicted = False;
            while not rAxisCentralPartPredicted :
                rFarBoundary = min(rLowBoundaryNext+imagePartDimensions[0], niiDimensions[0]) #Excluding.
                rLowBoundary = rFarBoundary - imagePartDimensions[0]
                rLowBoundaryNext = rLowBoundaryNext + strideOfImagePartsPerDimensionInVoxels[0]
                rAxisCentralPartPredicted = False if rFarBoundary < niiDimensions[0] else True
                
                if isinstance(brainMask, (np.ndarray)) : #In case I pass a brain-mask, I ll use it to only predict inside it. Otherwise, whole image.
                    if not np.any(brainMask[rLowBoundary:rFarBoundary,
                                            cLowBoundary:cFarBoundary,
                                            zLowBoundary:zFarBoundary
                                            ]) : #all of it is out of the brain so skip it.
                        continue
                    
                sliceCoordsOfSegmentsToReturn.append([ [rLowBoundary, rFarBoundary-1], [cLowBoundary, cFarBoundary-1], [zLowBoundary, zFarBoundary-1] ])
                
    #I need to have a total number of image-parts that can be exactly-divided by the 'batch_size'. For this reason, I add in the far end of the list multiple copies of the last element. I NEED THIS IN THEANO. I TRIED WITHOUT. NO.
    total_number_of_image_parts = len(sliceCoordsOfSegmentsToReturn)
    number_of_imageParts_missing_for_exact_division =  batch_size - total_number_of_image_parts%batch_size if total_number_of_image_parts%batch_size <> 0 else 0
    for extra_useless_image_part_i in xrange(number_of_imageParts_missing_for_exact_division) :
        sliceCoordsOfSegmentsToReturn.append(sliceCoordsOfSegmentsToReturn[total_number_of_image_parts-1])
        
    #I think that since the parts are acquired in a certain order and are sorted this way in the list, it is easy
    #to know which part of the image they came from, as it depends only on the stride-size and the imagePart size.
    
    myLogger.print3("Finished (tiling) extracting Segments from the images of the subject for Segmentation.")
    
    # sliceCoordsOfSegmentsToReturn: list with 3 dimensions. numberOfSegments x 3(rcz) x 2 (lower and upper limit of the segment, INCLUSIVE both sides)
    return [sliceCoordsOfSegmentsToReturn]


def extractDataOfASegmentFromImagesUsingSampledSliceCoords(
                                                        training0orValidation1,
                                                        
                                                        sliceCoordsOfThisImagePart,
                                                        numberOfNormalScaleChannels,
                                                        imagePartDimensions,
                                                        patchDimensions,
                                                        
                                                        allChannelsOfPatientInNpArray,
                                                        gtLabelsImage,
                                                        
                                                        usingSubsampledWaypath,
                                                        useSameSubChannelsAsSingleScale,
                                                        allSubsampledChannelsOfPatientInNpArray,
                                                        subSamplingFactor,
                                                        subsampledImagePartDimensions,
                                                        # Intensity Augmentation
                                                        normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                        stdsOfTheChannsOfThisImage
                                                        ) :
    
    channelsForThisImagePart = np.zeros((numberOfNormalScaleChannels, imagePartDimensions[0],imagePartDimensions[1],imagePartDimensions[2]), dtype = 'float32')
    #Inclusive both sides. That's the slice I should grab to get the imagePart, centrered around [0].
    
    channelsForThisImagePart[:numberOfNormalScaleChannels] = allChannelsOfPatientInNpArray[ :,
                                                                                            sliceCoordsOfThisImagePart[0][0]:sliceCoordsOfThisImagePart[0][1] +1,
                                                                                            sliceCoordsOfThisImagePart[1][0]:sliceCoordsOfThisImagePart[1][1] +1,
                                                                                            sliceCoordsOfThisImagePart[2][0]:sliceCoordsOfThisImagePart[2][1] +1]
    #############################
    #Normalization Augmentation of the Patches! For more randomness.
    #Get parameters by how much to renormalize-augment mean and std.
    if training0orValidation1 == 0 and normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 2 : #[0] == 2 means augment the intensities of the segments.
        muOfGaussToAdd = normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[2][0]
        stdOfGaussToAdd = normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[2][1]
        if stdOfGaussToAdd <> 0 : #np.random.normal does not work for an std==0.
            howMuchToAddForEachChannel = np.random.normal(muOfGaussToAdd, stdOfGaussToAdd, [numberOfNormalScaleChannels, 1,1,1])
        else :
            howMuchToAddForEachChannel = np.ones([numberOfNormalScaleChannels, 1,1,1], dtype="float32")*muOfGaussToAdd
        howMuchToAddForEachChannel = howMuchToAddForEachChannel * np.reshape(stdsOfTheChannsOfThisImage, [numberOfNormalScaleChannels, 1,1,1])

        muOfGaussToMultiply = normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[3][0]
        stdOfGaussToMultiply = normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[3][1]
        if stdOfGaussToMultiply <> 0 :
            howMuchToMultiplyForEachChannel = np.random.normal(muOfGaussToMultiply, stdOfGaussToMultiply, [numberOfNormalScaleChannels, 1,1,1])
        else :
            howMuchToMultiplyForEachChannel = np.ones([numberOfNormalScaleChannels, 1,1,1], dtype="float32")*muOfGaussToMultiply
        channelsForThisImagePart[:numberOfNormalScaleChannels] = (channelsForThisImagePart[:numberOfNormalScaleChannels] + howMuchToAddForEachChannel)*howMuchToMultiplyForEachChannel
    ##############################
    
    lesionLabelsForThisImagePart = gtLabelsImage[sliceCoordsOfThisImagePart[0][0]:sliceCoordsOfThisImagePart[0][1] +1,
                                                sliceCoordsOfThisImagePart[1][0]:sliceCoordsOfThisImagePart[1][1] +1,
                                                sliceCoordsOfThisImagePart[2][0]:sliceCoordsOfThisImagePart[2][1] +1]
    
    #Get the part of the GT-segments that correspond to the central (predicted) part of the segments:
    #I think both Patch and Part can be either even or odd, and this should work in both cases.
    rczPatchHalfWidth = [ (patchDimensions[i]-1)/2 for i in xrange(3) ] # eg rPatchHalfWidth = (patchDimensions[0]-1)/2
    rczUpperBoundOfCentralVoxelsLabels = [ imagePartDimensions[i] - rczPatchHalfWidth[i] for i in xrange(3) ] # Excluding
    #This used to be [rPatchHalfWidth : -rPatchHalfWidth], but in 2D case, where rPatchHalfWidth might be ==0, causes problem and you get a null slice.
    lesionLabelsForTheCentralClassifiedPartOfThisImagePart = lesionLabelsForThisImagePart[rczPatchHalfWidth[0] : rczUpperBoundOfCentralVoxelsLabels[0],
                                                                                          rczPatchHalfWidth[1] : rczUpperBoundOfCentralVoxelsLabels[1],
                                                                                          rczPatchHalfWidth[2] : rczUpperBoundOfCentralVoxelsLabels[2]]
    
    #FOR THE SUBSAMPLED IMAGE-PARTS:
    if usingSubsampledWaypath :
        #this datastructure is similar to channelsForThisImagePart, but contains voxels from the subsampled image.
        subsampledChannelsForThisImagePart = getImagePartFromSubsampledImageForTraining(imagePartDimensions,
                                                                                        patchDimensions,
                                                                                        allSubsampledChannelsOfPatientInNpArray,
                                                                                        sliceCoordsOfThisImagePart,
                                                                                        subSamplingFactor,
                                                                                        subsampledImagePartDimensions
                                                                                        )
        #############################
        #Normalization-Augmentation of the Patches! For more randomness.
        #Get parameters by how much to renormalize-augment mean and std.
        if training0orValidation1 == 0 and normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 2 and useSameSubChannelsAsSingleScale:
            #Use values  computed on the normal-resolution images. BUT PREREQUISITE IS TO HAVE THE SAME CHANNELS IN THE TWO PATHWAYS. Else need to recompute!
            subsampledChannelsForThisImagePart[:numberOfNormalScaleChannels] = (subsampledChannelsForThisImagePart[:numberOfNormalScaleChannels] + howMuchToAddForEachChannel)*howMuchToMultiplyForEachChannel
        elif normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 2 : 
            #Need to recompute. NOT IMPLEMENTED YET.
            myLogger.print3("ERROR: The system uses different channels for normal and subsampled pathway. And was asked to use Data Augmentation with intensity-noise. Not implemented yet. Exiting.")
            exit(1)
        ##############################
    else :
        subsampledChannelsForThisImagePart = None
        
    return [ channelsForThisImagePart, lesionLabelsForTheCentralClassifiedPartOfThisImagePart, subsampledChannelsForThisImagePart]


def extractDataOfSegmentsUsingSampledSliceCoords(sliceCoordsOfSegmentsToExtract,
                                                imagePartDimensions,
                                                channelsOfImageNpArray,#chans,niiDims
                                                channelsOfSubsampledImageNpArray, #chans,niiDims
                                                patchDimensions,
                                                subsampledImageChannels,
                                                subSamplingFactor,
                                                subsampledImagePartDimensions,
                                                ) :
    numberOfSegmentsToExtract = len(sliceCoordsOfSegmentsToExtract)
    numberOfChannels = channelsOfImageNpArray.shape[0]
    channelsForPartsToReturn = np.zeros( [numberOfSegmentsToExtract, numberOfChannels] + imagePartDimensions, dtype= "float32")
    
    if isinstance(channelsOfSubsampledImageNpArray, (np.ndarray)) : # Using subsampled pathway
        numberOfChannelsSubsampled = channelsOfSubsampledImageNpArray.shape[0]
        channelsForSubsampledPartsToReturn = np.zeros([numberOfSegmentsToExtract, numberOfChannelsSubsampled] + subsampledImagePartDimensions , dtype="float32")
    else :
        channelsForSubsampledPartsToReturn = None
        
    for segment_i in xrange(numberOfSegmentsToExtract) :
        rLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][0][0]; rFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][0][1]
        cLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][1][0]; cFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][1][1]
        zLowBoundary = sliceCoordsOfSegmentsToExtract[segment_i][2][0]; zFarBoundary = sliceCoordsOfSegmentsToExtract[segment_i][2][1]
        channelsForPartsToReturn[segment_i] = channelsOfImageNpArray[:,
                                                                    rLowBoundary:rFarBoundary+1,
                                                                    cLowBoundary:cFarBoundary+1,
                                                                    zLowBoundary:zFarBoundary+1
                                                                    ]
        #Subsampled pathway
        if isinstance(channelsOfSubsampledImageNpArray, (np.ndarray)) :
            imagePartSlicesCoords = sliceCoordsOfSegmentsToExtract[segment_i] #[ [rLowBoundary, rFarBoundary], [cLowBoundary, cFarBoundary], [zLowBoundary, zFarBoundary] ] #the right hand values are placeholders in this case.
            channelsForSubsampledPartsToReturn[segment_i] = getImagePartFromSubsampledImageForTraining(
                                                                                                    imagePartDimensions,
                                                                                                    patchDimensions,
                                                                                                    subsampledImageChannels,
                                                                                                    imagePartSlicesCoords,
                                                                                                    subSamplingFactor,
                                                                                                    subsampledImagePartDimensions
                                                                                                    )
    return [channelsForPartsToReturn, channelsForSubsampledPartsToReturn]


def shuffleTheSegmentsForThisSubepoch(  imagePartsChannelsToLoadOnGpuForSubepoch,
                                        lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch,
                                        subsampledImagePartsChannelsToLoadOnGpuForSubepoch,
                                        usingSubsampledWaypath) :
    if usingSubsampledWaypath :
        combined = zip(imagePartsChannelsToLoadOnGpuForSubepoch,
               lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch,
               subsampledImagePartsChannelsToLoadOnGpuForSubepoch)
        random.shuffle(combined)
        imagePartsChannelsToLoadOnGpuForSubepoch[:],\
                lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch[:],\
                subsampledImagePartsChannelsToLoadOnGpuForSubepoch[:] = zip(*combined)
    else :
        combined = zip(imagePartsChannelsToLoadOnGpuForSubepoch,
                       lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch)
        random.shuffle(combined)
        imagePartsChannelsToLoadOnGpuForSubepoch[:],\
            lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch[:] = zip(*combined)
        
    return [imagePartsChannelsToLoadOnGpuForSubepoch, lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch, subsampledImagePartsChannelsToLoadOnGpuForSubepoch]


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
        numberOfSamplesFromThisCategoryPerSubepochPerImage = arrayNumberOfSegmentsToExtractPerSamplingCategory[cat_i] / numOfSubjectsLoadingThisSubepochForSampling                
        arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject[cat_i] += numberOfSamplesFromThisCategoryPerSubepochPerImage
        numberOfSamplesFromThisCategoryPerSubepochLeftUnevenly = arrayNumberOfSegmentsToExtractPerSamplingCategory[cat_i] % numOfSubjectsLoadingThisSubepochForSampling
        for i_unevenSampleFromThisCat in xrange(numberOfSamplesFromThisCategoryPerSubepochLeftUnevenly):
            arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject[cat_i, random.randint(0, numOfSubjectsLoadingThisSubepochForSampling-1)] += 1
            
    return arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject

#-----------The function that is executed in parallel with gpu training:----------------
def getTheArraysOfImageChannelsAndLesionsToLoadToGpuForSubepoch(myLogger,
                                                                training0orValidation1,
                                                                
                                                                n_subjects_per_subepoch,
                                                                
                                                                numberOfImagePartsToLoadInGpuPerSubepoch,
                                                                
                                                                samplingTypeInstance,
                                                                
                                                                usingSubsampledWaypath,
                                                                
                                                                listOfFilepathsToEachChannelOfEachPatient,
                                                                
                                                                listOfFilepathsToGtLabelsOfEachPatientTrainOrVal,
                                                                
                                                                providedRoiMaskBool,
                                                                listOfFilepathsToRoiMaskOfEachPatient,
                                                                
                                                                providedWeightMapsToSampleForEachCategory,
                                                                forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatient,
                                                                
                                                                useSameSubChannelsAsSingleScale,
                                                                
                                                                listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                                                
                                                                imagePartDimensions,
                                                                patchDimensions,
                                                                subSamplingFactor,
                                                                subsampledImagePartDimensions,
                                                                
                                                                padInputImagesBool,
                                                                smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                                normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                                reflectImageWithHalfProbDuringTraining
                                                                ):
    start_getAllImageParts_time = time.clock()
    
    trainingOrValidationString = "Training" if training0orValidation1 == 0 else "Validation"
    
    myLogger.print3(":=:=:=:=:=:=:=:=: Starting to extract Segments from the images for next " + trainingOrValidationString + "... :=:=:=:=:=:=:=:=:")
    
    total_number_of_subjects = len(listOfFilepathsToEachChannelOfEachPatient)
    randomIndicesList_for_gpu = get_random_subject_indices_to_load_on_GPU(total_number_of_subjects = total_number_of_subjects,
                                                                        max_subjects_on_gpu_for_subepoch = n_subjects_per_subepoch,
                                                                        get_max_subjects_for_gpu_even_if_total_less = False,
                                                                        myLogger=myLogger)
    myLogger.print3("Out of [" + str(total_number_of_subjects) + "] subjects given for [" + trainingOrValidationString + "], it was specified to extract Segments from maximum [" + str(n_subjects_per_subepoch) + "] per subepoch.")
    myLogger.print3("Shuffled indices of subjects that were randomly chosen: "+str(randomIndicesList_for_gpu))
    
    imagePartsChannelsToLoadOnGpuForSubepoch = [] #This is x. Will end up with dimensions: partImagesLoadedPerSubepoch, channels, r,c,z, but flattened.
    lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch = [] # Labels only for the central/predicted part of segments.
    subsampledImagePartsChannelsToLoadOnGpuForSubepoch = []
    
    numOfSubjectsLoadingThisSubepochForSampling = len(randomIndicesList_for_gpu) #Can be different than n_subjects_per_subepoch, cause of available images number.
    
    # This is to separate each sampling category (fore/background, uniform, full-image, weighted-classes)
    stringsPerCategoryToSample = samplingTypeInstance.getStringsPerCategoryToSample()
    numberOfCategoriesToSample = samplingTypeInstance.getNumberOfCategoriesToSample()
    percentOfSamplesPerCategoryToSample = samplingTypeInstance.getPercentOfSamplesPerCategoryToSample()
    arrayNumberOfSegmentsToExtractPerSamplingCategoryAndSubject = getNumberOfSegmentsToExtractPerCategoryFromEachSubject(numberOfImagePartsToLoadInGpuPerSubepoch,
                                                                                                                        percentOfSamplesPerCategoryToSample,
                                                                                                                        numOfSubjectsLoadingThisSubepochForSampling)
    numberOfNormalScaleChannels = len(listOfFilepathsToEachChannelOfEachPatient[0])
    
    myLogger.print3("SAMPLING: Starting iterations to extract Segments from each subject for next " + trainingOrValidationString + "...")
    
    for index_for_vector_with_images_on_gpu in xrange(0, numOfSubjectsLoadingThisSubepochForSampling) :
        myLogger.print3("SAMPLING: Going to load the images and extract segments from the subject #" + str(index_for_vector_with_images_on_gpu + 1) + "/" +str(numOfSubjectsLoadingThisSubepochForSampling))
        
        [allChannelsOfPatientInNpArray, #a nparray(channels,dim0,dim1,dim2)
        gtLabelsImage,
        roiMask,
        arrayWithWeightMapsWhereToSampleForEachCategory, #can be returned "placeholderNothing" if it's testing phase or not "provided weighted maps". In this case, I will sample from GT/ROI.
        allSubsampledChannelsOfPatientInNpArray,  #a nparray(channels,dim0,dim1,dim2)
        tupleOfPaddingPerAxesLeftRight #( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). All 0s when no padding.
        ] = actual_load_patient_images_from_filepath_and_return_nparrays(
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
                                                
                                                usingSubsampledWaypath=usingSubsampledWaypath,
                                                listOfFilepathsToEachSubsampledChannelOfEachPatient=listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                                
                                                padInputImagesBool=padInputImagesBool,
                                                cnnReceptiveField=patchDimensions, # only used if padInputsBool
                                                imagePartDimensions=imagePartDimensions, # only used if padInputsBool
                                                
                                                smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage = smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc=normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                reflectImageWithHalfProb = reflectImageWithHalfProbDuringTraining
                                                )
        myLogger.print3("DEBUG: Index of this case in the original user-defined list of subjects: " + str(randomIndicesList_for_gpu[index_for_vector_with_images_on_gpu]))
        myLogger.print3("Images for subject loaded.")
        ########################
        #For normalization-augmentation: Get channels' stds if needed:
        stdsOfTheChannsOfThisImage = np.ones(numberOfNormalScaleChannels, dtype="float32")
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
                                                imagePartDimensions = imagePartDimensions,
                                                dimensionsOfImageChannel = dimensionsOfImageChannel, #image dimensions for this subject. All images should have the same.
                                                weightMapToSampleFrom=finalWeightMapToSampleFromForThisCat)
            myLogger.print3("Finished sampling segments of Category [" + catString + "]. Number sampled: " + str( len(imagePartsSampled[0][0]) ) )
            
            # Use the just sampled coordinates of slices to actually extract the segments (data) from the subject's images. 
            for image_part_i in xrange(len(imagePartsSampled[0][0])) :
                
                sliceCoordsOfThisImagePart = imagePartsSampled[1][:,image_part_i,:] #[0] is the central voxel coords.
                
                [channelsForThisImagePart,
                lesionLabelsForTheCentralClassifiedPartOfThisImagePart, # used to be lesionLabelsForThisImagePart, before extracting only for the central voxels.
                subsampledChannelsForThisImagePart] = extractDataOfASegmentFromImagesUsingSampledSliceCoords(
                                                                                        training0orValidation1,
                                                                                        
                                                                                        sliceCoordsOfThisImagePart,
                                                                                        numberOfNormalScaleChannels,
                                                                                        imagePartDimensions,
                                                                                        patchDimensions,
                                                                                        
                                                                                        allChannelsOfPatientInNpArray,
                                                                                        gtLabelsImage,
                                                                                        
                                                                                        usingSubsampledWaypath,
                                                                                        useSameSubChannelsAsSingleScale,
                                                                                        allSubsampledChannelsOfPatientInNpArray,
                                                                                        subSamplingFactor,
                                                                                        subsampledImagePartDimensions,
                                                                                        # Intensity Augmentation
                                                                                        normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                                                        stdsOfTheChannsOfThisImage
                                                                                        )
                imagePartsChannelsToLoadOnGpuForSubepoch.append(channelsForThisImagePart)
                lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch.append(lesionLabelsForTheCentralClassifiedPartOfThisImagePart)
                if usingSubsampledWaypath :
                    subsampledImagePartsChannelsToLoadOnGpuForSubepoch.append(subsampledChannelsForThisImagePart)
                    
    #I need to shuffle them, together imageParts and lesionParts!
    [imagePartsChannelsToLoadOnGpuForSubepoch,
    lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch,
    subsampledImagePartsChannelsToLoadOnGpuForSubepoch ] = shuffleTheSegmentsForThisSubepoch(imagePartsChannelsToLoadOnGpuForSubepoch,
                                                                                            lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch,
                                                                                            subsampledImagePartsChannelsToLoadOnGpuForSubepoch,
                                                                                            usingSubsampledWaypath)
    
    end_getAllImageParts_time = time.clock()
    myLogger.print3("TIMING: Extracting all the Segments for next " + trainingOrValidationString + " took time: "+str(end_getAllImageParts_time-start_getAllImageParts_time)+"(s)")
    
    myLogger.print3(":=:=:=:=:=:=:=:=: Finished extracting Segments from the images for next " + trainingOrValidationString + ". :=:=:=:=:=:=:=:=:")
    
    return [np.asarray(imagePartsChannelsToLoadOnGpuForSubepoch, dtype="float32"),
            np.asarray(lesionLabelsForTheCentralPredictedPartOfSegmentsInGpUForSubepoch, dtype="float32"),
            np.asarray(subsampledImagePartsChannelsToLoadOnGpuForSubepoch, dtype="float32")]
    
    
#A main routine in do_training, that runs for every batch of validation and training.
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
        printProgressStep = max(1, number_of_batches/5)
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


#---------------------------------------------TRAINING-------------------------------------
#batch_size should be 1 or even.
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
                n_epochs, #every epoch I save my cnnModel
                number_of_subepochs, #per epoch. Every subepoch I get my Accuracy reported
                n_subjects_per_subepoch,  #the max that can be fit in CPU memory. these are never in GPU. Only ImageParts in GPU
                imagePartsLoadedInGpuPerSubepoch, #Keep this even for now. So that I have same number of pos-neg PAIRS. If it's odd, still will be int divided by two so the lower even will be used.
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
                
                listOfFilepathsToEachSubsampledChannelOfEachPatientTraining,
                listOfFilepathsToEachSubsampledChannelOfEachPatientValidation,
                
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
    
    usingSubsampledWaypath = len(cnn3dInstance.cnnLayersSubsampled)>0 #Flag that says if I should be loading subsampled channels etc.
    
    start_training_time = time.clock()
    
    patchDimensions = cnn3dInstance.patchDimensions
    imagePartDimensionsTraining = cnn3dInstance.imagePartDimensionsTraining
    imagePartDimensionsValidation = cnn3dInstance.imagePartDimensionsValidation
    subsampledImagePartDimensionsTraining = cnn3dInstance.subsampledImagePartDimensionsTraining
    subsampledImagePartDimensionsValidation = cnn3dInstance.subsampledImagePartDimensionsValidation
    subSamplingFactor = cnn3dInstance.subsampleFactor
    
    #---------To run PARALLEL the extraction of parts for the next subepoch---
    ppservers = () # tuple of all parallel python servers to connect with
    job_server = pp.Server(ncpus=1, ppservers=ppservers) # Creates jobserver with automatically detected number of workers
    
    tupleWithParametersForTraining = (myLogger,
                                    0,
                                    n_subjects_per_subepoch,
                                    
                                    imagePartsLoadedInGpuPerSubepoch,
                                    samplingTypeInstanceTraining,
                                    
                                    usingSubsampledWaypath,
                                    
                                    listOfFilepathsToEachChannelOfEachPatientTraining,
                                    
                                    listOfFilepathsToGtLabelsOfEachPatientTraining,
                                    
                                    providedRoiMaskForTrainingBool,
                                    listOfFilepathsToRoiMaskOfEachPatientTraining,
                                    
                                    providedWeightMapsToSampleForEachCategoryTraining,
                                    forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatientTraining,
                                    
                                    useSameSubChannelsAsSingleScale,
                                    
                                    listOfFilepathsToEachSubsampledChannelOfEachPatientTraining,
                                    
                                    imagePartDimensionsTraining,
                                    patchDimensions,
                                    subSamplingFactor,
                                    subsampledImagePartDimensionsTraining,
                                    
                                    padInputImagesBool,
                                    smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                    normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                    reflectImageWithHalfProbDuringTraining
                                    )
    tupleWithParametersForValidation = (myLogger,
                                    1,
                                    len(listOfFilepathsToEachChannelOfEachPatientValidation),
                                    
                                    imagePartsLoadedInGpuPerSubepochValidation,
                                    samplingTypeInstanceValidation,
                                    
                                    usingSubsampledWaypath,
                                    
                                    listOfFilepathsToEachChannelOfEachPatientValidation,
                                    
                                    listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc,
                                    
                                    providedRoiMaskForValidationBool,
                                    listOfFilepathsToRoiMaskOfEachPatientValidation,
                                    
                                    providedWeightMapsToSampleForEachCategoryValidation,
                                    forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatientValidation,
                                    
                                    useSameSubChannelsAsSingleScale,
                                    
                                    listOfFilepathsToEachSubsampledChannelOfEachPatientValidation,
                                    
                                    imagePartDimensionsValidation,
                                    patchDimensions,
                                    subSamplingFactor,
                                    subsampledImagePartDimensionsValidation,
                                    
                                    padInputImagesBool,
                                    smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                    [0, -1,-1,-1], #don't perform intensity-augmentation during validation.
                                    [0,0,0] #don't perform reflection-augmentation during validation.
                                    )
    tupleWithLocalFunctionsThatWillBeCalledByTheMainJob = ( get_random_subject_indices_to_load_on_GPU,
                                                            actual_load_patient_images_from_filepath_and_return_nparrays,
                                                            smoothImageWithGaussianFilterIfNeeded,
                                                            reflectImageArrayIfNeeded,
                                                            padCnnInputs,
                                                            getNumberOfSegmentsToExtractPerCategoryFromEachSubject,
                                                            sampleImageParts,
                                                            extractDataOfASegmentFromImagesUsingSampledSliceCoords,
                                                            getImagePartFromSubsampledImageForTraining,
                                                            shuffleTheSegmentsForThisSubepoch
                                                            )
    tupleWithModulesToImportWhichAreUsedByTheJobFunctions = ("random", "time", "numpy as np", "nibabel as nib", "math", "from deepmedic.genericHelpers import *", "from scipy.ndimage.filters import gaussian_filter")
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
                    [imagePartsChannelsToLoadOnGpuForSubepochValidation,
                    lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochValidation,
                    subsampledImagePartsChannelsToLoadOnGpuForSubepochValidation] = getTheArraysOfImageChannelsAndLesionsToLoadToGpuForSubepoch(myLogger,
                                                                                        1,
                                                                                        
                                                                                        len(listOfFilepathsToEachChannelOfEachPatientValidation),
                                                                                        imagePartsLoadedInGpuPerSubepochValidation,
                                                                                        samplingTypeInstanceValidation,
                                                                                        usingSubsampledWaypath,
                                                                                        
                                                                                        listOfFilepathsToEachChannelOfEachPatientValidation,
                                                                                        
                                                                                        listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc,
                                                                                        
                                                                                        providedRoiMaskForValidationBool,
                                                                                        listOfFilepathsToRoiMaskOfEachPatientValidation,
                                                                                        
                                                                                        providedWeightMapsToSampleForEachCategoryValidation,
                                                                                        forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatientValidation,
                                                                                        
                                                                                        useSameSubChannelsAsSingleScale,
                                                                                        
                                                                                        listOfFilepathsToEachSubsampledChannelOfEachPatientValidation,
                                                                                        
                                                                                        imagePartDimensionsValidation,
                                                                                        patchDimensions,
                                                                                        subSamplingFactor,
                                                                                        subsampledImagePartDimensionsValidation,
                                                                                        
                                                                                        padInputImagesBool,
                                                                                        smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                                                        normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc=[0,-1,-1,-1],
                                                                                        reflectImageWithHalfProbDuringTraining = [0,0,0]
                                                                                        )
                    boolItIsTheVeryFirstSubepochOfThisProcess = False
                else : #It was done in parallel with the training of the previous epoch, just grab the results...
                    [imagePartsChannelsToLoadOnGpuForSubepochValidation,
                    lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochValidation,
                    subsampledImagePartsChannelsToLoadOnGpuForSubepochValidation] = parallelJobToGetDataForNextValidation() #fromParallelProcessing that had started from last loop when it was submitted.
                    
                #------------------------------LOAD DATA FOR VALIDATION----------------------
                myLogger.print3("Loading Validation data for subepoch #"+str(subepoch)+" on shared variable...")
                start_loadingToGpu_time = time.clock()
                
                numberOfBatchesValidation = len(imagePartsChannelsToLoadOnGpuForSubepochValidation) / cnn3dInstance.batchSizeValidation #Computed with number of extracted samples, in case I dont manage to extract as many as I wanted initially.
                
                myLogger.print3("DEBUG: For Validation, loading to shared variable that many Segments: " + str(len(imagePartsChannelsToLoadOnGpuForSubepochValidation))) 
                cnn3dInstance.sharedValidationNiiData_x.set_value(imagePartsChannelsToLoadOnGpuForSubepochValidation, borrow=borrowFlag)
                cnn3dInstance.sharedValidationNiiLabels_y.set_value(lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochValidation, borrow=borrowFlag)
                
                if usingSubsampledWaypath :
                    cnn3dInstance.sharedValidationSubsampledData_x.set_value(subsampledImagePartsChannelsToLoadOnGpuForSubepochValidation, borrow=borrowFlag)
                    
                end_loadingToGpu_time = time.clock()
                myLogger.print3("TIMING: Loading sharedVariables for Validation in epoch|subepoch="+str(epoch)+"|"+str(subepoch)+" took time: "+str(end_loadingToGpu_time-start_loadingToGpu_time)+"(s)")
                #TRY TO CLEAR THE VARIABLES before the parallel job starts loading stuff again? Or will it cause problems because the shared variables are borrow=True?
                imagePartsChannelsToLoadOnGpuForSubepochValidation = ""
                lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochValidation = ""
                if usingSubsampledWaypath :
                    subsampledImagePartsChannelsToLoadOnGpuForSubepochValidation = ""
                    
                    
                #------------------------SUBMIT PARALLEL JOB TO GET TRAINING DATA FOR NEXT TRAINING-----------------
                #submit the parallel job
                myLogger.print3("PARALLEL: Before Validation in subepoch #" +str(subepoch) + ", the parallel job for extracting Segments for the next Training is submitted.")
                parallelJobToGetDataForNextTraining = job_server.submit(getTheArraysOfImageChannelsAndLesionsToLoadToGpuForSubepoch, #local function to call and execute in parallel.
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
                [imagePartsChannelsToLoadOnGpuForSubepochTraining,
                lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining,
                subsampledImagePartsChannelsToLoadOnGpuForSubepochTraining] = getTheArraysOfImageChannelsAndLesionsToLoadToGpuForSubepoch(myLogger,
                                                                           0,
                                                                           n_subjects_per_subepoch,
                                                                           imagePartsLoadedInGpuPerSubepoch,
                                                                           samplingTypeInstanceTraining,
                                                                           usingSubsampledWaypath,
                                                                           
                                                                           listOfFilepathsToEachChannelOfEachPatientTraining,
                                                                           
                                                                           listOfFilepathsToGtLabelsOfEachPatientTraining,
                                                                           
                                                                           providedRoiMaskForTrainingBool,
                                                                           listOfFilepathsToRoiMaskOfEachPatientTraining,
                                                                           
                                                                           providedWeightMapsToSampleForEachCategoryTraining,
                                                                           forEachSamplingCategory_aListOfFilepathsToWeightMapsOfEachPatientTraining,
                                                                           
                                                                           useSameSubChannelsAsSingleScale,
                                                                           
                                                                           listOfFilepathsToEachSubsampledChannelOfEachPatientTraining,
                                                                           
                                                                           imagePartDimensionsTraining,
                                                                           patchDimensions,
                                                                           subSamplingFactor,
                                                                           subsampledImagePartDimensionsTraining,
                                                                           
                                                                           padInputImagesBool,
                                                                           smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                                           normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
                                                                           reflectImageWithHalfProbDuringTraining
                                                                           )
                boolItIsTheVeryFirstSubepochOfThisProcess = False
            else :
                #It was done in parallel with the validation (or with previous training iteration, in case I am not performing validation).
                [imagePartsChannelsToLoadOnGpuForSubepochTraining,
                lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining,
                subsampledImagePartsChannelsToLoadOnGpuForSubepochTraining] = parallelJobToGetDataForNextTraining() #fromParallelProcessing that had started from last loop when it was submitted.
                
            #-------------------------COMPUTE CLASS-WEIGHTS, TO WEIGHT COST FUNCTION AND COUNTER CLASS IMBALANCE----------------------
            #Do it for only few epochs, until I get to an ok local minima neighbourhood.
            if cnn3dInstance.numberOfEpochsTrained < numberOfEpochsToWeightTheClassesInTheCostFunction :
                numOfPatchesInTheSubepoch_notParts = np.prod(lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining.shape)
                actualNumOfPatchesPerClassInTheSubepoch_notParts = np.bincount(np.ravel(lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining).astype(int))
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
                
            #----------------------------------LOAD TRAINING DATA ON GPU-------------------------------
            myLogger.print3("Loading Training data for subepoch #"+str(subepoch)+" on shared variable...")
            start_loadingToGpu_time = time.clock()
            
            numberOfBatchesTraining = len(imagePartsChannelsToLoadOnGpuForSubepochTraining) / cnn3dInstance.batchSize #Computed with number of extracted samples, in case I dont manage to extract as many as I wanted initially.
            
            cnn3dInstance.sharedTrainingNiiData_x.set_value(imagePartsChannelsToLoadOnGpuForSubepochTraining, borrow=borrowFlag)
            cnn3dInstance.sharedTrainingNiiLabels_y.set_value(lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining, borrow=borrowFlag)
            if usingSubsampledWaypath :
                cnn3dInstance.sharedTrainingSubsampledData_x.set_value(subsampledImagePartsChannelsToLoadOnGpuForSubepochTraining, borrow=borrowFlag)
                
            end_loadingToGpu_time = time.clock()
            myLogger.print3("TIMING: Loading sharedVariables for Training in epoch|subepoch="+str(epoch)+"|"+str(subepoch)+" took time: "+str(end_loadingToGpu_time-start_loadingToGpu_time)+"(s)")
            #TRY TO CLEAR THE VARIABLES before the parallel job starts loading stuff again? Or will it cause problems because the shared variables are borrow=True?
            imagePartsChannelsToLoadOnGpuForSubepochTraining = ""
            lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining = ""
            if usingSubsampledWaypath :
                subsampledImagePartsChannelsToLoadOnGpuForSubepochTraining = ""
                
            #------------------------SUBMIT PARALLEL JOB TO GET VALIDATION/TRAINING DATA (if val is/not performed) FOR NEXT SUBEPOCH-----------------
            if performValidationOnSamplesDuringTrainingProcessBool :
                #submit the parallel job
                myLogger.print3("PARALLEL: Before Training in subepoch #" +str(subepoch) + ", submitting the parallel job for extracting Segments for the next Validation.")
                parallelJobToGetDataForNextValidation = job_server.submit(getTheArraysOfImageChannelsAndLesionsToLoadToGpuForSubepoch, #local function to call and execute in parallel.
                                                                            tupleWithParametersForValidation, #tuple with the arguments required
                                                                            tupleWithLocalFunctionsThatWillBeCalledByTheMainJob, #tuple of local functions that I need to call
                                                                            tupleWithModulesToImportWhichAreUsedByTheJobFunctions) #tuple of the external modules that I need, of which I am calling functions (not the mods of the ext-functions).
            else : #extract in parallel the samples for the next subepoch's training.
                myLogger.print3("PARALLEL: Before Training in subepoch #" +str(subepoch) + ", submitting the parallel job for extracting Segments for the next Training.")
                parallelJobToGetDataForNextTraining = job_server.submit(getTheArraysOfImageChannelsAndLesionsToLoadToGpuForSubepoch, #local function to call and execute in parallel.
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
            print "DEBUG: new learning rate was calculated: ", newLearningRate, " and new Momentum :", newMomentum
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
        
        if performFullInferenceOnValidationImagesEveryFewEpochsBool and (cnn3dInstance.numberOfEpochsTrained <> 0) and (cnn3dInstance.numberOfEpochsTrained % everyThatManyEpochsComputeDiceOnTheFullValidationImages == 0) :
            myLogger.print3("***Starting validation with Full Inference / Segmentation on validation subjects for Epoch #"+str(epoch)+"...***")
            validation0orTesting1 = 0
            #do_validation_or_testing(myLogger,
            performInferenceForTestingOnWholeVolumes(myLogger,
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
    
    
#---------------------------------------------TESTING-------------------------------------

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
    
    usingSubsampledWaypath = len(cnn3dInstance.cnnLayersSubsampled)>0 #Flag that says if I should be loading subsampled channels etc.
    
    total_number_of_images = len(listOfFilepathsToEachChannelOfEachPatient)    
    batch_size = cnn3dInstance.batchSizeTesting
    
    #one dice score for whole + for each class)
    # A list of dimensions: total_number_of_images X NUMBER_OF_CLASSES
    diceCoeffs1 = [ [-1] * NUMBER_OF_CLASSES for i in xrange(total_number_of_images) ] #AllpredictedLes/AllLesions
    diceCoeffs2 = [ [-1] * NUMBER_OF_CLASSES for i in xrange(total_number_of_images) ] #predictedInsideBrainmask/AllLesions
    diceCoeffs3 = [ [-1] * NUMBER_OF_CLASSES for i in xrange(total_number_of_images) ] #predictedInsideBrainMask/ LesionsInsideBrainMAsk (for comparisons)
    
    patchDimensions = cnn3dInstance.patchDimensions
    imagePartDimensions = cnn3dInstance.imagePartDimensionsTesting
    subsampledImagePartDimensions = cnn3dInstance.subsampledImagePartDimensionsTesting
    subSamplingFactor = cnn3dInstance.subsampleFactor
    
    #stride is how much I move in each dimension to acquire the next imagePart. 
    #I move exactly the number I segment in the centre of each image part (originally this was 9^3 segmented per imagePart).
    numberOfCentralVoxelsClassified = cnn3dInstance.finalLayer.outputShapeTest[2:]
    strideOfImagePartsPerDimensionInVoxels = numberOfCentralVoxelsClassified
    
    rPatchHalfWidth = (patchDimensions[0]-1)/2
    cPatchHalfWidth = (patchDimensions[1]-1)/2
    zPatchHalfWidth = (patchDimensions[2]-1)/2
    
    #Find the total number of feature maps that will be created:
    #NOTE: saveIndividualFmImagesForVisualisation should contain an entry per pathwayType, even if just []. If not [], the list should contain one entry per layer of the pathway, even if just []. The layer entries, if not [], they should have to integers, lower and upper FM to visualise.
    if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
        totalNumberOfFMsToProcess = 0
        for pathwayType_i in xrange(0, len(cnn3dInstance.typesOfCnnLayers)) :
            indicesOfFmsToVisualisePerLayerOfCertainPathway = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[pathwayType_i]
            if indicesOfFmsToVisualisePerLayerOfCertainPathway<>[] :
                for layer_i in xrange(0, len(cnn3dInstance.typesOfCnnLayers[pathwayType_i])) :
                    indicesOfFmsToVisualiseForCertainLayerOfCertainPathway = indicesOfFmsToVisualisePerLayerOfCertainPathway[layer_i]
                    if indicesOfFmsToVisualiseForCertainLayerOfCertainPathway<>[] :
                        #If the user specifies to grab more feature maps than exist (eg 9999), correct it, replacing it with the number of FMs in the layer.
                        numberOfFeatureMapsInThisLayer = cnn3dInstance.typesOfCnnLayers[pathwayType_i][layer_i].getNumberOfFeatureMaps()
                        indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] = min(indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1], numberOfFeatureMapsInThisLayer)
                        totalNumberOfFMsToProcess += indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] - indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[0]                            
                        
    for image_i in xrange(total_number_of_images) :
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~ Segmenting subject with index #"+str(image_i)+" ~~~~~~~~~~~~~~~~~~~~")
        
        #load the image channels in cpu
        
        [imageChannels, #a nparray(channels,dim0,dim1,dim2)
        gtLabelsImage, #only for accurate/correct DICE1-2 calculation
        brainMask, 
        arrayWithWeightMapsWhereToSampleForEachCategory, #only used in training. Placeholder here.
        allSubsampledChannelsOfPatientInNpArray,  #a nparray(channels,dim0,dim1,dim2)
        tupleOfPaddingPerAxesLeftRight #( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). All 0s when no padding.
        ] = actual_load_patient_images_from_filepath_and_return_nparrays(
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
                                                    usingSubsampledWaypath = usingSubsampledWaypath,
                                                    listOfFilepathsToEachSubsampledChannelOfEachPatient = listOfFilepathsToEachSubsampledChannelOfEachPatient,
                                                    
                                                    padInputImagesBool = padInputImagesBool,
                                                    cnnReceptiveField = patchDimensions, # only used if padInputsBool
                                                    imagePartDimensions = imagePartDimensions, # only used if padInputsBool
                                                    
                                                    smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage = smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                                                    normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc = [0, -1,-1,-1],
                                                    reflectImageWithHalfProb = [0,0,0]
                                                    )
        niiDimensions = list(imageChannels[0].shape)
        #The probability-map that will be constructed by the predictions.
        labelImageCreatedByPredictions = np.zeros([NUMBER_OF_CLASSES]+niiDimensions, dtype = "float32")
        #create the big array that will hold all the fms (for feature extraction, to save as a big multi-dim image).
        if saveIndividualFmImagesForVisualisation or saveMultidimensionalImageWithAllFms:
            multidimensionalImageWithAllToBeVisualisedFmsArray =  np.zeros([totalNumberOfFMsToProcess] + niiDimensions, dtype = "float32")
            
        #get all the parts of the image on the cpu
        [ sliceCoordsOfSegmentsInImage ] = getCoordsOfAllSegmentsOfAnImage(myLogger=myLogger,
                                                                         strideOfImagePartsPerDimensionInVoxels=strideOfImagePartsPerDimensionInVoxels,
                                                                         imagePartDimensions = imagePartDimensions,
                                                                         batch_size = batch_size,
                                                                         channelsOfImageNpArray = imageChannels,#chans,niiDims
                                                                         brainMask = brainMask
                                                                         )
        
        myLogger.print3("Starting to segment each image-part by calling the cnn.cnnTestModel(i). This part takes a few mins per volume...")
        
        #In the next part, for each imagePart in a batch I get from the cnn a vector with labels for the central voxels of the imagepart (9^3 originally).
        #I will reshape the 9^3 vector to a cube and "put it" in the new-segmentation-image, where it corresponds.
        #I have to find exactly to which voxels these labels correspond to. Consider that the image part is bigger than the 9^3 label box...
        #by half-patch at the top and half-patch at the bottom of each dimension.
        
        #Here I calculate how many imageParts can fit in each r-c-z direction/dimension.
        #It is how many times the stride (originally 9^3) can fit in the niiDimension-1patch (half up, half bottom)
        imagePartsPerRdirection = (niiDimensions[0]-patchDimensions[0]+1) / strideOfImagePartsPerDimensionInVoxels[0]
        imagePartsPerCdirection = (niiDimensions[1]-patchDimensions[1]+1) / strideOfImagePartsPerDimensionInVoxels[1]
        imagePartsPerZdirection = (niiDimensions[2]-patchDimensions[2]+1) / strideOfImagePartsPerDimensionInVoxels[2]
        imagePartsPerZSlice = imagePartsPerRdirection*imagePartsPerCdirection
        
        totalNumberOfImagePartsToProcessForThisImage = len(sliceCoordsOfSegmentsInImage)
        myLogger.print3("Total number of Segments to process:"+str(totalNumberOfImagePartsToProcessForThisImage))
        
        imagePartOfConstructedProbMap_i = 0
        imagePartOfConstructedFeatureMaps_i = 0
        number_of_batches = totalNumberOfImagePartsToProcessForThisImage/batch_size
        extractTimePerSubject = 0; loadingTimePerSubject = 0; fwdPassTimePerSubject = 0
        for batch_i in xrange(number_of_batches) : #batch_size = how many image parts in one batch. Has to be the same with the batch_size it was created with. This is no problem for testing. Could do all at once, or just 1 image part at time.
            
            printProgressStep = max(1, number_of_batches/5)
            if batch_i%printProgressStep == 0:
                myLogger.print3("Processed "+str(batch_i*batch_size)+"/"+str(number_of_batches*batch_size)+" Segments.")
                
            # Extract the data for the segments of this batch. ( I could modularize extractDataOfASegmentFromImagesUsingSampledSliceCoords() of training and use it here as well. )
            start_extract_time = time.clock()
            sliceCoordsOfSegmentsInBatch = sliceCoordsOfSegmentsInImage[ batch_i*batch_size : (batch_i+1)*batch_size ]
            [imagePartsChannelsToLoadOnGpuForThisBatch,
            subsampledImagePartsChannelsToLoadOnGpuForThisBatch] = extractDataOfSegmentsUsingSampledSliceCoords(sliceCoordsOfSegmentsToExtract=sliceCoordsOfSegmentsInBatch,
                                                                                                                imagePartDimensions=imagePartDimensions,
                                                                                                                channelsOfImageNpArray=imageChannels,#chans,niiDims
                                                                                                                channelsOfSubsampledImageNpArray=allSubsampledChannelsOfPatientInNpArray,
                                                                                                                patchDimensions=patchDimensions,
                                                                                                                subsampledImageChannels=allSubsampledChannelsOfPatientInNpArray,
                                                                                                                subSamplingFactor=subSamplingFactor,
                                                                                                                subsampledImagePartDimensions=subsampledImagePartDimensions,
                                                                                                                )
            end_extract_time = time.clock()
            extractTimePerSubject += end_extract_time - start_extract_time
            
            # Load the data of the batch on the GPU
            start_loading_time = time.clock()
            cnn3dInstance.sharedTestingNiiData_x.set_value(imagePartsChannelsToLoadOnGpuForThisBatch, borrow=borrowFlag)
            if usingSubsampledWaypath :
                cnn3dInstance.sharedTestingSubsampledData_x.set_value(subsampledImagePartsChannelsToLoadOnGpuForThisBatch, borrow=borrowFlag)
            end_loading_time = time.clock()
            loadingTimePerSubject += end_loading_time - start_loading_time
            
            # Do the inference
            start_training_time = time.clock()
            featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = cnn3dInstance.cnnTestAndVisualiseAllFmsFunction(0)
            #featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = cnn3dInstance.cnnTestAndVisualiseAllFmsFunction(batch_i)
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
                labelImageCreatedByPredictions[
                        :,
                        coordsOfTopLeftVoxelForThisPart[0] + rPatchHalfWidth : coordsOfTopLeftVoxelForThisPart[0] + rPatchHalfWidth + strideOfImagePartsPerDimensionInVoxels[0],
                        coordsOfTopLeftVoxelForThisPart[1] + cPatchHalfWidth : coordsOfTopLeftVoxelForThisPart[1] + cPatchHalfWidth + strideOfImagePartsPerDimensionInVoxels[1],
                        coordsOfTopLeftVoxelForThisPart[2] + zPatchHalfWidth : coordsOfTopLeftVoxelForThisPart[2] + zPatchHalfWidth + strideOfImagePartsPerDimensionInVoxels[2],
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
                
                for typeOfPathway_i in xrange(len(cnn3dInstance.typesOfCnnLayers)) :
                    layersOfThePathwayToVisualiseOfTheCnnInstance = cnn3dInstance.typesOfCnnLayers[typeOfPathway_i]
                    for layer_i in xrange(len(layersOfThePathwayToVisualiseOfTheCnnInstance)) :
                        indexOfTheLayerInTheReturnedListByTheBatchTraining += 1
                        if indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[typeOfPathway_i]==[] or indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[typeOfPathway_i][layer_i]==[] :
                            continue
                        indicesOfFmsToExtractFromThisLayer = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[typeOfPathway_i][layer_i]
                        
                        fmsReturnedForATestBatchForCertainLayer = listWithTheFmsOfAllLayersSortedByPathwayTypeForTheBatch[indexOfTheLayerInTheReturnedListByTheBatchTraining][:, indicesOfFmsToExtractFromThisLayer[0]:indicesOfFmsToExtractFromThisLayer[1],:,:,:]
                        #We specify a range of fms to visualise from a layer. currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray : highIndexOfFmsInTheMultidimensionalImageToFillInThisIterationExcluding defines were to put them in the multidimensional-image-array.
                        highIndexOfFmsInTheMultidimensionalImageToFillInThisIterationExcluding = currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray + indicesOfFmsToExtractFromThisLayer[1] - indicesOfFmsToExtractFromThisLayer[0]
                        fmImageInMultidimArrayToReconstructInThisIteration = multidimensionalImageWithAllToBeVisualisedFmsArray[currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray: highIndexOfFmsInTheMultidimensionalImageToFillInThisIterationExcluding]
                        
                        #=========================================================================================================================================
                        #====the following calculations could be move OUTSIDE THE FOR LOOPS, by using the kernel-size parameter (from the cnn instance) instead of the shape of the returned value.
                        #====fmsReturnedForATestBatchForCertainLayer.shape[2] - (numberOfCentralVoxelsClassified[0]-1) is essentially the width of the patch left after the convolutions.
                        #====These calculations are pathway and layer-specific. So they could be done once, prior to image processing, and results cached in a list to be accessed during the loop.
                        numberOfVoxToSubtrToGetPatchWidthAtThisFm_R =  numberOfCentralVoxelsClassified[0]-1 if typeOfPathway_i <> 1 else math.ceil((numberOfCentralVoxelsClassified[0]*1.0)/subSamplingFactor[0]) -1
                        numberOfVoxToSubtrToGetPatchWidthAtThisFm_C =  numberOfCentralVoxelsClassified[1]-1 if typeOfPathway_i <> 1 else math.ceil((numberOfCentralVoxelsClassified[1]*1.0)/subSamplingFactor[1]) -1
                        numberOfVoxToSubtrToGetPatchWidthAtThisFm_Z =  numberOfCentralVoxelsClassified[2]-1 if typeOfPathway_i <> 1 else math.ceil((numberOfCentralVoxelsClassified[2]*1.0)/subSamplingFactor[2]) -1
                        rPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions = fmsReturnedForATestBatchForCertainLayer.shape[2] - numberOfVoxToSubtrToGetPatchWidthAtThisFm_R
                        cPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions = fmsReturnedForATestBatchForCertainLayer.shape[3] - numberOfVoxToSubtrToGetPatchWidthAtThisFm_C
                        zPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions = fmsReturnedForATestBatchForCertainLayer.shape[4] - numberOfVoxToSubtrToGetPatchWidthAtThisFm_Z
                        rOfTopLeftCentralVoxelAtTheFm = (rPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions-1)/2 #-1 so that if width is even, I'll get the left voxel from the centre as 1st, which I THINK is how I am getting the patches from the original image.
                        cOfTopLeftCentralVoxelAtTheFm = (cPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions-1)/2
                        zOfTopLeftCentralVoxelAtTheFm = (zPatchDimensionAtTheFmThatWeVisualiseAfterConvolutions-1)/2
                        
                        #the math.ceil / subsamplingFactor is a trick to make it work for even subsamplingFactor too. Eg 9/2=4.5 => Get 5. Combined with the trick at repeat, I get my correct number of central voxels hopefully.
                        numberOfCentralVoxelsToGetInDirectionR = math.ceil((numberOfCentralVoxelsClassified[0]*1.0)/subSamplingFactor[0]) if typeOfPathway_i == 1 else numberOfCentralVoxelsClassified[0]
                        numberOfCentralVoxelsToGetInDirectionC = math.ceil((numberOfCentralVoxelsClassified[1]*1.0)/subSamplingFactor[1]) if typeOfPathway_i == 1 else numberOfCentralVoxelsClassified[1]
                        numberOfCentralVoxelsToGetInDirectionZ = math.ceil((numberOfCentralVoxelsClassified[2]*1.0)/subSamplingFactor[2]) if typeOfPathway_i == 1 else numberOfCentralVoxelsClassified[2]
                        #=========================================================================================================================================
                        
                        #Grab the central voxels of the predicted fms from the cnn in this batch.
                        centralVoxelsOfAllFmsInLayer = fmsReturnedForATestBatchForCertainLayer[:, #batchsize
                                                            :, #number of featuremaps
                                                            rOfTopLeftCentralVoxelAtTheFm:rOfTopLeftCentralVoxelAtTheFm+numberOfCentralVoxelsToGetInDirectionR,
                                                            cOfTopLeftCentralVoxelAtTheFm:cOfTopLeftCentralVoxelAtTheFm+numberOfCentralVoxelsToGetInDirectionC,
                                                            zOfTopLeftCentralVoxelAtTheFm:zOfTopLeftCentralVoxelAtTheFm+numberOfCentralVoxelsToGetInDirectionZ
                                                            ]
                        #If the pathway that is visualised currently is the subsampled, I need to upsample the central voxels to the normal resolution, before reconstructing the image-fm.
                        if typeOfPathway_i == 1: #subsampled layer. Remember that this returns smaller dimension outputs, cause it works in the subsampled space. I need to repeat it, to bring it to the dimensions of the normal-voxel-space.
                            expandedOutputOfFmsR = np.repeat(centralVoxelsOfAllFmsInLayer, subSamplingFactor[0],axis = 2)
                            expandedOutputOfFmsRC = np.repeat(expandedOutputOfFmsR, subSamplingFactor[1],axis = 3)
                            expandedOutputOfFmsRCZ = np.repeat(expandedOutputOfFmsRC, subSamplingFactor[2],axis = 4)
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
                                    coordsOfTopLeftVoxelForThisPart[0] + rPatchHalfWidth : coordsOfTopLeftVoxelForThisPart[0] + rPatchHalfWidth + strideOfImagePartsPerDimensionInVoxels[0],
                                    coordsOfTopLeftVoxelForThisPart[1] + cPatchHalfWidth : coordsOfTopLeftVoxelForThisPart[1] + cPatchHalfWidth + strideOfImagePartsPerDimensionInVoxels[1],
                                    coordsOfTopLeftVoxelForThisPart[2] + zPatchHalfWidth : coordsOfTopLeftVoxelForThisPart[2] + zPatchHalfWidth + strideOfImagePartsPerDimensionInVoxels[2]
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
        
        #=================Save Predicted-Probability-Map and Evaluate Dice====================
        segmentationImage = np.argmax(labelImageCreatedByPredictions, axis=0) #The SEGMENTATION.
        
        #Save Result:
        if savePredictionImagesSegmentationAndProbMapsList[0] == True : #save predicted segmentation
            npDtypeForPredictedImage = np.dtype(np.int16)
            suffixToAdd = "_Segm"
            #Save the image. Pass the filename paths of the normal image so that I can dublicate the header info, eg RAS transformation.
            unpaddedSegmentationImage = segmentationImage if not padInputImagesBool else unpadCnnOutputs(segmentationImage, tupleOfPaddingPerAxesLeftRight)
            savePredictedImageToANewNiiWithHeaderFromOther( unpaddedSegmentationImage,
                                                            listOfNamesToGiveToPredictionsIfSavingResults,
                                                            
                                                            listOfFilepathsToEachChannelOfEachPatient,
                                                            
                                                            image_i,
                                                            suffixToAdd,
                                                            npDtypeForPredictedImage,
                                                            myLogger
                                                            )
        for class_i in xrange(0, NUMBER_OF_CLASSES) :
            if (len(savePredictionImagesSegmentationAndProbMapsList[1]) >= class_i + 1) and (savePredictionImagesSegmentationAndProbMapsList[1][class_i] == True) : #save predicted probMap for class
                npDtypeForPredictedImage = np.dtype(np.float32)
                suffixToAdd = "_ProbMapClass" + str(class_i)
                #Save the image. Pass the filename paths of the normal image so that I can dublicate the header info, eg RAS transformation.
                predictedLabelImageForSpecificClass = labelImageCreatedByPredictions[class_i,:,:,:]
                unpaddedPredictedLabelImageForSpecificClass = predictedLabelImageForSpecificClass if not padInputImagesBool else unpadCnnOutputs(predictedLabelImageForSpecificClass, tupleOfPaddingPerAxesLeftRight)
                savePredictedImageToANewNiiWithHeaderFromOther(unpaddedPredictedLabelImageForSpecificClass,
                                        listOfNamesToGiveToPredictionsIfSavingResults,
                                        
                                        listOfFilepathsToEachChannelOfEachPatient,
                                        
                                        image_i,
                                        suffixToAdd,
                                        npDtypeForPredictedImage,
                                        myLogger
                                        )
        #=================Save FEATURE MAPS ====================
        if saveIndividualFmImagesForVisualisation :
            currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray = 0
            for pathwayType_i in xrange(len(cnn3dInstance.typesOfCnnLayers)) :
                indicesOfFmsToVisualisePerLayerOfCertainPathway = indicesOfFmsToVisualisePerPathwayTypeAndPerLayer[pathwayType_i]
                if indicesOfFmsToVisualisePerLayerOfCertainPathway<>[] :
                    for layer_i in xrange(len(cnn3dInstance.typesOfCnnLayers[pathwayType_i])) :
                        indicesOfFmsToVisualiseForCertainLayerOfCertainPathway = indicesOfFmsToVisualisePerLayerOfCertainPathway[layer_i]
                        if indicesOfFmsToVisualiseForCertainLayerOfCertainPathway<>[] :
                            #If the user specifies to grab more feature maps than exist (eg 9999), correct it, replacing it with the number of FMs in the layer.
                            for fmActualNumber in xrange(indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[0], indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1]) :
                                fmToSave = multidimensionalImageWithAllToBeVisualisedFmsArray[currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray]
                                unpaddedFmToSave = fmToSave if not padInputImagesBool else unpadCnnOutputs(fmToSave, tupleOfPaddingPerAxesLeftRight)
                                saveFmActivationImageToANewNiiWithHeaderFromOther(  unpaddedFmToSave,
                                                                                    listOfNamesToGiveToFmVisualisationsIfSaving,
                                                                                    listOfFilepathsToEachChannelOfEachPatient,
                                                                                    image_i,
                                                                                    pathwayType_i,
                                                                                    layer_i,
                                                                                    fmActualNumber,
                                                                                    myLogger
                                                                                    ) 
                                currentIndexInTheMultidimensionalImageWithAllToBeVisualisedFmsArray += 1
        if saveMultidimensionalImageWithAllFms :
            """
            multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms =  np.zeros(niiDimensions + [totalNumberOfFMsToProcess], dtype = "float32")
            for fm_i in xrange(0, totalNumberOfFMsToProcess) :
                multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms[:,:,:,fm_i] = multidimensionalImageWithAllToBeVisualisedFmsArray[fm_i]
            """
            multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms =  np.transpose(multidimensionalImageWithAllToBeVisualisedFmsArray, (1,2,3, 0) )
            
            unpaddedMultidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms = multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms if not padInputImagesBool else \
                unpadCnnOutputs(multidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms, tupleOfPaddingPerAxesLeftRight)
                
            #Save a multidimensional Nii image. 3D Image, with the 4th dimension being all the Fms...
            saveMultidimensionalImageWithAllVisualisedFmsToANewNiiWithHeaderFromOther(  unpaddedMultidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms,
                                                                                        listOfNamesToGiveToFmVisualisationsIfSaving,
                                                                                        listOfFilepathsToEachChannelOfEachPatient,
                                                                                        image_i,
                                                                                        myLogger)
        #=================IMAGES SAVED. PROBABILITY MAPS AND FEATURE MAPS TOO (if wanted). ====================
        
        #=================EVALUATE DSC FROM THE PROBABILITY MAPS FOR EACH IMAGE. ====================
        if providedGtLabelsBool : # Ground Truth was provided for calculation of DSC. Do DSC calculation.
            myLogger.print3("+++++++++++++++++++++ Reporting Segmentation Metrics for the subject #" + str(image_i) + " ++++++++++++++++++++++++++")
            #Unpad all segmentation map, gt, brainmask
            unpaddedSegmentationImage = segmentationImage if not padInputImagesBool else unpadCnnOutputs(segmentationImage, tupleOfPaddingPerAxesLeftRight)
            unpaddedGtLabelsImage = gtLabelsImage if not padInputImagesBool else unpadCnnOutputs(gtLabelsImage, tupleOfPaddingPerAxesLeftRight)
            #Hack, for it to work for the case that I do not use a brainMask.
            if isinstance(brainMask, (np.ndarray)) : #If brainmask was given:
                multiplyWithBrainMaskOr1 = brainMask if not padInputImagesBool else unpadCnnOutputs(brainMask, tupleOfPaddingPerAxesLeftRight)
            else :
                multiplyWithBrainMaskOr1 = 1
            #calculate DSC per class.
            for class_i in xrange(0, NUMBER_OF_CLASSES) :
                if class_i == 0 : #in this case, do the evaluation for the WHOLE segmentation (not background)
                    booleanPredictedLabelImage = unpaddedSegmentationImage>0 #Whatever is not background
                    booleanGtLesionLabelsForDiceEvaluation_unstripped = unpaddedGtLabelsImage>0
                else :
                    booleanPredictedLabelImage = unpaddedSegmentationImage==class_i
                    booleanGtLesionLabelsForDiceEvaluation_unstripped = unpaddedGtLabelsImage==class_i
                    
                predictedLabelImageConvolvedWithBrainMask = booleanPredictedLabelImage*multiplyWithBrainMaskOr1
                
                #Calculate the 3 Dices. Dice1 = Allpredicted/allLesions, Dice2 = PredictedWithinBrainMask / AllLesions , Dice3 = PredictedWithinBrainMask / LesionsInsideBrainMask.
                #Dice1 = Allpredicted/allLesions
                diceCoeff1 = calculateDiceCoefficient(booleanPredictedLabelImage, booleanGtLesionLabelsForDiceEvaluation_unstripped)
                diceCoeffs1[image_i][class_i] = diceCoeff1 if diceCoeff1 <> -1 else NA_PATTERN
                #Dice2 = PredictedWithinBrainMask / AllLesions
                diceCoeff2 = calculateDiceCoefficient(predictedLabelImageConvolvedWithBrainMask, booleanGtLesionLabelsForDiceEvaluation_unstripped)
                diceCoeffs2[image_i][class_i] = diceCoeff2 if diceCoeff2 <> -1 else NA_PATTERN
                #Dice3 = PredictedWithinBrainMask / LesionsInsideBrainMask
                diceCoeff3 = calculateDiceCoefficient(predictedLabelImageConvolvedWithBrainMask, booleanGtLesionLabelsForDiceEvaluation_unstripped * multiplyWithBrainMaskOr1)
                diceCoeffs3[image_i][class_i] = diceCoeff3 if diceCoeff3 <> -1 else NA_PATTERN
                
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
    
def printExplanationsAboutDice(myLogger) :
    myLogger.print3("EXPLANATION: DICE1/2/3 are lists with the DICE per class. For Class-0 we calculate DICE for the whole foreground (useful for multi-class problems).")
    myLogger.print3("EXPLANATION: DICE1 is calculated whole segmentation vs whole Ground Truth (GT). DICE2 is the segmentation within the ROI vs GT. DICE3 is segmentation within the ROI vs the GT within the ROI.")
    
    