# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import sys
import time
import datetime
import numpy as np
import nibabel as nib
import random
import math

from scipy.ndimage.filters import gaussian_filter

import pp

from deepmedic.cnnHelpers import dump_cnn_to_gzip_file_dotSave
from deepmedic.genericHelpers import *

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
 	#tupleOfPaddingPerAxesLeftRight : ( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). "placeholderNothing" when no padding.

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

                                                                 listOfFilepathsToEachChannelOfEachPatient, #NEW

                                                                 providedGtLabelsBool, #NEW
                                                                 listOfFilepathsToGtLabelsOfEachPatient, #NEW

                                                                 providedMaskWhereToGetPositiveSamples, #NEW
                                                                 listOfFilepathsToMasksOfEachPatientForPosSamplingForTrainOrVal,#NEW

                                                                 providedRoiMaskBool, #NEW
                                                                 listOfFilepathsToRoiMaskOfEachPatient, #NEW

                                                                 providedMaskWhereToGetNegativeSamples, #NEW
                                                                 listOfFilepathsToMasksOfEachPatientForNegSamplingForTrainOrVal,#NEW

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
    
    myLogger.print3("Loading subject with 1st channel at:"+str(listOfFilepathsToEachChannelOfEachPatient[index_of_wanted_image][0]))
    
    numberOfNormalScaleChannels = len(listOfFilepathsToEachChannelOfEachPatient[0])

    #reflect Image with 50% prob, for each axis:
    reflectFlags = []
    for reflectImageWithHalfProb_dimi in xrange(0, len(reflectImageWithHalfProb)) :
    	reflectFlags.append(reflectImageWithHalfProb[reflectImageWithHalfProb_dimi] * random.randint(0,1))
    
    tupleOfPaddingPerAxesLeftRight = "placeholderNothing" #This will be given a proper value if padding is performed.

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

	if allChannelsOfPatientInNpArray == None :
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
			# >>>>>>ROI MASK MIGHT BE A PLACEHOLDER!<<<<<<<<
			if roiMask <> "placeholderNothing":
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
        gtLabelsData = imgGtLabels_proxy.get_data()
        gtLabelsData = reflectImageArrayIfNeeded(reflectFlags, gtLabelsData) #reflect if flag ==1 .
        imageGtLabels = gtLabelsData
        imgGtLabels_proxy.uncache()
	[imageGtLabels, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(imageGtLabels, cnnReceptiveField, imagePartDimensions) if padInputImagesBool else [imageGtLabels, tupleOfPaddingPerAxesLeftRight]
    else : 
        imageGtLabels = "placeholderNothing" #For validation and testing


    if providedMaskWhereToGetPositiveSamples :
        fullFilenamePathOfMaskWhereToGetPositiveSamples = listOfFilepathsToMasksOfEachPatientForPosSamplingForTrainOrVal[index_of_wanted_image]
        img_proxy = nib.load(fullFilenamePathOfMaskWhereToGetPositiveSamples)
	maskWhereToGetPositiveSamplesData = img_proxy.get_data()
	maskWhereToGetPositiveSamplesData = reflectImageArrayIfNeeded(reflectFlags, maskWhereToGetPositiveSamplesData)
        maskWhereToGetPositiveSamples = maskWhereToGetPositiveSamplesData #np.asarray(maskWhereToGetPositiveSamplesData, dtype="float32") #the .get_data returns a nparray but it is not a float64
        img_proxy.uncache()
	[maskWhereToGetPositiveSamples, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(maskWhereToGetPositiveSamples, cnnReceptiveField, imagePartDimensions) if padInputImagesBool else [maskWhereToGetPositiveSamples, tupleOfPaddingPerAxesLeftRight]
    else :
        maskWhereToGetPositiveSamples = "placeholderNothing"

    if providedMaskWhereToGetNegativeSamples :
        fullFilenamePathOfMaskWhereToGetNegativeSamples = listOfFilepathsToMasksOfEachPatientForNegSamplingForTrainOrVal[index_of_wanted_image]
        img_proxy = nib.load(fullFilenamePathOfMaskWhereToGetNegativeSamples)
	maskWhereToGetNegativeSamplesData = img_proxy.get_data()
	maskWhereToGetNegativeSamplesData = reflectImageArrayIfNeeded(reflectFlags, maskWhereToGetNegativeSamplesData)
        maskWhereToGetNegativeSamples = maskWhereToGetNegativeSamplesData #np.asarray(maskWhereToGetNegativeSamplesData, dtype="float32") #the .get_data returns a nparray but it is not a float64
        img_proxy.uncache()
	[maskWhereToGetNegativeSamples, tupleOfPaddingPerAxesLeftRight] = padCnnInputs(maskWhereToGetNegativeSamples, cnnReceptiveField, imagePartDimensions) if padInputImagesBool else [maskWhereToGetNegativeSamples, tupleOfPaddingPerAxesLeftRight]
    else :
        maskWhereToGetNegativeSamples = "placeholderNothing"
        
        
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
	
    return [allChannelsOfPatientInNpArray, imageGtLabels, maskWhereToGetPositiveSamples, roiMask, maskWhereToGetNegativeSamples, allSubsampledChannelsOfPatientInNpArray, tupleOfPaddingPerAxesLeftRight]


#made for 3d
def get_positive_and_negative_imageParts(myLogger,
                                         numberOfPositiveSamples_forThisImage,
                                         numberOfNegativeSamples_forThisImage,
                                         imagePartDimensions,
                                         channelsOfImageNpArray,#chans,niiDims
                                         maskWhereToGetPositiveSamples,
                                         maskWhereToGetNegativeSamples,
                                         theMasksWhereToGetAreProbabilityMaps,
                                         get_randomly_only_one_of_the_two = 'False'):
    #This function, IF get_randomly_only_one_of_the_two==False,
    #...it will EITHER return NO PAIR OR the same number of Lesions And Brains (complete pairs)
    
    """
	This function returns the coordinate (index) of the "central" voxel of an image part (1v to the left if even part-dimension). It ALSO returns the indeces of the image part, left and right indices, INCLUSIVE BOTH SIDES.
    """
    
    HUGE_INTEGER = 9999999 #intensities should never exceed this. Used in get_positive_and_negative_imageParts()

    channelImageDimensions = channelsOfImageNpArray[0].shape

    posNegParts = []

    if get_randomly_only_one_of_the_two == 'False' :
        iterationsForPositiveAndNegative = [0,1] #iterate for 0=positive and 1=negative
    else : #batch_size = 1 and the function was called to give back only one, either positive (lesion) or negative (healthy) image-part
        iterationsForPositiveAndNegative = [random.randint(0,1)]
    
    for pos_or_neg in iterationsForPositiveAndNegative : # 0=pos, 1=neg image_part
        mask_to_check = ""
        if pos_or_neg == 0 : #iteration for positive image part. Check for lesion.
            #If I do not have a certain mask to sample from, the previous function will return "placeholderNothing". In this case, sample from anywhere.
            mask_to_check = maskWhereToGetPositiveSamples if maskWhereToGetPositiveSamples <> "placeholderNothing" else channelsOfImageNpArray[0]<HUGE_INTEGER
            if (theMasksWhereToGetAreProbabilityMaps==False) and (maskWhereToGetPositiveSamples <> "placeholderNothing") :
		#In this case I am probably loading the label-map. So just make it binary, for extraction, in case it was multiclass. I ll have to change this for multiclass.
		mask_to_check = mask_to_check > 0
            numberPosOrNegSamplesLookingForInThisImage = numberOfPositiveSamples_forThisImage
        else : #iteration for negative
            #Make sure that the negatives are outside of the positive-mask. This makes sure that if I load the brainmask for negatives, I dont get lesions...
            mask_to_check = maskWhereToGetNegativeSamples if maskWhereToGetNegativeSamples <> "placeholderNothing" else channelsOfImageNpArray[0]<HUGE_INTEGER
            if (theMasksWhereToGetAreProbabilityMaps==False) and (maskWhereToGetPositiveSamples <> "placeholderNothing") :
    		#Make sure that the negatives are outside of the positive-mask. This makes sure that if I load the brainmask for negatives, I dont get lesions...
    		#When using 20 images, this adds half a second to the loading time (of all images, not each). So it is fast.
		mask_to_check[maskWhereToGetPositiveSamples>0] = 0
            elif (theMasksWhereToGetAreProbabilityMaps==False) and (maskWhereToGetPositiveSamples == "placeholderNothing") :
		#This is supposed to be "uniform sampling".
		myLogger.print3("WARN: No mask was provided for the foreground (Ground Truth Labels). In this case, Background-Centered segments are actually extracted from the whole ROI in a UNIFORM manner (thus extracting Foreground-Centered too). This is a hacked implementation, which will be re-written cleaner soon.")
            numberPosOrNegSamplesLookingForInThisImage = numberOfNegativeSamples_forThisImage

        #Choose a random POSITIVE example:

        if np.sum(mask_to_check>0) == 0 : #[1] and [2] also have the same length, so no prob.
            myLogger.print3("WARN: NO GT-Labels(0) OR ROI-MASK(1) FOR THIS! no pair found! Iteration was for:"+str(pos_or_neg))
            return (False, posNegParts) #return found_pair==false



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
        booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries = np.zeros(mask_to_check.shape, dtype="int32")
        
	#The following loop leads to booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries to be true for the indices that allow you to get an image part CENTERED on them, and be safely within image boundaries. Note that if the imagePart is of even dimension, the "central" voxel is one voxel to the left.
        for rcz_i in xrange( len(imagePartDimensions) ) :
            if imagePartDimensions[rcz_i]%2 == 0: #even
                dimensionDividedByTwo = imagePartDimensions[rcz_i]/2
                halfImagePartBoundaries[rcz_i] = [dimensionDividedByTwo - 1, dimensionDividedByTwo] #central of ImagePart is 1 vox closer to begining of axes.
            else: #odd
                dimensionDividedByTwoFloor = math.floor(imagePartDimensions[rcz_i]/2) #eg 5/2 = 2, with the 3rd voxel being the "central"
                halfImagePartBoundaries[rcz_i] = [dimensionDividedByTwoFloor, dimensionDividedByTwoFloor] 
        #used to be [halfImagePartBoundaries[0][0]: -halfImagePartBoundaries[0][1]], but in 2D case halfImagePartBoundaries might be ==0, causes problem and you get a null slice.
	booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries[halfImagePartBoundaries[0][0]: channelImageDimensions[0] - halfImagePartBoundaries[0][1],
						halfImagePartBoundaries[1][0]: channelImageDimensions[1] - halfImagePartBoundaries[1][1],
						halfImagePartBoundaries[2][0]: channelImageDimensions[2] - halfImagePartBoundaries[2][1]] = 1

	constrainedWithImageBoundariesMaskToSample = mask_to_check * booleanNpArray_voxelsToCentraliseImPartsWithinBoundaries
	#normalize the probabilities to sum to 1, cause the function needs it as so.
	constrainedWithImageBoundariesMaskToSample = constrainedWithImageBoundariesMaskToSample / (1.0* np.sum(constrainedWithImageBoundariesMaskToSample))
        
	flattenedConstrainedWithImageBoundariesMaskToSample = constrainedWithImageBoundariesMaskToSample.flatten()

        #slice_coordinates : a list with as many elements as the number of pairs taken from this image part...
        #Each of them is : [ [rlow,rhigh], [clow,chigh], [zlow,zhigh] ]. Both inclusive. So do zhigh+1 when getting the slice! 
        allPosOrNegImagePartsEntries = []
        
	#This is going to be a 3xNumberOfImagePartSamples array.
	indicesInTheFlattenArrayThatWereSampledAsCentralVoxelsOfImageParts = np.random.choice(constrainedWithImageBoundariesMaskToSample.size,
												size=numberPosOrNegSamplesLookingForInThisImage,
												replace=True,
												p=flattenedConstrainedWithImageBoundariesMaskToSample)
	#np.unravel_index([listOfIndicesInFlattened], dims) returns a tuple of arrays (eg 3 of them if 3 dimImage), where each of the array in the tuple has the same shape as the listOfIndices. They have the r/c/z coords that correspond to the index of the flattened version.
	#So, coordsOfCentralVoxelsOfPartsSampled will end up being an array with shape: 3(rcz)xnumberPosOrNegSamplesLookingForInThisImage.
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
        posNegParts.append([coordsOfCentralVoxelsOfPartsSampled, sliceCoordsOfImagePartsSampled]) #(allPosOrNegImagePartsEntries)
    
    """
	OLD!!!!!!!
    posNegParts: List of 2 slots. First is positive, second is negative if pair is acquired. Only exception...
        ...is when batch_size is 1, and then this function is called with argument: get_randomly_only_one_of_the_two=='True'.
        ...In this case, randomly only the neg OR pos is returned. In this case, the returned is placed in first slot of posNegParts.
        
        Now, in each slot, the structure is the following: a List [coordinates_rcz, slice_coordinates].
        coordinates_rcz : a list [r,c,z] of the coords of the central pixel of the ImagePart (in the whole image)
        #slice_coordinates : a list with as many elements as the number of pairs taken from this image part...
        #...Each of them is : [ [rlow,rhigh], [clow,chigh], [zlow,zhigh] ]. Both inclusive. So do zhigh+1 when getting the slice! 
    """
    
    myLogger.print3("End of the function that extracts segments. Segments extracted centered on: Foreground="+str(len(posNegParts[0][0][0])) + " Background=" + str(len(posNegParts[1][0][0])))
    
    return posNegParts






        
        
def getImagePartFromSubsampledImageForTraining(imagePartDimensions,
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
    
    subsampledChannelsForThisImagePart = np.ones((len(subsampledImageChannels), 
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
    #1
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


    dimensionsOfTheSliceOfSubsampledImageNotPadded = [int(math.ceil((rhighNonInclCorrected - rlowCorrected)*1.0/subSamplingFactor[0])),
						int(math.ceil((chighNonInclCorrected - clowCorrected)*1.0/subSamplingFactor[1])),
						int(math.ceil((zhighNonInclCorrected - zlowCorrected)*1.0/subSamplingFactor[2]))
						]

    #I now have exactly where to get the slice from and where to put it in the new array.

    for channel_i in xrange(len(subsampledImageChannels)) :
        intensityZeroOfChannel = calculateTheZeroIntensityOf3dImage(subsampledImageChannels[channel_i])        
        subsampledChannelsForThisImagePart[channel_i] *= intensityZeroOfChannel
        
        sliceOfSubsampledImageNotPadded = subsampledImageChannels[channel_i][rlowCorrected : rhighNonInclCorrected : subSamplingFactor[0],
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
                


    
   
    
def getImagePartsAndTheirSlices(myLogger,
				strideOfImagePartsPerDimensionInVoxels,
                                imagePartDimensions,
                                batch_size,
                                channelsOfImageNpArray,#chans,niiDims
                                brainMask,
                                #New for the extension of the cnn.
                                channelsOfSubsampledImageNpArray, #chans,niiDims
                                patchDimensions,
                                subsampledImageChannels,
                                subSamplingFactor,
                                subsampledImagePartDimensions,
                                ) :
    myLogger.print3("Starting to (tile) extract Segments from the images of the subject for Segmentation...")

    channelsForPartsToReturn = []
    coordsOfTopLeftVoxelForPartsToReturn = []
    channelsForSubsampledPartsToReturn = []
    
    niiDimensions = list(channelsOfImageNpArray[0].shape)

    zLowBoundaryNext=0; zAxisCentralPartPredicted = False;
    while not zAxisCentralPartPredicted :
	zFarBoundary = min(zLowBoundaryNext+imagePartDimensions[2], niiDimensions[2]) #Excluding.
	zLowBoundary = zFarBoundary - imagePartDimensions[2]
	zLowBoundaryNext = zLowBoundaryNext + strideOfImagePartsPerDimensionInVoxels[2]
	zAxisCentralPartPredicted = False if zFarBoundary < niiDimensions[2] else True

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

		thisImagePartChannels = []

		if brainMask<>"placeholderNothing" : #In case I pass a brain-mask, I ll use it to only predict inside it. Otherwise, whole image.
                    brainMaskForThisImagePart = brainMask[rLowBoundary:rFarBoundary,
                                                          cLowBoundary:cFarBoundary,
                                                          zLowBoundary:zFarBoundary
                                                          ]
                    if not np.any(brainMaskForThisImagePart) : #all of it is out of the brain so skip it.
                        continue
                    
                coordsOfTopLeftVoxelForPartsToReturn.append([rLowBoundary,cLowBoundary,zLowBoundary])
                
                for channel_i in xrange(len(channelsOfImageNpArray)) :
                    thisImagePartChannels.append(channelsOfImageNpArray[channel_i,
                                                                        rLowBoundary:rFarBoundary,
                                                                        cLowBoundary:cFarBoundary,
                                                                        zLowBoundary:zFarBoundary
                                                                        ])
                channelsForPartsToReturn.append(thisImagePartChannels)
                
                
                #NEW ADDITION FOR EXTENSION OF CNN
		if channelsOfSubsampledImageNpArray <> "placeholderNothing" :
                    imagePartSlicesCoords = [ [rLowBoundary, rFarBoundary-1], [cLowBoundary, cFarBoundary-1], [zLowBoundary, zFarBoundary-1] ] #the right hand values are placeholders in this case.
                    channelsForThisSubsampledImagePart = getImagePartFromSubsampledImageForTraining(
                                                                                                   imagePartDimensions,
                                                                                                   patchDimensions,
                                                                                                   subsampledImageChannels,
                                                                                                   imagePartSlicesCoords,
                                                                                                   subSamplingFactor,
                                                                                                   subsampledImagePartDimensions
                                                                                                   )
                    channelsForSubsampledPartsToReturn.append(channelsForThisSubsampledImagePart)

                
    #I need to have a total number of image-parts that can be exactly-divided by the 'batch_size'. For this reason, I add in the far end of the list multiple copies of the last element. I NEED THIS IN THEANO. I TRIED WITHOUT. NO.
    total_number_of_image_parts = len(channelsForPartsToReturn)
    number_of_imageParts_missing_for_exact_division =  batch_size - total_number_of_image_parts%batch_size if total_number_of_image_parts%batch_size <> 0 else 0
    for extra_useless_image_part_i in xrange(number_of_imageParts_missing_for_exact_division) :
        channelsForPartsToReturn.append(channelsForPartsToReturn[total_number_of_image_parts-1])
        coordsOfTopLeftVoxelForPartsToReturn.append(coordsOfTopLeftVoxelForPartsToReturn[total_number_of_image_parts-1])
	if channelsOfSubsampledImageNpArray <> "placeholderNothing" :
            channelsForSubsampledPartsToReturn.append(channelsForSubsampledPartsToReturn[total_number_of_image_parts-1])

    #I think that since the parts are acquired in a certain order and are sorted this way in the list, it is easy
    #to know which part of the image they came from, as it depends only on the stride-size and the imagePart size.
    
    myLogger.print3("Finished (tiling) extracting Segments from the images of the subject for Segmentation.")

    return [channelsForPartsToReturn, coordsOfTopLeftVoxelForPartsToReturn, channelsForSubsampledPartsToReturn]
                




#-----------The function that is executed in parallel with gpu training:----------------
def getTheArraysOfImageChannelsAndLesionsToLoadToGpuForSubepoch(myLogger,
								training0orValidation1,

                                                                n_images_per_subepoch,
                                                                numberOfPositiveSamplesPerSubepoch,
                                                                numberOfNegativeSamplesPerSubepoch,
                                                                usingSubsampledWaypath,

								listOfFilepathsToEachChannelOfEachPatient, #NEW

								listOfFilepathsToGtLabelsOfEachPatientTrainOrVal,#NEW

								providedRoiMaskBool, #NEW
								listOfFilepathsToRoiMaskOfEachPatient, #NEW

								providedMaskWhereToGetPositiveSamples, #NEW
                                                                listOfFilepathsToMasksOfEachPatientForPosSamplingForTrainOrVal, #NEW
								providedMaskWhereToGetNegativeSamples, #NEW
								listOfFilepathsToMasksOfEachPatientForNegSamplingForTrainOrVal, #NEW
								theMasksWhereToGetAreProbabilityMaps,
								useSameSubChannelsAsSingleScale,

								listOfFilepathsToEachSubsampledChannelOfEachPatient, #NEW

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


    total_number_of_images = len(listOfFilepathsToEachChannelOfEachPatient)
    randomIndicesList_for_gpu = get_random_image_indices_to_load_on_GPU(total_number_of_images = total_number_of_images,
                                                                    max_images_on_gpu_for_subepoch = n_images_per_subepoch,
                                                                    get_max_images_for_gpu_even_if_total_less = False,
                                                                    myLogger=myLogger)
    myLogger.print3("Out of the " + str(total_number_of_images) + " subjects given for " + trainingOrValidationString + ", it was specified to extract Segments from " + str(n_images_per_subepoch) + " per subepoch.")
    myLogger.print3("Shuffled indices of subjects that were randomly chosen: "+str(randomIndicesList_for_gpu))
    
    imagePartsChannelsToLoadOnGpuForSubepoch = [] #This is x. Will end up with dimensions: partImagesLoadedPerSubepoch, channels, r,c,z, but flattened.
    lesionLabelsForTheImagePartsInGpUForSubepoch = [] #These are the FULL LABEL SLICES for the image parts. I ll have to reformat to get the ones for the patches I think.
    subsampledImagePartsChannelsToLoadOnGpuForSubepoch = []
            
    n_images_loading_this_subepoch = len(randomIndicesList_for_gpu) #Can be different than n_images_per_subepoch, cause of available images number.
    #Load this images into a sharedVariable.set_value(). In the memory I want the IMAGES, not the parts.
    #In each batch I ll be changing parts, but the whole images should be loaded. I dont want to be changing that!
            
    #here normally I would just .setValue(). No theano weird functions needed.
    #sharedVariableForGpuImagesOfSubepoch = np.asarray(subepochsImagesLoadedInCpuMemory, dtype='float32')
    #Do I also need the lesions and brains on shared? Probably I do not. Calculate on CPU which image parts to get...
    #...calculate the slices/view of the shared variable that you want and pass it to the givens.
            
            
    numberOfPositiveSamplesPerSubepochPerImage = numberOfPositiveSamplesPerSubepoch / n_images_loading_this_subepoch
    numberOfPositiveSamplesPerSubepochLeftUnevenly = numberOfPositiveSamplesPerSubepoch % n_images_loading_this_subepoch
    numberOfNegativeSamplesPerSubepochPerImage = numberOfNegativeSamplesPerSubepoch / n_images_loading_this_subepoch
    numberOfNegativeSamplesPerSubepochLeftUnevenly = numberOfNegativeSamplesPerSubepoch % n_images_loading_this_subepoch
            
    myLogger.print3("SAMPLING: Starting iterations to extract Segments from each image for next " + trainingOrValidationString + "...")
    #get number_of_images_to_get_the_pairs_for_the_batch_from pairs from that many random images.
            
            
    numberOfPositiveSamplesFound = 0
    numberOfNegativeSamplesFound = 0
            
    array_numberOfPositiveSamplesFromEachImageToGet = np.zeros(n_images_loading_this_subepoch, dtype='int32') + numberOfPositiveSamplesPerSubepochPerImage
    array_numberOfNegativeSamplesFromEachImageToGet = np.zeros(n_images_loading_this_subepoch, dtype='int32') + numberOfNegativeSamplesPerSubepochPerImage
            
    for i_uneven_positiveSample in xrange(numberOfPositiveSamplesPerSubepochLeftUnevenly):
        array_numberOfPositiveSamplesFromEachImageToGet[i_uneven_positiveSample]+=1
    for i_uneven_negativeSample in xrange(numberOfNegativeSamplesPerSubepochLeftUnevenly):
        array_numberOfNegativeSamplesFromEachImageToGet[i_uneven_negativeSample]+=1
    index_next_image_with_less = numberOfPositiveSamplesPerSubepochLeftUnevenly
            

    numberOfNormalScaleChannels = len(listOfFilepathsToEachChannelOfEachPatient[0])

    for index_for_vector_with_images_on_gpu in xrange(0, n_images_loading_this_subepoch) :
	myLogger.print3("SAMPLING: Going to load the images and extract segments from the subject (iteration) #" + str(index_for_vector_with_images_on_gpu) + "/" +str(n_images_loading_this_subepoch))
        #THE number of imageParts in memory per subepoch I think does not need to be constant. The batch_size needs.
        #But I could have less batches per subepoch if some images dont have lesions I guess. Anyway.
        #THEY WAY I DO IT, with taking a pair from each image, the batch_size = 2(pair)*images_in_gpu
        numberOfPositiveSamples_forThisImage = array_numberOfPositiveSamplesFromEachImageToGet[index_for_vector_with_images_on_gpu]
        numberOfNegativeSamples_forThisImage = array_numberOfNegativeSamplesFromEachImageToGet[index_for_vector_with_images_on_gpu]
                
        [allChannelsOfPatientInNpArray, #a nparray(channels,dim0,dim1,dim2)
        gtLabelsImage,
        maskWhereToGetPositiveSamples, #can be returned "placeholderNothing". In this case, I will later grab from whole image.
        roiMask,
        maskWhereToGetNegativeSamples, #can be returned "placeholderNothing". In this case, I will later grab from whole image.
        allSubsampledChannelsOfPatientInNpArray,  #a nparray(channels,dim0,dim1,dim2)
	tupleOfPaddingPerAxesLeftRight #( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). "placeholderNothing" when no padding.
        ] = actual_load_patient_images_from_filepath_and_return_nparrays(
                                                myLogger,
						training0orValidation1,

                                                randomIndicesList_for_gpu[index_for_vector_with_images_on_gpu],

						listOfFilepathsToEachChannelOfEachPatient, #NEW

                                                providedGtLabelsBool=True, #NEW. If this getTheArr function is called, gtLabels should already been provided.
                                                listOfFilepathsToGtLabelsOfEachPatient=listOfFilepathsToGtLabelsOfEachPatientTrainOrVal, #NEW

                                                providedMaskWhereToGetPositiveSamples = providedMaskWhereToGetPositiveSamples, #NEW
                                                listOfFilepathsToMasksOfEachPatientForPosSamplingForTrainOrVal=listOfFilepathsToMasksOfEachPatientForPosSamplingForTrainOrVal, #can be given "no" and will return placeholder

                                                providedRoiMaskBool= providedRoiMaskBool if training0orValidation1 == 0 else False, #NEW
                                                listOfFilepathsToRoiMaskOfEachPatient = listOfFilepathsToRoiMaskOfEachPatient if training0orValidation1 == 0 else "placeholder", #NEW

                                                providedMaskWhereToGetNegativeSamples = providedMaskWhereToGetNegativeSamples, #NEW
                                                listOfFilepathsToMasksOfEachPatientForNegSamplingForTrainOrVal=listOfFilepathsToMasksOfEachPatientForNegSamplingForTrainOrVal, #can be given "no" and will return placeholder
						useSameSubChannelsAsSingleScale=useSameSubChannelsAsSingleScale,

                                                usingSubsampledWaypath=usingSubsampledWaypath,
						listOfFilepathsToEachSubsampledChannelOfEachPatient=listOfFilepathsToEachSubsampledChannelOfEachPatient,

						padInputImagesBool=padInputImagesBool,
						cnnReceptiveField=patchDimensions, # only used if padInputsBool
						imagePartDimensions=imagePartDimensions, # only used if padInputsBool

						smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage = smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
						normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc=\
							normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
						reflectImageWithHalfProb = reflectImageWithHalfProbDuringTraining
                                                )
	myLogger.print3("DEBUG: Original index of this case in list of subjects: " + str(randomIndicesList_for_gpu[index_for_vector_with_images_on_gpu]))
        myLogger.print3("Images for subject loaded.")


        myLogger.print3("From current subject #"+str(index_for_vector_with_images_on_gpu)+" extracting that many Segments centered on : Foreground=" + str(numberOfPositiveSamples_forThisImage) + " Background:" + str(numberOfNegativeSamples_forThisImage) )
        posNegParts = get_positive_and_negative_imageParts(myLogger = myLogger,
                                                        numberOfPositiveSamples_forThisImage = numberOfPositiveSamples_forThisImage,
                                                        numberOfNegativeSamples_forThisImage = numberOfNegativeSamples_forThisImage,
                                                        imagePartDimensions = imagePartDimensions,
                                                        channelsOfImageNpArray = allChannelsOfPatientInNpArray,#chans,niiDims
                                                        maskWhereToGetPositiveSamples = maskWhereToGetPositiveSamples,
                                                        maskWhereToGetNegativeSamples = maskWhereToGetNegativeSamples,
							theMasksWhereToGetAreProbabilityMaps = theMasksWhereToGetAreProbabilityMaps,
                                                        get_randomly_only_one_of_the_two = 'False')
                                


	########################
	#For normalization-augmentation: Get channels' stds if needed:
	stdsOfTheChannsOfThisImage = np.ones(numberOfNormalScaleChannels, dtype="float32")
	if training0orValidation1 == 0 and normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[0] == 2 and\
		normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc[1] == 0 : #intensity-augm is to be done, but images are not normalized.
		if roiMask <> "placeholderNothing": # >>>>>>BRAIN MASK MIGHT BE A PLACEHOLDER!<<<<<<<<
			stdsOfTheChannsOfThisImage = np.std(allChannelsOfPatientInNpArray[:, roiMask>0], axis=(1,2,3)) #We'll use this for the downsampled version too.
		else : #no brain mask provided:
			stdsOfTheChannsOfThisImage = np.std(allChannelsOfPatientInNpArray, axis=(1,2,3))
	#######################
        #THIS FOR LOOP COULD PROBABLY BE DONE AS A MATRIX-OPERATION SOMEHOW. Actually, after thought probably not as matrix-op but easily parallelisable on many threads. And will save me ALOT of time when extracting. This is THE next optimization point for time efficiency..!
        for pos_neg_i in [0,1] :
            for image_part_i in xrange(len(posNegParts[pos_neg_i][0][0])) :
                channelsForThisImagePart = np.zeros((numberOfNormalScaleChannels, imagePartDimensions[0],imagePartDimensions[1],imagePartDimensions[2]), dtype = 'float32')
                #Inclusive both sides. That's the slice I should grab to get the imagePart, centrered around [0].
                image_part_slices_coords = posNegParts[pos_neg_i][1][:,image_part_i,:] #[0] is the central voxel coords.
                                                
		channelsForThisImagePart[:numberOfNormalScaleChannels] = allChannelsOfPatientInNpArray[:,
												image_part_slices_coords[0][0]:image_part_slices_coords[0][1] +1,
												image_part_slices_coords[1][0]:image_part_slices_coords[1][1] +1,
												image_part_slices_coords[2][0]:image_part_slices_coords[2][1] +1]

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



                lesionLabelsForThisImagePart = gtLabelsImage[image_part_slices_coords[0][0]:image_part_slices_coords[0][1] +1,
                                                                image_part_slices_coords[1][0]:image_part_slices_coords[1][1] +1,
                                                                image_part_slices_coords[2][0]:image_part_slices_coords[2][1] +1]
                imagePartsChannelsToLoadOnGpuForSubepoch.append(channelsForThisImagePart)
                lesionLabelsForTheImagePartsInGpUForSubepoch.append(lesionLabelsForThisImagePart)
                                                                    
                #FOR THE SUBSAMPLED IMAGE-PARTS:
                if usingSubsampledWaypath :
                    #this datastructure is similar to channelsForThisImagePart, but contains voxels from the subsampled image.
                    subsampledChannelsForThisImagePart = getImagePartFromSubsampledImageForTraining(imagePartDimensions,
                                                                                                        patchDimensions,
                                                                                                        allSubsampledChannelsOfPatientInNpArray,
                                                                                                        image_part_slices_coords,
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
                    subsampledImagePartsChannelsToLoadOnGpuForSubepoch.append(subsampledChannelsForThisImagePart)
                                                                                
                                                                                
                                                                                
    #FINISHED WITH CALCULATING Image Parts etc. Now lets just put them in the correct format and load them on gpu
    lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepoch = []
    #The patch should always be of odd dimensions, and this will work. The image PART can be of even too, it's fine.
    rPatchHalfWidth = (patchDimensions[0]-1)/2
    cPatchHalfWidth = (patchDimensions[1]-1)/2
    zPatchHalfWidth = (patchDimensions[2]-1)/2
    rUpperBoundToGrabLabelsOfCentralVoxels = imagePartDimensions[0]-rPatchHalfWidth
    cUpperBoundToGrabLabelsOfCentralVoxels = imagePartDimensions[1]-cPatchHalfWidth
    zUpperBoundToGrabLabelsOfCentralVoxels = imagePartDimensions[2]-zPatchHalfWidth

    #Get the part of the GT-segments that correspond to the central (predicted) part of the segments.
    for lesionLabelsForTheImagePart_i in xrange(len(lesionLabelsForTheImagePartsInGpUForSubepoch)) :
        lesLabelsForTheImagePart = lesionLabelsForTheImagePartsInGpUForSubepoch[lesionLabelsForTheImagePart_i]
	#used to be [rPatchHalfWidth : -rPatchHalfWidth], but in 2D case, where rPatchHalfWidth might be ==0, causes problem and you get a null slice.
        lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepoch.append(lesLabelsForTheImagePart[rPatchHalfWidth : rUpperBoundToGrabLabelsOfCentralVoxels,
                                                                                              cPatchHalfWidth : cUpperBoundToGrabLabelsOfCentralVoxels,
                                                                                              zPatchHalfWidth : zUpperBoundToGrabLabelsOfCentralVoxels])
    #I need to SHUFFLE THEM FIRST, together imageParts and lesionParts!
    if usingSubsampledWaypath :
        combined = zip(imagePartsChannelsToLoadOnGpuForSubepoch,
                       lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepoch,
                       subsampledImagePartsChannelsToLoadOnGpuForSubepoch)
        random.shuffle(combined)
        imagePartsChannelsToLoadOnGpuForSubepoch[:],\
            lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepoch[:],\
            subsampledImagePartsChannelsToLoadOnGpuForSubepoch[:] = zip(*combined)
    else :
        combined = zip(imagePartsChannelsToLoadOnGpuForSubepoch,
                       lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepoch)
        random.shuffle(combined)
        imagePartsChannelsToLoadOnGpuForSubepoch[:],\
            lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepoch[:] = zip(*combined)


    end_getAllImageParts_time = time.clock()
    myLogger.print3("TIMING: Extracting all the Segments for next " + trainingOrValidationString + " took time: "+str(end_getAllImageParts_time-start_getAllImageParts_time)+"(s)")

    myLogger.print3(":=:=:=:=:=:=:=:=: Finished extracting Segments from the images for next " + trainingOrValidationString + ". :=:=:=:=:=:=:=:=:")

    return [imagePartsChannelsToLoadOnGpuForSubepoch,lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepoch, subsampledImagePartsChannelsToLoadOnGpuForSubepoch]



#Called by do_training()
def reportAccuracyValuesOverWholeEpochForEachClass(myLogger,
						numberOfClasses,
						epoch,
						performValidationOnSamplesDuringTrainingProcessBool,
						arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringValidation,
						arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringTraining,
						) :

	if not performValidationOnSamplesDuringTrainingProcessBool:
		myLogger.print3( "WARN: Validation on samples was not performed during the training-process, because flag performValidationOnSamplesDuringTrainingProcessBool was set to False!!!" )

	for class_i in xrange(0, numberOfClasses) :

		classString = "(whole-foreground)Class-0" if class_i == 0 else "Class-"+str(class_i)

		myLogger.print3( ">>>>>>>>>>>> Reporting Accuracy over whole epoch for " + classString + " <<<<<<<<<<<<<" )

		for train0orValidation1 in [1,0] :
			if (train0orValidation1 == 1) and not performValidationOnSamplesDuringTrainingProcessBool:
				continue

			trainOrValCapitalString = "TRAINING" if train0orValidation1 == 0 else "VALIDATION"
			if train0orValidation1 == 0 :
				arrayThatHoldsAccuracyValuesForTrainOrVal = arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringTraining
			else :
				arrayThatHoldsAccuracyValuesForTrainOrVal = arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringValidation

			# [meanErrorOfSubepoch, meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch,
			#	meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, meanCostOfSubepoch].

			averageAccuracyOfAllSubepochsOfThisEpoch = arrayThatHoldsAccuracyValuesForTrainOrVal[class_i, 1, :]
			averageAccuracyPosPatchesOfAllSubepochsOfThisEpoch = arrayThatHoldsAccuracyValuesForTrainOrVal[class_i, 2, :]
			averageAccuracyNegPatchesOfAllSubepochsOfThisEpoch = arrayThatHoldsAccuracyValuesForTrainOrVal[class_i, 3, :]
			averageDiceOfAllSubepochsOfThisEpoch = arrayThatHoldsAccuracyValuesForTrainOrVal[class_i, 4, :]
			if train0orValidation1 == 0 :
				averageCostsOfAllSubepochsOfThisEpoch = arrayThatHoldsAccuracyValuesForTrainOrVal[class_i, 5, :]


			initialStringOfLine = trainOrValCapitalString + ": Epoch #" + str(epoch) + ", " + classString + ":"
			myLogger.print3(initialStringOfLine + " the mean accuracy of whole epoch was: "+str(np.mean(averageAccuracyOfAllSubepochsOfThisEpoch)))
			myLogger.print3(initialStringOfLine + " the mean accuracy of whole epoch, for Positive Samples(voxels) was: "+str(np.mean(averageAccuracyPosPatchesOfAllSubepochsOfThisEpoch)))
			myLogger.print3(initialStringOfLine + " the mean accuracy of whole epoch, for Negative Samples(voxels) was: "+str(np.mean(averageAccuracyNegPatchesOfAllSubepochsOfThisEpoch)))
			myLogger.print3(initialStringOfLine + " the mean Dice of whole epoch was: "+str(np.mean(averageDiceOfAllSubepochsOfThisEpoch)))
			if train0orValidation1 == 0 :
				myLogger.print3(initialStringOfLine + " the mean cost of whole epoch was: "+str(np.mean(averageCostsOfAllSubepochsOfThisEpoch)))

			#Visualised in my scripts:
			myLogger.print3(initialStringOfLine + " the mean accuracy of each subepoch was: "+str(averageAccuracyOfAllSubepochsOfThisEpoch))
			myLogger.print3(initialStringOfLine + " the mean accuracy of each subepoch, for Positive Samples(voxels) was: "+str(averageAccuracyPosPatchesOfAllSubepochsOfThisEpoch))
			myLogger.print3(initialStringOfLine + " the mean accuracy of each subepoch, for Negative Samples(voxels) was: "+str(averageAccuracyNegPatchesOfAllSubepochsOfThisEpoch))
			myLogger.print3(initialStringOfLine + " the mean Dice of each subepoch was: "+str(averageDiceOfAllSubepochsOfThisEpoch))
			if train0orValidation1 == 0 :
				myLogger.print3(initialStringOfLine + " the mean cost of each subepoch was: "+str(averageCostsOfAllSubepochsOfThisEpoch))
        	

	myLogger.print3( ">>>>>>>>>>>>>>>>>>>>>>>>> End Of Accuracy Report at the end of Epoch <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" )
	myLogger.print3( ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" )



#Called by doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch()
def calculateAndReportMeanAccuracyOverWholeSubepochForEachClass(myLogger,
								cnn3dInstance,
								train0orValidation1,
								subepoch,
								number_of_batches,
								errorsOfBatchesAdded,
								costsOfBatchesAdded,
								arrayWithNumbersOfPerClassRpRnTpTnInSubepoch
								) :

	trainOrValCapitalString = "TRAINING" if train0orValidation1 == 0 else "VALIDATION"

	#Calculate mean accuracy over subepoch for each class_i:

	#[ class-i: [meanErrorOfSubepoch, meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch, meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, meanCostOfSubepoch], ...]
	arrayWithReportedValuesPerClass = np.zeros([cnn3dInstance.numberOfOutputClasses, 6], dtype="float32")

	for class_i in xrange(0, cnn3dInstance.numberOfOutputClasses) :

		numberOfRealPositivesInSubepoch = arrayWithNumbersOfPerClassRpRnTpTnInSubepoch[class_i,0]
		numberOfRealNegativesInSubepoch = arrayWithNumbersOfPerClassRpRnTpTnInSubepoch[class_i,1]
		numberOfTruePositivesInSubepoch = arrayWithNumbersOfPerClassRpRnTpTnInSubepoch[class_i,2]
		numberOfTrueNegativesInSubepoch = arrayWithNumbersOfPerClassRpRnTpTnInSubepoch[class_i,3]

		meanErrorOfSubepoch = 999 if number_of_batches == 0 else errorsOfBatchesAdded/float(number_of_batches)
		meanAccuracyOfSubepoch = 1.0 - meanErrorOfSubepoch #For validation, where patches are 50/50 exactly (should be), this should be the mean acc of the pos and neg acc. For training, if parts used, this WILL NOT be the mean of Pos/Neg mean accuracy.
		meanAccuracyOnPositivesOfSubepoch = 999 if numberOfRealPositivesInSubepoch == 0 else numberOfTruePositivesInSubepoch*1.0/numberOfRealPositivesInSubepoch
		meanAccuracyOnNegativesOfSubepoch = 999 if numberOfRealNegativesInSubepoch == 0 else numberOfTrueNegativesInSubepoch*1.0/numberOfRealNegativesInSubepoch
		#New addition: Compute dice for the subepoch training/validation batches!
		numberOfPredictedPositivesInSubepoch = (numberOfRealNegativesInSubepoch + numberOfRealPositivesInSubepoch) - numberOfTrueNegativesInSubepoch - numberOfRealPositivesInSubepoch + numberOfTruePositivesInSubepoch
		meanDiceOfSubepoch = 999 if (numberOfPredictedPositivesInSubepoch + numberOfRealPositivesInSubepoch == 0) else (2.0*numberOfTruePositivesInSubepoch)/(numberOfPredictedPositivesInSubepoch + numberOfRealPositivesInSubepoch)
		# In case of validation, meanCostOfSubepoch is just a placeholder. Cause this does not get calculated and reported in this case.
		meanCostOfSubepoch = 999 if (train0orValidation1 == 1 or number_of_batches == 0) else costsOfBatchesAdded/float(number_of_batches)

		arrayWithReportedValuesPerClass[class_i] = [
			meanErrorOfSubepoch, meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch, meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, meanCostOfSubepoch]

	#Report mean accuracy over subepoch for each class_i:
	currentEpoch = cnn3dInstance.numberOfEpochsTrained
	for class_i in xrange(0, cnn3dInstance.numberOfOutputClasses) :
		classString = "(whole-foreground)Class-0" if class_i == 0 else "Class-"+str(class_i)
		myLogger.print3( "++++++++++++++++ Reporting Accuracy over whole subepoch for " + classString + " ++++++++++++++++" )

		[meanErrorOfSubepoch,
		meanAccuracyOfSubepoch,
		meanAccuracyOnPositivesOfSubepoch,
		meanAccuracyOnNegativesOfSubepoch,
		meanDiceOfSubepoch,
		meanCostOfSubepoch] = arrayWithReportedValuesPerClass[class_i]

		numberOfRealPositivesInSubepoch = arrayWithNumbersOfPerClassRpRnTpTnInSubepoch[class_i,0]
		numberOfRealNegativesInSubepoch = arrayWithNumbersOfPerClassRpRnTpTnInSubepoch[class_i,1]
		numberOfTruePositivesInSubepoch = arrayWithNumbersOfPerClassRpRnTpTnInSubepoch[class_i,2]
		numberOfTrueNegativesInSubepoch = arrayWithNumbersOfPerClassRpRnTpTnInSubepoch[class_i,3]
		
		initialStringOfLine = trainOrValCapitalString + ": Epoch #" + str(currentEpoch) + ", Subepoch #" + str(subepoch) + ", " + classString + ":"

		myLogger.print3(initialStringOfLine + " mean error: "+str(meanErrorOfSubepoch))
		myLogger.print3(initialStringOfLine + " mean accuracy "+str(meanAccuracyOfSubepoch))	
	 	myLogger.print3(initialStringOfLine + " mean accuracy for Positive Samples(voxels): " + str(meanAccuracyOnPositivesOfSubepoch) + ". True Predicted Pos="+str(numberOfTruePositivesInSubepoch)+"/"+str(numberOfRealPositivesInSubepoch))
		myLogger.print3(initialStringOfLine + " mean accuracy for Negative Samples(voxels): " + str(meanAccuracyOnNegativesOfSubepoch) + ". True Predicted Neg="+str(numberOfTrueNegativesInSubepoch)+"/"+str(numberOfRealNegativesInSubepoch))
		myLogger.print3(initialStringOfLine + " mean Dice: "+str(meanDiceOfSubepoch))
		if train0orValidation1 == 0 : 
			myLogger.print3(initialStringOfLine + " mean cost: "+str(meanCostOfSubepoch))

	return arrayWithReportedValuesPerClass


#A main routine in do_training, that runs for every batch of validation and training.
def doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(myLogger,
								train0orValidation1,
								number_of_batches, #This is the integer division of (numb-o-segments/batchSize)
								cnn3dInstance,
								vectorWithWeightsOfTheClassesForCostFunctionOfTraining,
								epoch, #Deprecated
								subepoch) :
	"""
	Returned array is of dimensions [NumberOfClasses x 6]
	For each class: [meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch, meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, meanCostOfSubepoch]
	In the case of VALIDATION, meanCostOfSubepoch is just a placeholder. Only valid when training.
	"""
	errorsOfBatchesAdded = 0
	costsOfBatchesAdded = 0
	#each row in the array below will hold the number of Real Positives, Real Negatives, True Predicted Positives and True Predicted Negatives in the subepoch, in this order.
	shapeOfArrWithNumbersOfPerClassRpRnTpTnInSubep = [ cnn3dInstance.numberOfOutputClasses, 4 ]
	arrayWithNumbersOfPerClassRpRnTpTnInSubepoch = np.zeros(shapeOfArrWithNumbersOfPerClassRpRnTpTnInSubep, dtype="int32")

        for batch_i in xrange(number_of_batches):
		printProgressStep = max(1, number_of_batches/5)
		if train0orValidation1==0 : #training
		        if  batch_i%printProgressStep == 0 :
		            myLogger.print3("Trained on "+str(batch_i)+"/"+str(number_of_batches)+" of the batches for this subepoch...")
			
			listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining = cnn3dInstance.cnnTrainModel(batch_i, vectorWithWeightsOfTheClassesForCostFunctionOfTraining)
			[costOfThisBatch, meanErrorOfBatch] = listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining[:2]
			listWithNumberOfRpRnPpPnForEachClass = listWithCostMeanErrorAndRpRnTpTnForEachClassFromTraining[2:]

		else : #validation
			if  batch_i%printProgressStep == 0 :
		            myLogger.print3("Validated on "+str(batch_i)+"/"+str(number_of_batches)+" of the batches for this subepoch...")

			listWithMeanErrorAndRpRnTpTnForEachClassFromValidation = cnn3dInstance.cnnValidateModel(batch_i)
			costOfThisBatch = 0 #placeholder in case of validation. This function needs to return it, valid for the case of training only.
			meanErrorOfBatch = listWithMeanErrorAndRpRnTpTnForEachClassFromValidation[0]
			listWithNumberOfRpRnPpPnForEachClass = listWithMeanErrorAndRpRnTpTnForEachClassFromValidation[1:]

		#The returned listWithNumberOfRpRnPpPnForEachClass holds Real Positives, Real Negatives, True Predicted Positives and True Predicted Negatives for all classes in this order, flattened. First RpRnTpTn are for WHOLE "class".
		arrayWithNumberOfRpRnPpPnForEachClassForBatch = np.asarray(listWithNumberOfRpRnPpPnForEachClass, dtype="int32").reshape(shapeOfArrWithNumbersOfPerClassRpRnTpTnInSubep, order='C')
	
		#New addition, to calculate the dice on each training/validation batch
		errorsOfBatchesAdded += meanErrorOfBatch
		costsOfBatchesAdded += costOfThisBatch #only really used in training.
		arrayWithNumbersOfPerClassRpRnTpTnInSubepoch += arrayWithNumberOfRpRnPpPnForEachClassForBatch

		#============BATCH REGULARIZATION ROLLING AVERAGE===============
		cnn3dInstance.updateTheMatricesOfTheLayersWithTheLastMusAndVarsForTheRollingAverageOfTheBatchNormInference()

	
	#Calculate and Report mean accuracy over subepoch for each class_i:
	arrayWithReportedValuesPerClass = calculateAndReportMeanAccuracyOverWholeSubepochForEachClass(	myLogger,
													cnn3dInstance,
													train0orValidation1,
													subepoch,
													number_of_batches,
													errorsOfBatchesAdded,
													costsOfBatchesAdded,
													arrayWithNumbersOfPerClassRpRnTpTnInSubepoch
													)

	#return [meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch, meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, meanCostOfSubepoch]
	return arrayWithReportedValuesPerClass


	





#---------------------------------------------TRAINING-------------------------------------

#batch_size should be 1 or even.
def do_training(myLogger,
		fileToSaveTrainedCnnModelTo,
		cnn3dInstance,

		performValidationOnSamplesDuringTrainingProcessBool, #REQUIRED FOR AUTO SCHEDULE.

		savePredictionImagesSegmentationAndProbMapsListWhenEvaluatingDiceForValidation,
		listOfNamesToGiveToPredictionsValidationIfSavingWhenEvalDice,

                listOfFilepathsToEachChannelOfEachPatientTraining, #NEW
                listOfFilepathsToEachChannelOfEachPatientValidation, #NEW

		listOfFilepathsToGtLabelsOfEachPatientTraining, #NEW
		providedGtForValidationBool, #NEW
		listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc, #NEW

                providedMaskWhereToGetPositiveSamplesTraining, #NEW NOT SURE IF THIS IS NEEDED. But lets keep it simple and consistent.
		listOfFilepathsToMasksOfEachPatientForPosSamplingTraining, #NEW
                providedMaskWhereToGetPositiveSamplesValidation, #NEW
		listOfFilepathsToMasksOfEachPatientForPosSamplingValidation, #NEW

		providedRoiMaskForTrainingBool, #NEW
		listOfFilepathsToRoiMaskTrainAugmOfEachPatientTraining, #NEW
                providedRoiMaskForFastInfValidationBool, #NEW
                listOfFilepathsToRoiMaskFastInfOfEachPatientValidation, #NEW

                providedMaskWhereToGetNegativeSamplesTraining, #NEW
		listOfFilepathsToMasksOfEachPatientForNegSamplingTraining, #NEW
                providedMaskWhereToGetNegativeSamplesValidation, #NEW
		listOfFilepathsToMasksOfEachPatientForNegSamplingValidation, #NEW

		theMasksWhereToGetAreProbabilityMapsTraining,
		theMasksWhereToGetAreProbabilityMapsValidation,
                borrowFlag,
                n_epochs, #every epoch I save my cnnModel
                number_of_subepochs, #per epoch. Every subepoch I get my Accuracy reported
                n_images_per_subepoch,  #the max that can be fit in CPU memory. these are never in GPU. Only ImageParts in GPU
                imagePartsLoadedInGpuPerSubepoch, #Keep this even for now. So that I have same number of pos-neg PAIRS. If it's odd, still will be int divided by two so the lower even will be used.
		imagePartsLoadedInGpuPerSubepochValidation,
		percentThatArePositiveSamplesTraining,
		percentThatArePositiveSamplesValidation,

		#-------Preprocessing-----------
		padInputImagesBool,
		smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
		#-------Data Augmentation-------
		normAugmNone0OnImages1OrSegments2AlreadyNormalized1SubtrUpToPropOfStdAndDivideWithUpToPerc,
		reflectImageWithHalfProbDuringTraining,

                useSameSubChannelsAsSingleScale,

                listOfFilepathsToEachSubsampledChannelOfEachPatientTraining, #NEW
                listOfFilepathsToEachSubsampledChannelOfEachPatientValidation, #NEW
		
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
    
    NUMBER_OF_CLASSES = cnn3dInstance.numberOfOutputClasses # normally ==2, positive lesions and negative healthy

    usingSubsampledWaypath = len(cnn3dInstance.cnnLayersSubsampled)>0 #Flag that says if I should be loading subsampled channels etc.

    start_training_time = time.clock()


    patchDimensions = cnn3dInstance.patchDimensions
    imagePartDimensionsTraining = cnn3dInstance.imagePartDimensionsTraining
    imagePartDimensionsValidation = cnn3dInstance.imagePartDimensionsValidation
    subsampledImagePartDimensionsTraining = cnn3dInstance.subsampledImagePartDimensionsTraining
    subsampledImagePartDimensionsValidation = cnn3dInstance.subsampledImagePartDimensionsValidation
    subSamplingFactor = cnn3dInstance.subsampleFactor



    number_of_batches_training = imagePartsLoadedInGpuPerSubepoch/cnn3dInstance.batchSize
    numberOfPositiveSamplesPerSubepoch = int(imagePartsLoadedInGpuPerSubepoch*percentThatArePositiveSamplesTraining)
    numberOfNegativeSamplesPerSubepoch = imagePartsLoadedInGpuPerSubepoch - numberOfPositiveSamplesPerSubepoch
    number_of_batches_validation = imagePartsLoadedInGpuPerSubepochValidation/cnn3dInstance.batchSizeValidation
    numberOfPositiveSamplesPerSubepochValidation = int(imagePartsLoadedInGpuPerSubepochValidation * percentThatArePositiveSamplesValidation)
    numberOfNegativeSamplesPerSubepochValidation = imagePartsLoadedInGpuPerSubepochValidation - numberOfPositiveSamplesPerSubepochValidation

    # Instantiate the two arrays that will hold the accuracy-values achieved in each subepoch, so that I report them in the end of each epoch:
    # [meanErrorOfSubepoch, meanAccuracyOfSubepoch, meanAccuracyOnPositivesOfSubepoch,
    #	meanAccuracyOnNegativesOfSubepoch, meanDiceOfSubepoch, meanCostOfSubepoch].
    # meanErrorOfSubepoch is not really reported. meanCostOfSubepoch is placeholder in Validation.
    arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringValidation = np.zeros([cnn3dInstance.numberOfOutputClasses, 6, number_of_subepochs], dtype="float32")
    arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringTraining = np.zeros([cnn3dInstance.numberOfOutputClasses, 6, number_of_subepochs], dtype="float32")

    meanValidationAccuracyOfLastEpoch = -1 #For the auto-reduction of the Learning Rate


    #---------To run PARALLEL the extraction of parts for the next subepoch---
    ppservers = () # tuple of all parallel python servers to connect with
    job_server = pp.Server(ppservers=ppservers) # Creates jobserver with automatically detected number of workers

    tupleWithParametersForTraining = (myLogger,
                           0,
                           n_images_per_subepoch,
                           numberOfPositiveSamplesPerSubepoch,
                           numberOfNegativeSamplesPerSubepoch,
                           usingSubsampledWaypath,

                           listOfFilepathsToEachChannelOfEachPatientTraining, #NEW

                           listOfFilepathsToGtLabelsOfEachPatientTraining,

                           providedRoiMaskForTrainingBool, #NEW
                           listOfFilepathsToRoiMaskTrainAugmOfEachPatientTraining, #NEW

                           providedMaskWhereToGetPositiveSamplesTraining, #NEW
                           listOfFilepathsToMasksOfEachPatientForPosSamplingTraining, #NEW
                           providedMaskWhereToGetNegativeSamplesTraining, #NEW
                           listOfFilepathsToMasksOfEachPatientForNegSamplingTraining, #NEW
                           theMasksWhereToGetAreProbabilityMapsTraining,
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
                           numberOfPositiveSamplesPerSubepochValidation,
                           numberOfNegativeSamplesPerSubepochValidation,
                           usingSubsampledWaypath,

                           listOfFilepathsToEachChannelOfEachPatientValidation, #NEW

                           listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc, #NEW

                           "placeholderProvidedRoiMaskBool",
                           "placeholderListOfFilepathsToRoiMask",

                           providedMaskWhereToGetPositiveSamplesValidation, #NEW
                           listOfFilepathsToMasksOfEachPatientForPosSamplingValidation, #NEW
                           providedMaskWhereToGetNegativeSamplesValidation, #NEW
                           listOfFilepathsToMasksOfEachPatientForNegSamplingValidation, #NEW
                           theMasksWhereToGetAreProbabilityMapsValidation,
                           useSameSubChannelsAsSingleScale,

                           listOfFilepathsToEachSubsampledChannelOfEachPatientValidation,

                           imagePartDimensionsValidation,
                           patchDimensions,
                           subSamplingFactor,
                           subsampledImagePartDimensionsValidation,

                           padInputImagesBool,
                           smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
                           [0, -1,-1,-1],
                           [0,0,0]
                           )
    tupleWithLocalFunctionsThatWillBeCalledByTheMainJob = (get_random_image_indices_to_load_on_GPU,
                                                           actual_load_patient_images_from_filepath_and_return_nparrays,
                                                           smoothImageWithGaussianFilterIfNeeded,
                                                           reflectImageArrayIfNeeded,
                                                           padCnnInputs,
                                                           get_positive_and_negative_imageParts,
                                                           getImagePartFromSubsampledImageForTraining
                                                           )
    tupleWithModulesToImportWhichAreUsedByTheJobFunctions = ("random", "time", "numpy as np", "nibabel as nib", "math", "from deepmedic.genericHelpers import *", "from scipy.ndimage.filters import gaussian_filter")
    boolItIsTheVeryFirstSubepochOfThisProcess = True #to know so that in the very first I sequencially load the data for it.
    #------End for parallel------


    while cnn3dInstance.numberOfEpochsTrained < n_epochs :
	epoch = cnn3dInstance.numberOfEpochsTrained

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
		                  						numberOfPositiveSamplesPerSubepochValidation,
		                  						numberOfNegativeSamplesPerSubepochValidation,
		                  						usingSubsampledWaypath,

		                  						listOfFilepathsToEachChannelOfEachPatientValidation, #NEW

										listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc, #NEW

										"placeholderProvidedRoiMaskBool",
										"placeholderListOfFilepathsToRoiMask",

										providedMaskWhereToGetPositiveSamplesValidation, #NEW
		                  						listOfFilepathsToMasksOfEachPatientForPosSamplingValidation, #NEW
										providedMaskWhereToGetNegativeSamplesValidation, #NEW
		                  						listOfFilepathsToMasksOfEachPatientForNegSamplingValidation, #NEW
										theMasksWhereToGetAreProbabilityMapsValidation,
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

		    arrayNormalizedImagePartsChannelsToLoadOnGpuForSubepochValidation = np.asarray(imagePartsChannelsToLoadOnGpuForSubepochValidation, dtype='float32')
		    arrayLesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochValidation = np.asarray(lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochValidation, dtype='float32')

		    myLogger.print3("DEBUG: For Validation, loading to shared variable that many Segments: " + str(len(arrayNormalizedImagePartsChannelsToLoadOnGpuForSubepochValidation))) 
		    cnn3dInstance.sharedValidationNiiData_x.set_value(arrayNormalizedImagePartsChannelsToLoadOnGpuForSubepochValidation, borrow=borrowFlag)
		    cnn3dInstance.sharedValidationNiiLabels_y.set_value(arrayLesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochValidation, borrow=borrowFlag)

		    if usingSubsampledWaypath :
			arraySubsampledNormalizedImagePartsChannelsToLoadOnGpuForSubepochValidation = np.asarray(subsampledImagePartsChannelsToLoadOnGpuForSubepochValidation, dtype='float32')
		        cnn3dInstance.sharedValidationSubsampledData_x.set_value(arraySubsampledNormalizedImagePartsChannelsToLoadOnGpuForSubepochValidation, borrow=borrowFlag)
		    
		    end_loadingToGpu_time = time.clock()
		    myLogger.print3("TIMING: Loading sharedVariables for Validation in epoch|subepoch="+str(epoch)+"|"+str(subepoch)+" took time: "+str(end_loadingToGpu_time-start_loadingToGpu_time)+"(s)")
		    #TRY TO CLEAR THE VARIABLES before the parallel job starts loading stuff again? Or will it cause problems because the shared variables are borrow=True?
		    arrayNormalizedImagePartsChannelsToLoadOnGpuForSubepochValidation = ""
		    arrayLesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochValidation = ""
		    if usingSubsampledWaypath :
			arraySubsampledNormalizedImagePartsChannelsToLoadOnGpuForSubepochValidation = ""


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

		    arrayWithReportedAccuracyValuesPerClassForValidationForSubepoch = doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(myLogger,
													train0orValidation1,
													number_of_batches_validation,
													cnn3dInstance,
													vectorWithWeightsOfTheClassesForCostFunctionOfTraining,
													epoch,
													subepoch)

		    arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringValidation[:,:, subepoch] = arrayWithReportedAccuracyValuesPerClassForValidationForSubepoch

		    cnn3dInstance.freeGpuValidationData()

		    end_validationForSubepoch_time = time.clock()
		    myLogger.print3("TIMING: Validating on the batches of this subepoch #" + str(subepoch) + " took time: "+str(end_validationForSubepoch_time-start_validationForSubepoch_time)+"(s)")
	
		    #Update cnn's top achieved validation accuracy if needed: (for the autoReduction of Learning Rate.)
		    averageValidationAccuracyOfAllSubepochsOfThisEpochForWHOLEClass = arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringValidation[0, 1, :]
		    averageValidationAccuracyOfThisLastEpochForWHOLEClass = np.mean(averageValidationAccuracyOfAllSubepochsOfThisEpochForWHOLEClass)
		    cnn3dInstance.checkMeanValidationAccOfLastEpochAndUpdateCnnsTopAccAchievedIfNeeded(myLogger,
											averageValidationAccuracyOfThisLastEpochForWHOLEClass,
											minIncreaseInValidationAccuracyConsideredForLrSchedule)
            #-------------------END OF THE VALIDATION-DURING-TRAINING-LOOP-------------------------


            #-------------------------GET DATA FOR THIS SUBEPOCH's TRAINING---------------------------------
            if not performValidationOnSamplesDuringTrainingProcessBool and boolItIsTheVeryFirstSubepochOfThisProcess :
		[imagePartsChannelsToLoadOnGpuForSubepochTraining,
		lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining,
		subsampledImagePartsChannelsToLoadOnGpuForSubepochTraining] = getTheArraysOfImageChannelsAndLesionsToLoadToGpuForSubepoch(myLogger,
									   0,
									   n_images_per_subepoch,
									   numberOfPositiveSamplesPerSubepoch,
									   numberOfNegativeSamplesPerSubepoch,
									   usingSubsampledWaypath,

									   listOfFilepathsToEachChannelOfEachPatientTraining, #NEW

									   listOfFilepathsToGtLabelsOfEachPatientTraining,

									   providedRoiMaskForTrainingBool, #NEW
									   listOfFilepathsToRoiMaskTrainAugmOfEachPatientTraining, #NEW

									   providedMaskWhereToGetPositiveSamplesTraining, #NEW
									   listOfFilepathsToMasksOfEachPatientForPosSamplingTraining, #NEW
									   providedMaskWhereToGetNegativeSamplesTraining, #NEW
									   listOfFilepathsToMasksOfEachPatientForNegSamplingTraining, #NEW
									   theMasksWhereToGetAreProbabilityMapsTraining,
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
            #Do it for only few epochs, until I get to an ok local minimum neighbourhood.
            if cnn3dInstance.numberOfEpochsTrained < numberOfEpochsToWeightTheClassesInTheCostFunction :
		#This part was recently changed to multiclass. NOT VALIDATED, but should work.
		numOfPatchesInTheSubepoch_notParts = np.prod(arrayLesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining.shape)
		actualNumOfPatchesPerClassInTheSubepoch_notParts = np.bincount(np.ravel(arrayLesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining).astype(int))
		multiplierToFadePerEpoch = (numberOfEpochsToWeightTheClassesInTheCostFunction - cnn3dInstance.numberOfEpochsTrained)*1.0 / numberOfEpochsToWeightTheClassesInTheCostFunction
		nominatorForEachClass = actualNumOfPatchesPerClassInTheSubepoch_notParts + (numOfPatchesInTheSubepoch_notParts - actualNumOfPatchesPerClassInTheSubepoch_notParts) * multiplierToFadePerEpoch
		vectorWithWeightsOfTheClassesForCostFunctionOfTraining = nominatorForEachClass / ((actualNumOfPatchesPerClassInTheSubepoch_notParts)*NUMBER_OF_CLASSES+TINY_FLOAT)
		myLogger.print3("DEBUG: numOfPatchesInTheSubepoch_notParts="+str(numOfPatchesInTheSubepoch_notParts))
		myLogger.print3("DEBUG: actualNumOfPatchesPerClassInTheSubepoch_notParts="+str(actualNumOfPatchesPerClassInTheSubepoch_notParts))
		myLogger.print3("DEBUG: multiplierToFadePerEpoch ="+str(multiplierToFadePerEpoch))
		myLogger.print3("DEBUG: vectorWithWeightsOfTheClassesForCostFunctionOfTraining="+str(vectorWithWeightsOfTheClassesForCostFunctionOfTraining))

            else :
		vectorWithWeightsOfTheClassesForCostFunctionOfTraining = np.ones(NUMBER_OF_CLASSES, dtype='float32')


            #----------------------------------LOAD TRAINING DATA ON GPU-------------------------------
            myLogger.print3("Loading Training data for subepoch #"+str(subepoch)+" on shared variable...")
            start_loadingToGpu_time = time.clock()

            arrayNormalizedImagePartsChannelsToLoadOnGpuForSubepochTraining = np.asarray(imagePartsChannelsToLoadOnGpuForSubepochTraining, dtype='float32')
            arrayLesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining = np.asarray(lesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining, dtype='float32')

            cnn3dInstance.sharedTrainingNiiData_x.set_value(arrayNormalizedImagePartsChannelsToLoadOnGpuForSubepochTraining, borrow=borrowFlag)
            cnn3dInstance.sharedTrainingNiiLabels_y.set_value(arrayLesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining, borrow=borrowFlag)
            if usingSubsampledWaypath :
		arraySubsampledNormalizedImagePartsChannelsToLoadOnGpuForSubepochTraining = np.asarray(subsampledImagePartsChannelsToLoadOnGpuForSubepochTraining, dtype='float32')
                cnn3dInstance.sharedTrainingSubsampledData_x.set_value(arraySubsampledNormalizedImagePartsChannelsToLoadOnGpuForSubepochTraining, borrow=borrowFlag)
            end_loadingToGpu_time = time.clock()
            myLogger.print3("TIMING: Loading sharedVariables for Training in epoch|subepoch="+str(epoch)+"|"+str(subepoch)+" took time: "+str(end_loadingToGpu_time-start_loadingToGpu_time)+"(s)")
            #TRY TO CLEAR THE VARIABLES before the parallel job starts loading stuff again? Or will it cause problems because the shared variables are borrow=True?
            arrayNormalizedImagePartsChannelsToLoadOnGpuForSubepochTraining = ""
            arrayLesionLabelsForPATCHESOfTheImagePartsInGpUForSubepochTraining = ""
            if usingSubsampledWaypath :
		arraySubsampledNormalizedImagePartsChannelsToLoadOnGpuForSubepochTraining = ""

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
            arrayWithReportedAccuracyValuesPerClassForTrainingForSubepoch = doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch(myLogger,
												train0orValidation1,
												number_of_batches_training,
												cnn3dInstance,
												vectorWithWeightsOfTheClassesForCostFunctionOfTraining,
												epoch,
												subepoch)

            arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringTraining[:,:, subepoch] = arrayWithReportedAccuracyValuesPerClassForTrainingForSubepoch

            cnn3dInstance.freeGpuTrainingData()

            end_trainingForSubepoch_time = time.clock()
            myLogger.print3("TIMING: Training on the batches of this subepoch #" + str(subepoch) + " took time: "+str(end_trainingForSubepoch_time-start_trainingForSubepoch_time)+"(s)")


	myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )
	myLogger.print3("~~~~~~~~~~~~~~~~~ Epoch #" + str(epoch) + " finished. Reporting Accuracy over whole epoch. ~~~~~~~~~~~~~~~~~" )
	myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" )

	reportAccuracyValuesOverWholeEpochForEachClass(myLogger,
						cnn3dInstance.numberOfOutputClasses,
						epoch,
						performValidationOnSamplesDuringTrainingProcessBool,
						arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringValidation,
						arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringTraining,
						)
	arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringValidation *= 0.
	arrayThatHoldsForEachClassTheAccuracyValuesAchievedInEachSubepochDuringTraining *= 0.


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
        cnn3dInstance.numberOfEpochsTrained += 1

        myLogger.print3("SAVING: Epoch #"+str(epoch)+" finished. Saving CNN model.")
        dump_cnn_to_gzip_file_dotSave(cnn3dInstance, fileToSaveTrainedCnnModelTo+"."+str(datetime.datetime.now()), myLogger)
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

					listOfFilepathsToEachChannelOfEachPatientValidation, #NEW

					providedGtForValidationBool, #NEW
					listOfFilepathsToGtLabelsOfEachPatientValidationOnSamplesAndDsc, #NEW

		                        providedRoiMaskForFastInfValidationBool, #NEW
		                        listOfFilepathsToRoiMaskFastInfOfEachPatientValidation, #NEW

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

    dump_cnn_to_gzip_file_dotSave(cnn3dInstance, fileToSaveTrainedCnnModelTo+".final."+str(datetime.datetime.now()), myLogger)

    end_training_time = time.clock()
    myLogger.print3("TIMING: Training process took time: "+str(end_training_time-start_training_time)+"(s)")
    myLogger.print3("The whole do_training() function has finished.")


#---------------------------------------------TESTING-------------------------------------

def performInferenceForTestingOnWholeVolumes(myLogger,
                             validation0orTesting1,
                             savePredictionImagesSegmentationAndProbMapsList,
                             cnn3dInstance,

                             listOfFilepathsToEachChannelOfEachPatient, #NEW

                             providedGtLabelsBool, #boolean NEW. DSC calculation will be performed if this is provided.
                             listOfFilepathsToGtLabelsOfEachPatient, #NEW

                             providedRoiMaskForFastInfBool, #NEW
                             listOfFilepathsToRoiMaskFastInfOfEachPatient, #NEW

                             borrowFlag,
                             listOfNamesToGiveToPredictionsIfSavingResults,
                             
                             #----Preprocessing------
                             padInputImagesBool,
                             smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,

                             useSameSubChannelsAsSingleScale,
                             listOfFilepathsToEachSubsampledChannelOfEachPatient, #NEW


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
    
    NUMBER_OF_CLASSES = cnn3dInstance.numberOfOutputClasses

    usingSubsampledWaypath = len(cnn3dInstance.cnnLayersSubsampled)>0 #Flag that says if I should be loading subsampled channels etc.

    total_number_of_images = len(listOfFilepathsToEachChannelOfEachPatient)    
    batch_size = cnn3dInstance.batchSizeTesting

    #one dice score for whole + for each class)
    diceCoefficients1 = np.zeros((total_number_of_images, NUMBER_OF_CLASSES), dtype="float32") #AllpredictedLes/AllLesions
    diceCoefficients2 = np.zeros((total_number_of_images, NUMBER_OF_CLASSES), dtype="float32") #predictedInsideBrainmask/AllLesions
    diceCoefficients3 = np.zeros((total_number_of_images, NUMBER_OF_CLASSES), dtype="float32") #predictedInsideBrainMask/ LesionsInsideBrainMAsk (for comparisons)

    patchDimensions = cnn3dInstance.patchDimensions
    imagePartDimensions = cnn3dInstance.imagePartDimensionsTesting
    subsampledImagePartDimensions = cnn3dInstance.subsampledImagePartDimensionsTesting
    subSamplingFactor = cnn3dInstance.subsampleFactor

    #stride is how much I move in each dimension to acquire the next imagePart. 
    #I move exactly the number I segment in the centre of each image part (originally this was 9^3 segmented per imagePart).
    numberOfCentralVoxelsClassified = [imagePartDimensions[0]-patchDimensions[0]+1,
                                              imagePartDimensions[1]-patchDimensions[1]+1,
                                              imagePartDimensions[2]-patchDimensions[2]+1] #this is so that I can do smalles strides for the RMF extension.
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
					numberOfFeatureMapsInThisLayer = cnn3dInstance.typesOfCnnLayers[pathwayType_i][layer_i].numberOfFeatureMaps
					indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] = min(indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1], numberOfFeatureMapsInThisLayer)
					totalNumberOfFMsToProcess += indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[1] - indicesOfFmsToVisualiseForCertainLayerOfCertainPathway[0]
				

    for image_i in xrange(total_number_of_images) :
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        myLogger.print3("~~~~~~~~~~~~~~~~~~~~ Segmenting subject with index #"+str(image_i)+" ~~~~~~~~~~~~~~~~~~~~")
                                                  
        #load the image channels in cpu

        [imageChannels, #a nparray(channels,dim0,dim1,dim2)
	gtLabelsImage, #only for accurate/correct DICE1-2 calculation
        maskWhereToGetPositiveSamplesFromPlaceholder, #only used in training
        brainMask, 
        maskWhereToGetNegativeSamplesFromPlaceholder, #only used in training
        allSubsampledChannelsOfPatientInNpArray,  #a nparray(channels,dim0,dim1,dim2)
	tupleOfPaddingPerAxesLeftRight #( (padLeftR, padRightR), (padLeftC,padRightC), (padLeftZ,padRightZ)). "placeholderNothing" when no padding.
        ] = actual_load_patient_images_from_filepath_and_return_nparrays(
                                                    myLogger,
                                                    2,

                                                    image_i,

                                                    listOfFilepathsToEachChannelOfEachPatient, #NEW
						
                                                    providedGtLabelsBool, #NEW
                                                    listOfFilepathsToGtLabelsOfEachPatient, #NEW

                                                    providedMaskWhereToGetPositiveSamples = False, #NEW
                                                    listOfFilepathsToMasksOfEachPatientForPosSamplingForTrainOrVal = "placeholder", #NEW

                                                    providedRoiMaskBool = providedRoiMaskForFastInfBool, #NEW
                                                    listOfFilepathsToRoiMaskOfEachPatient = listOfFilepathsToRoiMaskFastInfOfEachPatient, #NEW

                                                    providedMaskWhereToGetNegativeSamples = False, #NEW
                                                    listOfFilepathsToMasksOfEachPatientForNegSamplingForTrainOrVal = "placeholder", #Keep them, I ll need them for automatic accuracy reports.
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
        [imagePartsChannelsToLoadOnGpuForThisImage,
         coordsOfTopLeftVoxelForPartsToReturn,
         subsampledImagePartsChannelsToLoadOnGpuForThisImage
         ] = getImagePartsAndTheirSlices(myLogger=myLogger,
                                         strideOfImagePartsPerDimensionInVoxels=strideOfImagePartsPerDimensionInVoxels,
                                         imagePartDimensions = imagePartDimensions,
                                         batch_size = batch_size,
                                         channelsOfImageNpArray = imageChannels,#chans,niiDims
                                         brainMask = brainMask,
                                         #New for the extension of the cnn.
                                         channelsOfSubsampledImageNpArray=allSubsampledChannelsOfPatientInNpArray,
                                         patchDimensions=patchDimensions,
                                         subsampledImageChannels=allSubsampledChannelsOfPatientInNpArray,
                                         subSamplingFactor=subSamplingFactor,
                                         subsampledImagePartDimensions=subsampledImagePartDimensions,
                                         )
        
        myLogger.print3("Loading data for subject #"+str(image_i)+" on sharedvariable...")
        cnn3dInstance.sharedTestingNiiData_x.set_value(np.asarray(imagePartsChannelsToLoadOnGpuForThisImage, dtype='float32'), borrow=borrowFlag)
        if usingSubsampledWaypath :
            cnn3dInstance.sharedTestingSubsampledData_x.set_value(np.asarray(subsampledImagePartsChannelsToLoadOnGpuForThisImage, dtype='float32'), borrow=borrowFlag)
 
        myLogger.print3("All the Segments for the current subject were loaded on the shared variable.")
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
                
        totalNumberOfImagePartsToProcessForThisImage = len(imagePartsChannelsToLoadOnGpuForThisImage)
        myLogger.print3("Total number of Segments to process:"+str(totalNumberOfImagePartsToProcessForThisImage))

        imagePartOfConstructedProbMap_i = 0
        imagePartOfConstructedFeatureMaps_i = 0
	number_of_batches = totalNumberOfImagePartsToProcessForThisImage/batch_size
        for batch_i in xrange(number_of_batches) : #batch_size = how many image parts in one batch. Has to be the same with the batch_size it was created with. This is no problem for testing. Could do all at once, or just 1 image part at time.

            printProgressStep = max(1, number_of_batches/5)
            if batch_i%printProgressStep == 0:
		myLogger.print3("Processed "+str(batch_i*batch_size)+"/"+str(number_of_batches*batch_size)+" Segments.")

            #NEW WAY, WITH THE TEST+VISUALISE:
            featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch = cnn3dInstance.cnnTestAndVisualiseAllFmsFunction(batch_i)
            predictionForATestBatch = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[-1]
            listWithTheFmsOfAllLayersSortedByPathwayTypeForTheBatch = featureMapsOfEachLayerAndPredictionProbabilitiesAtEndForATestBatch[:-1]
            #No reshape needed, cause I now do it internally. But to dimensions (batchSize, FMs, R,C,Z).

            #~~~~~~~~~~~~~~~~CONSTRUCT THE PREDICTED PROBABILITY MAPS~~~~~~~~~~~~~~
            #From the results of this batch, create the prediction image by putting the predictions to the correct place in the image.
            for imagePart_in_this_batch_i in xrange(batch_size) :

                #Now put the label-cube in the new-label-segmentation-image, at the correct position. 
                #The very first label goes not in index 0,0,0 but half-patch further away! At the position of the central voxel of the top-left patch!
                coordsOfTopLeftVoxelForThisPart = coordsOfTopLeftVoxelForPartsToReturn[imagePartOfConstructedProbMap_i]
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
		       			coordsOfTopLeftVoxelForThisPart = coordsOfTopLeftVoxelForPartsToReturn[imagePartOfConstructedFeatureMaps_i + imagePart_in_this_batch_i]
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


	#=================Save Predicted-Probability-Map and Evaluate Dice====================
	segmentationImage = np.argmax(labelImageCreatedByPredictions, axis=0) #The SEGMENTATION.

	#Save Result:
        if savePredictionImagesSegmentationAndProbMapsList[0] == True : #save predicted segmentation
            npDtypeForPredictedImage = np.dtype(np.int16)
            suffixToAdd = "_Segm"
            #Save the image. Pass the filename paths of the normal image so that I can dublicate the header info, eg RAS transformation.
            unpaddedSegmentationImage = segmentationImage if not padInputImagesBool else unpadCnnOutputs(segmentationImage, tupleOfPaddingPerAxesLeftRight)
            savePredictedImageToANewNiiWithHeaderFromOther(unpaddedSegmentationImage,
                               listOfNamesToGiveToPredictionsIfSavingResults,

                               listOfFilepathsToEachChannelOfEachPatient, #NEW

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

				               listOfFilepathsToEachChannelOfEachPatient, #NEW

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
							unpaddedFmToSave =  fmToSave if not padInputImagesBool else unpadCnnOutputs(fmToSave, tupleOfPaddingPerAxesLeftRight)
							saveFmActivationImageToANewNiiWithHeaderFromOther(unpaddedFmToSave,
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
		saveMultidimensionalImageWithAllVisualisedFmsToANewNiiWithHeaderFromOther(unpaddedMultidimensionalImageWithAllToBeVisualisedFmsArrayWith4thDimAsFms,
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
		if brainMask <> "placeholderNothing" : #If brainmask was given:
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
			diceCoefficients1[image_i, class_i] = diceCoeff1
			#Dice2 = PredictedWithinBrainMask / AllLesions
			diceCoeff2 = calculateDiceCoefficient(predictedLabelImageConvolvedWithBrainMask, booleanGtLesionLabelsForDiceEvaluation_unstripped)
			diceCoefficients2[image_i, class_i] = diceCoeff2
			#Dice3 = PredictedWithinBrainMask / LesionsInsideBrainMask
			diceCoeff3 = calculateDiceCoefficient(predictedLabelImageConvolvedWithBrainMask, booleanGtLesionLabelsForDiceEvaluation_unstripped * multiplyWithBrainMaskOr1)
			diceCoefficients3[image_i, class_i] = diceCoeff3

		myLogger.print3("ACCURACY: (" + str(validationOrTestingString) + ") The Per-Class DICE Coefficients for subject with index #"+str(image_i)+" equal: DICE1="+strFlList4Dec(diceCoefficients1[image_i])+" DICE2="+strFlList4Dec(diceCoefficients2[image_i])+" DICE3="+strFlList4Dec(diceCoefficients3[image_i]))
		printExplanationsAboutDice(myLogger)

    #================= Loops for all patients have finished. Now lets just report the average DSC over all the processed patients. ====================
    if providedGtLabelsBool : # Ground Truth was provided for calculation of DSC. Do DSC calculation.
	    myLogger.print3("+++++++++++++++++++++++++++++++ Segmentation of all subjects finished +++++++++++++++++++++++++++++++++++")
	    myLogger.print3("+++++++++++++++++++++ Reporting Average Segmentation Metrics over all subjects ++++++++++++++++++++++++++")
	    meanDiceCoefficients1 = np.mean(diceCoefficients1,axis=0) if total_number_of_images>0 else [9999]
	    meanDiceCoefficients2 = np.mean(diceCoefficients2,axis=0) if total_number_of_images>0 else [9999]
	    meanDiceCoefficients3 = np.mean(diceCoefficients3,axis=0) if total_number_of_images>0 else [9999]

	    myLogger.print3("ACCURACY: (" + str(validationOrTestingString) + ") The Per-Class average DICE Coefficients over all subjects are: DICE1=" + strFlList4Dec(meanDiceCoefficients1) + " DICE2="+strFlList4Dec(meanDiceCoefficients2)+" DICE3="+strFlList4Dec(meanDiceCoefficients3))
	    printExplanationsAboutDice(myLogger)


    end_validationOrTesting_time = time.clock()
    myLogger.print3("TIMING: "+validationOrTestingString+" process took time: "+str(end_validationOrTesting_time-start_validationOrTesting_time)+"(s)")
    
    myLogger.print3("###########################################################################################################")
    myLogger.print3("############################# Finished full Segmentation of " + str(validationOrTestingString) + " subjects ##########################")
    myLogger.print3("###########################################################################################################")

def printExplanationsAboutDice(myLogger) :
	myLogger.print3("EXPLANATION: DICE1/2/3 are lists with the DICE per class. For Class-0 we calculate DICE for the whole foreground (useful for multi-class problems).")
	myLogger.print3("EXPLANATION: DICE1 is calculated whole segmentation vs whole Ground Truth (GT). DICE2 is the segmentation within the ROI vs GT. DICE3 is segmentation within the ROI vs the GT within the ROI.")

