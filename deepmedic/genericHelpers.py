# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import os
import cPickle
import gzip

import random
import nibabel as nib
import numpy as np

from math import ceil

import datetime

def strFlXDec(flNum, numDec) :
	#ala "{0:.2f}".format(13.94999)
	stringThatSaysHowManyDecimals = "{0:." + str(numDec) + "f}"
	return stringThatSaysHowManyDecimals.format(flNum)

def strFl4Dec(flNum) :
	return strFlXDec(flNum, 4)

#This function is used to get a STRING-list-of-float-with-X-decimal-points. Used for printing.
def strFlListXDec(listWithFloats, numDec) :
	stringThatSaysHowManyDecimals = "%."+str(numDec)+"f"
	strList = "[ "
	for num in listWithFloats :
		strList += stringThatSaysHowManyDecimals % num + " "
	strList += "]"
	#listWithStringFloatsWithLimitedDecimals = [ stringThatSaysHowManyDecimals % num for num in listWithFloats]
	return strList

def strFlList4Dec(listWithFloats) :
	return strFlListXDec(listWithFloats, 4)

def calculateTheZeroIntensityOf3dImage(image3d) :

        intensityZeroOfChannel = np.mean([image3d[0,0,0],
                                    image3d[-1,0,0],
                                    image3d[0,-1,0],
                                    image3d[-1,-1,0],
                                    image3d[0,0,-1],
                                    image3d[-1,0,-1],
                                    image3d[0,-1,-1],
                                    image3d[-1,-1,-1]
                                    ])
        return intensityZeroOfChannel

def calculateDiceCoefficient(predictedBinaryLabels, groundTruthBinaryLabels) :
	unionCorrectlyPredicted = predictedBinaryLabels * groundTruthBinaryLabels
	numberOfTruePositives = np.sum(unionCorrectlyPredicted)
	numberOfGtPositives = np.sum(groundTruthBinaryLabels)
	diceCoeff = (2.0 * numberOfTruePositives) / (np.sum(predictedBinaryLabels) + numberOfGtPositives) if numberOfGtPositives<>0 else 9999 
	return diceCoeff


def load_object_from_file(filenameWithPath) :
    f = file(filenameWithPath, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

def dump_object_to_file(my_obj, filenameWithPath) :
    """
        my_obj = object to pickle
        filenameWithPath = a string with the full path+name
        
        The function uses the 'highest_protocol' which is supposed to be more storage efficient.
        It uses cPickle, which is coded in c and is supposed to be faster than pickle.
        Remember, this instance is safe to load only from a code which is fully-compatible (same version)
        ...with the code this was saved from, i.e. same classes define.
        If I need forward compatibility, read this: http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
    """
    f = file(filenameWithPath, 'wb')
    cPickle.dump(my_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def load_object_from_gzip_file(filenameWithPath) :
    f = gzip.open(filenameWithPath, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

def dump_object_to_gzip_file(my_obj, filenameWithPath) :
    f = gzip.open(filenameWithPath, 'wb')
    cPickle.dump(my_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()



#This could be renamed to be more generic.
def get_random_image_indices_to_load_on_GPU(total_number_of_images, 
                                            max_images_on_gpu_for_subepoch, 
                                            get_max_images_for_gpu_even_if_total_less=False,
                                            myLogger=None):

    images_indices = range(total_number_of_images)
    random_order_chosen_images=[]
    
    random.shuffle(images_indices) #does it in place. Now they are shuffled
    
    if max_images_on_gpu_for_subepoch>=total_number_of_images:
        random_order_chosen_images += images_indices

        if get_max_images_for_gpu_even_if_total_less : #This is if I want to have a certain amount on GPU, even if total images are less.
            while (len(random_order_chosen_images)<max_images_on_gpu_for_subepoch):
                random.shuffle(images_indices)
                number_of_extra_images_to_get_to_fill_gpu = min(max_images_on_gpu_for_subepoch - len(random_order_chosen_images), total_number_of_images)
                random_order_chosen_images += (images_indices[:number_of_extra_images_to_get_to_fill_gpu])
            if len(random_order_chosen_images)<>max_images_on_gpu_for_subepoch :
		if myLogger<>None :
                    myLogger.print3("ERROR: in get_random_image_indices_to_load_on_GPU(), something is wrong!")
		else :
                    print "ERROR: in get_random_image_indices_to_load_on_GPU(), something is wrong!"
                exit(1)
    else:
        random_order_chosen_images += images_indices[:max_images_on_gpu_for_subepoch]

    return random_order_chosen_images







def getSuggestedStdForSubsampledImage(subsampleFactor) :
	arraySubsampledFactor = np.asarray(subsampleFactor)
	suggestedStdsForSubsampledChannels = arraySubsampledFactor/2.0
	#if subsampledFactor == 1 for a certain axis (eg z axis), it means I am actually doing 2D processing. In this case, use std=0 on this axis, and I dont smooth at all. I do clean slice-by-slice.
	suggestedStdsForSubsampledChannels = suggestedStdsForSubsampledChannels * (arraySubsampledFactor<>1)
	return suggestedStdsForSubsampledChannels

#This is the generic function.
def saveImageToANewNiiWithHeaderFromOtherGivenExactFilePaths(labelImageCreatedByPredictions,
                                          fullFilenameToSaveWith,
                                          fullFilenameOfOriginalImageToCopyHeader,
                                          npDtype = np.dtype(np.float32),
                                          myLogger=None) :

    fullFilenameToSaveWith = os.path.abspath(fullFilenameToSaveWith) # Cleans the .././/...
    img_proxy_for_orig_image = nib.load(fullFilenameOfOriginalImageToCopyHeader)
    hdr_for_orig_image = img_proxy_for_orig_image.header
    
    affine_trans_to_ras = img_proxy_for_orig_image.affine
    
    newLabelImg = nib.Nifti1Image(labelImageCreatedByPredictions, affine_trans_to_ras) #Nifti Constructor. data is the image itself, dimensions x,y,z,time. The second argument is the affine RAS transf.
    newLabelImg.set_data_dtype(npDtype)

    dimensionsOfTheGivenArrayImageToSave = len(labelImageCreatedByPredictions.shape)
    newZooms = list(hdr_for_orig_image.get_zooms()[:dimensionsOfTheGivenArrayImageToSave])
    if len(newZooms) < dimensionsOfTheGivenArrayImageToSave : #Eg if original image was 3D, but I need to save a multichannel image.
	newZooms = newZooms + [1.0]*(dimensionsOfTheGivenArrayImageToSave - len(newZooms))
    newLabelImg.header.set_zooms(newZooms)

    if not fullFilenameToSaveWith.endswith(".nii.gz") :
            fullFilenameToSaveWith = fullFilenameToSaveWith + ".nii.gz"
    nib.save(newLabelImg, fullFilenameToSaveWith)

    if myLogger<>None :
        myLogger.print3("Image saved at: " + str(fullFilenameToSaveWith))
    else :
	print("Image saved at: " + str(fullFilenameToSaveWith)) 


def savePredictedImageToANewNiiWithHeaderFromOther(labelImageCreatedByPredictions,
                                          listOfNamesToGiveToPredictions,

                                          listOfFilepathsToEachChannelOfEachPatient,

                                          case_i, #the index (in the list of filepathnames) of the current image segmented.
                                          suffixToAdd = "",
                                          npDtype = np.dtype(np.float32),
                                          myLogger=None) :

    #give as arguments the list of the patient filepaths and the index of the currently segmented image, so that ...
    #... I can get the header, affine RAS trans etc from it and copy it for the new image.
    if myLogger<>None :
        myLogger.print3("Saving the new label (segmentation) image for the subject #"+str(case_i))
    else :
	print("Saving the new label (segmentation) image for the subject #"+str(case_i))


    fullFilenameOfOriginalImageToCopyHeader = listOfFilepathsToEachChannelOfEachPatient[case_i][0]

    fullFilenameToSaveWith = "PLACEHOLDER"
    if listOfNamesToGiveToPredictions[case_i].endswith(".nii.gz") :
        fullFilenameToSaveWith = listOfNamesToGiveToPredictions[case_i][:-7] + suffixToAdd + ".nii.gz"
    elif listOfNamesToGiveToPredictions[case_i].endswith(".nii") :
        fullFilenameToSaveWith = listOfNamesToGiveToPredictions[case_i][:-4] + suffixToAdd + ".nii.gz"
    else :
        fullFilenameToSaveWith = listOfNamesToGiveToPredictions[case_i] + suffixToAdd + ".nii.gz"


    saveImageToANewNiiWithHeaderFromOtherGivenExactFilePaths(labelImageCreatedByPredictions,
                                          fullFilenameToSaveWith,
                                          fullFilenameOfOriginalImageToCopyHeader,
                                          npDtype,
                                          myLogger)
    


def saveFmActivationImageToANewNiiWithHeaderFromOther(fmImageCreatedByVisualisation,
                                          listOfNamesToGiveToPredictions,

                                          listOfFilepathsToEachChannelOfEachPatient,

                                          image_i,
                                          index_of_typeOfPathway_to_visualize,
                                          index_of_layer_in_pathway_to_visualize,
                                          index_of_FM_in_pathway_to_visualize,
                                          myLogger=None) : #the index (in the list of filepathnames) of the current image segmented :
    #give as arguments the list of the patient filepaths and the index of the currently segmented image, so that ...
    #... I can get the header, affine RAS trans etc from it and copy it for the new image.
    
    stringToPrint = "Saving the new Fm-activation image for the subject #"+str(image_i)+", pathway:" + str(index_of_typeOfPathway_to_visualize)\
           + ", layer: " + str(index_of_layer_in_pathway_to_visualize) + " FM: " + str(index_of_FM_in_pathway_to_visualize)	
    if myLogger<>None :
        myLogger.print3(stringToPrint)
    else :
	print(stringToPrint)

    fullFilenameOfOriginalImageToCopyHeader = listOfFilepathsToEachChannelOfEachPatient[image_i][0]
    fullFilenameToSaveWith = "PLACEHOLDER"
    if listOfNamesToGiveToPredictions[image_i].endswith(".nii.gz") :
        fullFilenameToSaveWith = listOfNamesToGiveToPredictions[image_i][:-7] + "_pathway" + str(index_of_typeOfPathway_to_visualize)\
        + "_layer" + str(index_of_layer_in_pathway_to_visualize) + "_fm" + str(index_of_FM_in_pathway_to_visualize) + ".nii.gz"
    elif listOfNamesToGiveToPredictions[image_i].endswith(".nii") :
        fullFilenameToSaveWith = listOfNamesToGiveToPredictions[image_i][:-4] + "_pathway" + str(index_of_typeOfPathway_to_visualize)\
        + "_layer" + str(index_of_layer_in_pathway_to_visualize) + "_fm" + str(index_of_FM_in_pathway_to_visualize) + ".nii.gz"
    else :
        fullFilenameToSaveWith = listOfNamesToGiveToPredictions[image_i] + "_pathway" + str(index_of_typeOfPathway_to_visualize)\
        + "_layer" + str(index_of_layer_in_pathway_to_visualize) + "_fm" + str(index_of_FM_in_pathway_to_visualize) + ".nii.gz"

    saveImageToANewNiiWithHeaderFromOtherGivenExactFilePaths(fmImageCreatedByVisualisation,
                                          fullFilenameToSaveWith,
                                          fullFilenameOfOriginalImageToCopyHeader,
                                          np.dtype(np.float32),
                                          myLogger)



def saveMultidimensionalImageWithAllVisualisedFmsToANewNiiWithHeaderFromOther(multidimImageWithAllVisualisedFms,
                                          listOfNamesToGiveToFmVisualisationsIfSaving,

                                          listOfFilepathsToEachChannelOfEachPatient,

                                          image_i,
                                          myLogger=None) : #the index (in the list of filepathnames) of the current image segmented :
    #give as arguments the list of the patient filepaths and the index of the currently segmented image, so that ...
    #... I can get the header, affine RAS trans etc from it and copy it for the new image.
    
    stringToPrint = "Saving a multi-dimensional image, with all the FMs as a 4th dimension, for the subject #"+str(image_i)	
    if myLogger<>None :
        myLogger.print3(stringToPrint)
    else :
	print(stringToPrint)

    fullFilenameOfOriginalImageToCopyHeader = listOfFilepathsToEachChannelOfEachPatient[image_i][0]
    fullFilenameToSaveWith = "PLACEHOLDER"
    if listOfNamesToGiveToFmVisualisationsIfSaving[image_i].endswith(".nii.gz") :
        fullFilenameToSaveWith = listOfNamesToGiveToFmVisualisationsIfSaving[image_i][:-7] + "_allFmsMultiDim.nii.gz"
    elif listOfNamesToGiveToFmVisualisationsIfSaving[image_i].endswith(".nii") :
        fullFilenameToSaveWith = listOfNamesToGiveToFmVisualisationsIfSaving[image_i][:-4] + "_allFmsMultiDim.nii.gz"
    else :
        fullFilenameToSaveWith = listOfNamesToGiveToFmVisualisationsIfSaving[image_i] + "_allFmsMultiDim.nii.gz"

    saveImageToANewNiiWithHeaderFromOtherGivenExactFilePaths(multidimImageWithAllVisualisedFms,
                                          fullFilenameToSaveWith,
                                          fullFilenameOfOriginalImageToCopyHeader,
                                          np.dtype(np.float32),
                                          myLogger)

def datetimeNowAsStr() :
    #datetime returns in the format: YYYY-MM-DD HH:MM:SS.millis but ':' is not supported for Windows' naming convention.
    return str(datetime.datetime.now()).replace(':','.')