# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, division

import os
import nibabel as nib
import numpy as np


def loadVolume(filepath):
    # Loads the image specified by filepath.
    # Returns a 3D np array.
    # The image can be 2D, but will be returned as 3D, with dimensions =[x, y, 1]
    # It can also be 4D, of shape [x,y,z,1], and will be returned as 3D.
    # If it's 4D with 4th dimension > 1, assertion will be raised.
    proxy = nib.load(filepath)
    img = proxy.get_data()
    proxy.uncache()
    
    if len(img.shape) == 2:
        # 2D image could have been given.
        img = np.expand_dims(img, axis=2)
    elif len(img.shape) > 3 :
        # 4D volumes could have been given. Often 3Ds are stored as 4Ds with 4th dim == 1.
        assert img.shape[3] > 1
        img = img[:,:,:,0]

    return img


#This is the generic function.
def saveImgToNiiWithOriginalHdr(imgToSave,
                                    filepathTarget,
                                    filepathOriginToCopyHeader,
                                    npDtype = np.dtype(np.float32),
                                    log=None):
    # imgToSave: 3d np array.
    # filepathTarget: filepath where to save.
    # filepathOriginToCopyHeader: original image, where to copy the header over to the target image.
    
    # Load original image.
    proxy_origin = nib.load(filepathOriginToCopyHeader)
    hdr_origin = proxy_origin.header
    affine_origin = proxy_origin.affine
    proxy_origin.uncache()
    
    newLabelImg = nib.Nifti1Image(imgToSave, affine_origin)
    newLabelImg.set_data_dtype(npDtype)
    
    dimsImgToSave = len(imgToSave.shape)
    newZooms = list(hdr_origin.get_zooms()[:dimsImgToSave])
    if len(newZooms) < dimsImgToSave : #Eg if original image was 3D, but I need to save a multi-channel image.
        newZooms = newZooms + [1.0]*(dimsImgToSave - len(newZooms))
    newLabelImg.header.set_zooms(newZooms)
    
    filepathTarget = os.path.abspath(filepathTarget)
    if not filepathTarget.endswith(".nii.gz") :
        filepathTarget = filepathTarget + ".nii.gz"
    nib.save(newLabelImg, filepathTarget)
    
    if log!=None :
        log.print3("Image saved at: " + str(filepathTarget))
    else :
        print("Image saved at: " + str(filepathTarget)) 
        
        
        
def savePredImgToNiiWithOriginalHdr(labelImageCreatedByPredictions,
                                    namesForSavingPreds,
                                    listOfFilepathsToEachChannelOfEachPatient,
                                    case_i,
                                    suffixToAdd = "",
                                    npDtype = np.dtype(np.float32),
                                    log=None):
    # case_i: # the index (in the list of filepathnames) of the current image segmented.
    # Needs as argument the cases' filepaths and index of the currently segmented case, so that ...
    # ... I can get the header, affine RAS trans etc from it and copy it for the new image.
    if log is not None :
        log.print3("Saving the new label (segmentation) image for the subject #"+str(case_i))
    else :
        print("Saving the new label (segmentation) image for the subject #"+str(case_i))
        
    filepathOriginToCopyHeader = listOfFilepathsToEachChannelOfEachPatient[case_i][0]
    
    filepathTarget = "PLACEHOLDER"
    nameForSavingPred = namesForSavingPreds[case_i]
    if os.path.isdir(nameForSavingPred) : # Only if names not given and it's only 1 case (see testSessionParams)
        filepathTarget = nameForSavingPred + "/" + suffixToAdd + ".nii.gz"
    elif nameForSavingPred.endswith(".nii.gz") :
        filepathTarget = nameForSavingPred[:-7] + "_" + suffixToAdd + ".nii.gz"
    elif nameForSavingPred.endswith(".nii") :
        filepathTarget = nameForSavingPred[:-4] + "_" + suffixToAdd + ".nii.gz"
    else :
        filepathTarget = nameForSavingPred + "_" + suffixToAdd + ".nii.gz"
        
    saveImgToNiiWithOriginalHdr(labelImageCreatedByPredictions,
                                filepathTarget,
                                filepathOriginToCopyHeader,
                                npDtype,
                                log)



def saveFmImgToNiiWithOriginalHdr(  fmImageCreatedByVisualisation,
                                    namesForSavingPreds,
                                    listOfFilepathsToEachChannelOfEachPatient,
                                    image_i,
                                    index_of_typeOfPathway_to_visualize,
                                    index_of_layer_in_pathway_to_visualize,
                                    index_of_FM_in_pathway_to_visualize,
                                    log=None):
    # case_i: # the index (in the list of filepathnames) of the current image segmented.
    # Needs as argument the cases' filepaths and index of the currently segmented case, so that ...
    # ... I can get the header, affine RAS trans etc from it and copy it for the new image.
    
    stringToPrint = "Saving the new FM-activation image for the subject #"+str(image_i)+", pathway:" + str(index_of_typeOfPathway_to_visualize)\
           + ", layer: " + str(index_of_layer_in_pathway_to_visualize) + " FM: " + str(index_of_FM_in_pathway_to_visualize)        
    if log!=None :
        log.print3(stringToPrint)
    else :
        print(stringToPrint)
        
    filepathOriginToCopyHeader = listOfFilepathsToEachChannelOfEachPatient[image_i][0]
    filepathTarget = "PLACEHOLDER"
    if namesForSavingPreds[image_i].endswith(".nii.gz") :
        filepathTarget = namesForSavingPreds[image_i][:-7] + "_pathway" + str(index_of_typeOfPathway_to_visualize)\
        + "_layer" + str(index_of_layer_in_pathway_to_visualize) + "_fm" + str(index_of_FM_in_pathway_to_visualize) + ".nii.gz"
    elif namesForSavingPreds[image_i].endswith(".nii") :
        filepathTarget = namesForSavingPreds[image_i][:-4] + "_pathway" + str(index_of_typeOfPathway_to_visualize)\
        + "_layer" + str(index_of_layer_in_pathway_to_visualize) + "_fm" + str(index_of_FM_in_pathway_to_visualize) + ".nii.gz"
    else :
        filepathTarget = namesForSavingPreds[image_i] + "_pathway" + str(index_of_typeOfPathway_to_visualize)\
        + "_layer" + str(index_of_layer_in_pathway_to_visualize) + "_fm" + str(index_of_FM_in_pathway_to_visualize) + ".nii.gz"
        
    saveImgToNiiWithOriginalHdr(fmImageCreatedByVisualisation,
                                filepathTarget,
                                filepathOriginToCopyHeader,
                                np.dtype(np.float32),
                                log)



def save4DImgWithAllFmsToNiiWithOriginalHdr(multidimImageWithAllVisualisedFms,
                                            namesForSavingFms,
                                            listOfFilepathsToEachChannelOfEachPatient,
                                            image_i,
                                            log=None):
    stringToPrint = "Saving multi-dimensional image, with all FMs as 4th dimension, " +\
                    "for subject #"+str(image_i)
    if log!=None :
        log.print3(stringToPrint)
    else :
        print(stringToPrint)
        
    filepathOriginToCopyHeader = listOfFilepathsToEachChannelOfEachPatient[image_i][0]
    filepathTarget = "PLACEHOLDER"
    if namesForSavingFms[image_i].endswith(".nii.gz") :
        filepathTarget = namesForSavingFms[image_i][:-7] + "_allFmsMultiDim.nii.gz"
    elif namesForSavingFms[image_i].endswith(".nii") :
        filepathTarget = namesForSavingFms[image_i][:-4] + "_allFmsMultiDim.nii.gz"
    else :
        filepathTarget = namesForSavingFms[image_i] + "_allFmsMultiDim.nii.gz"
        
    saveImgToNiiWithOriginalHdr(multidimImageWithAllVisualisedFms,
                                filepathTarget,
                                filepathOriginToCopyHeader,
                                np.dtype(np.float32),
                                log)
    
    
