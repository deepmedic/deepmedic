# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange
import os
import numpy as np

from deepmedic.genericHelpers import *

# The API for these classes should resemble the API of Pathway and Cnn3d classes. But only what is needed by the sampling process of the training procedure.
class PathwayWrapperForSampling(object):
    # For CnnWrapperForSampling class.
    def __init__(self, pathwayInstance) :
        self._pType = pathwayInstance.pType()
        self._subsFactor = pathwayInstance.subsFactor()
        self._shapeOfInputTrainValTest = pathwayInstance.getShapeOfInput()
        self._shapeOfOutputTrainValTest = pathwayInstance.getShapeOfOutput()
    def pType(self):
        return self._pType
    def subsFactor(self):
        return self._subsFactor
    def getShapeOfInput(self):
        return self._shapeOfInputTrainValTest
    def getShapeOfOutput(self):
        return self._shapeOfOutputTrainValTest
        
class CnnWrapperForSampling(object):
    # Only for the parallel process used during training. So that it won't re-load theano etc. There was a problem with cnmem when reloading theano.
    def __init__(self, cnn3dInstance) :
        # Cnn
        self.recFieldCnn = cnn3dInstance.recFieldCnn
        self.batchSizeTrainValTest = [ cnn3dInstance.batchSize, cnn3dInstance.batchSizeValidation, cnn3dInstance.batchSizeTesting ]
        self.finalTargetLayer_outputShapeTrainValTest = [cnn3dInstance.finalTargetLayer.outputShapeTrain,
                                                        cnn3dInstance.finalTargetLayer.outputShapeVal,
                                                        cnn3dInstance.finalTargetLayer.outputShapeTest ]
        # Pathways related
        self._numPathwaysThatRequireInput = cnn3dInstance.getNumPathwaysThatRequireInput()
        self.numSubsPaths = cnn3dInstance.numSubsPaths
        
        self.pathways = []
        for pathway_i in xrange(len(cnn3dInstance.pathways)) :
            self.pathways.append( PathwayWrapperForSampling(cnn3dInstance.pathways[pathway_i]) )
        
    def getNumPathwaysThatRequireInput(self) :
        return self._numPathwaysThatRequireInput
    
    
def dump_cnn_to_gzip_file_dotSave(cnnInstance, filenameWithPathToSaveTo, logger=None) :
    filenameWithPathToSaveToDotSave = os.path.abspath(filenameWithPathToSaveTo + ".save")
    cnnInstance.freeGpuTrainingData(); cnnInstance.freeGpuValidationData(); cnnInstance.freeGpuTestingData();
    # Clear out the compiled functions, so that they are not saved with the instance:
    compiledFunctionTrain = cnnInstance.cnnTrainModel; cnnInstance.cnnTrainModel = ""
    compiledFunctionVal = cnnInstance.cnnValidateModel; cnnInstance.cnnValidateModel = ""
    compiledFunctionTest = cnnInstance.cnnTestModel; cnnInstance.cnnTestModel = ""
    compiledFunctionVisualise = cnnInstance.cnnVisualiseFmFunction; cnnInstance.cnnVisualiseFmFunction = ""
    
    if logger != None :
        logger.print3("Saving network to: "+str(filenameWithPathToSaveToDotSave))
    else:
        print("Saving network to: "+str(filenameWithPathToSaveToDotSave))
        
    dump_object_to_gzip_file(cnnInstance, filenameWithPathToSaveToDotSave)
    
    if logger != None :
        logger.print3("Model saved.")
    else:
        print("Model saved.")
        
    # Restore instance's values, which were cleared for the saving of the instance:
    cnnInstance.cnnTrainModel = compiledFunctionTrain
    cnnInstance.cnnValidateModel = compiledFunctionVal
    cnnInstance.cnnTestModel = compiledFunctionTest
    cnnInstance.cnnVisualiseFmFunction = compiledFunctionVisualise
    
    return filenameWithPathToSaveToDotSave


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
    if numberOfRFDim != numberOfSegmDim :
        print("ERROR: [in function checkReceptiveFieldFineInComparisonToSegmentSize()] : Receptive field and image segment have different number of dimensions! (should be 3 for both! Exiting!)")
        exit(1)
    for dim_i in xrange(numberOfRFDim) :
        if receptiveFieldDim[dim_i] > segmentDim[dim_i] :
            print("ERROR: [in function checkReceptiveFieldFineInComparisonToSegmentSize()] : The segment-size (input) should be at least as big as the receptive field of the model! This was not found to hold! Dimensions of Receptive Field:", receptiveFieldDim, ". Dimensions of Segment: ", segmentDim)
            return False
    return True

def checkKernDimPerLayerCorrect3dAndNumLayers(kernDimensionsPerLayer, numOfLayers) :
    #kernDimensionsPerLayer : a list with sublists. One sublist per layer. Each sublist should have 3 integers, specifying the dimensions of the kernel at the corresponding layer of the pathway. eg: kernDimensionsPerLayer = [ [3,3,3], [3,3,3], [5,5,5] ] 
    if kernDimensionsPerLayer == None or len(kernDimensionsPerLayer) != numOfLayers :
        return False
    for kernDimInLayer in kernDimensionsPerLayer :
        if len(kernDimInLayer) != 3 :
            return False
    return True

def checkSubsampleFactorEven(subFactor) :
    for dim_i in xrange(len(subFactor)) :
        if subFactor[dim_i]%2 != 1 :
            return False
    return True

