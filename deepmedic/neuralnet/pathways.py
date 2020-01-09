# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
import copy
from math import ceil

import tensorflow as tf

from deepmedic.neuralnet.pathwayTypes import PathwayTypes
from deepmedic.neuralnet.utils import calcRecFieldFromKernDimListPerLayerWhenStrides1
from deepmedic.neuralnet.blocks import ConvBlock, LowRankConvBlock


#################################################################
#                         Pathway Types                         #
#################################################################

def cropRczOf5DimArrayToMatchOther(array5DimToCrop, dimensionsOf5DimArrayToMatchInRcz):
    # dimensionsOf5DimArrayToMatchInRcz : [ batch size, num of fms, r, c, z] 
    output = array5DimToCrop[:,
                            :,
                            :dimensionsOf5DimArrayToMatchInRcz[2],
                            :dimensionsOf5DimArrayToMatchInRcz[3],
                            :dimensionsOf5DimArrayToMatchInRcz[4]]
    return output
    
def repeatRcz5DimArrayByFactor(array5Dim, factor3Dim):
    # array5Dim: [batch size, num of FMs, r, c, z]. Ala input/output of conv layers.
    # Repeat FM in the three last dimensions, to upsample back to the normal resolution space.
    # In numpy below: (but tf has no repeat, only tile, so, implementation is funny.
    #expandedR = array5Dim.repeat(factor3Dim[0], axis=2)
    #expandedRC = expandedR.repeat(factor3Dim[1], axis=3)
    #expandedRCZ = expandedRC.repeat(factor3Dim[2], axis=4)
    res = array5Dim
    res_shape = tf.shape(array5Dim) # Dynamic. For batch and r,c,z dimensions. (unknown prior to runtime)
    n_fms = array5Dim.get_shape()[1] # Static via get_shape(). Known. For reshape to return tensor with *known* shape[1].
    # If tf.shape()[1] is used, reshape changes res.get_shape()[1] to (?).
    
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1]*res_shape[2], 1, res_shape[3], res_shape[4]] ),
                               multiples=[1, 1, factor3Dim[0], 1, 1] ),
                    shape=[res_shape[0], n_fms, res_shape[2]*factor3Dim[0], res_shape[3], res_shape[4]] )
    res_shape = tf.shape(res)
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1], res_shape[2]*res_shape[3], 1, res_shape[4]] ),
                               multiples=[1, 1, 1, factor3Dim[1], 1] ),
                    shape=[res_shape[0], n_fms, res_shape[2], res_shape[3]*factor3Dim[1], res_shape[4]] )
    res_shape = tf.shape(res)
    res = tf.reshape( tf.tile( tf.reshape( res, shape=[res_shape[0], res_shape[1], res_shape[2], res_shape[3]*res_shape[4], 1] ),
                               multiples=[1, 1, 1, 1, factor3Dim[2]] ),
                    shape=[res_shape[0], n_fms, res_shape[2], res_shape[3], res_shape[4]*factor3Dim[2]] )
    return res
    
def upsampleRcz5DimArrayAndOptionalCrop(array5dimToUpsample,
                                        upsamplingFactor,
                                        upsamplingScheme="repeat",
                                        dimensionsOf5DimArrayToMatchInRcz=None) :
    # array5dimToUpsample : [batch_size, numberOfFms, r, c, z].
    if upsamplingScheme == "repeat" :
        upsampledOutput = repeatRcz5DimArrayByFactor(array5dimToUpsample, upsamplingFactor)
    else :
        print("ERROR: in upsampleRcz5DimArrayAndOptionalCrop(...). Not implemented type of upsampling! Exiting!"); exit(1)
        
    if dimensionsOf5DimArrayToMatchInRcz is not None:
        # If the central-voxels are eg 10, the susampled-part will have 4 central voxels. Which above will be repeated to 3*4 = 12.
        # I need to clip the last ones, to have the same dimension as the input from 1st pathway, which will have dimensions equal to the centrally predicted voxels (10)
        output = cropRczOf5DimArrayToMatchOther(upsampledOutput, dimensionsOf5DimArrayToMatchInRcz)
    else :
        output = upsampledOutput
        
    return output
    
def getMiddlePartOfFms(fms, listOfNumberOfCentralVoxelsToGetPerDimension) :
    # fms: a 5D tensor, [batch, fms, r, c, z]
    # listOfNumberOfCentralVoxelsToGetPerDimension: list of 3 scalars or Tensorflow 1D tensor (eg from tf.shape(x)). [r, c, z]
    fmsShape = tf.shape(fms) #fms.shape works too.
    # if part is of even width, one voxel to the left is the centre.
    rCentreOfPartIndex = (fmsShape[2] - 1) // 2
    rIndexToStartGettingCentralVoxels = rCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[0] - 1) // 2
    rIndexToStopGettingCentralVoxels = rIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[0]  # Excluding
    cCentreOfPartIndex = (fmsShape[3] - 1) // 2
    cIndexToStartGettingCentralVoxels = cCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[1] - 1) // 2
    cIndexToStopGettingCentralVoxels = cIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[1]  # Excluding
    zCentreOfPartIndex = (fmsShape[4] - 1) // 2
    zIndexToStartGettingCentralVoxels = zCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[2] - 1) // 2
    zIndexToStopGettingCentralVoxels = zIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[2]  # Excluding
    return fms[	:, :,
                rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels,
                zIndexToStartGettingCentralVoxels : zIndexToStopGettingCentralVoxels]

        
def makeResidualConnection(log, deeperLOut, earlierLOut) :
    # Add the outputs of the two layers and return the output, as well as its dimensions.
    # deeperLOut & earlierLOut: 5D tensors [batchsize, chans, x, y, z], outputs of deepest and earliest layer of the Res.Conn.
    # Result: Shape of result should be exactly the same as the output of Deeper layer.
    deeperLOutShape = tf.shape(deeperLOut)
    earlierLOutShape = tf.shape(earlierLOut)
    # Get part of the earlier layer that is of the same dimensions as the FMs of the deeper:
    partOfEarlierFmsToAddTrain = getMiddlePartOfFms(earlierLOut, deeperLOutShape[2:])
    # Add the FMs, after taking care of zero padding if the deeper layer has more FMs.
    if deeperLOut.get_shape()[1] >= earlierLOut.get_shape()[1] : # ifs not allowed via tensor (from tf.shape(...))
        zeroFmsToConcatTrain = tf.zeros(shape=[deeperLOutShape[0],
                                               deeperLOutShape[1] - earlierLOutShape[1],
                                               deeperLOutShape[2], deeperLOutShape[3], deeperLOutShape[4]], dtype="float32")
        outputOfResConnTrain = deeperLOut + tf.concat( [partOfEarlierFmsToAddTrain, zeroFmsToConcatTrain], axis=1)

    else : # Deeper FMs are fewer than earlier. This should not happen in most architectures. But oh well...
        outputOfResConnTrain = deeperLOut + partOfEarlierFmsToAddTrain[:, :deeperLOutShape[1], :,:,:]
        
    # Dimensions of output are the same as those of the deeperLayer
    return outputOfResConnTrain
    
    
#################################################################
#                        Classes of Pathways                    #
#################################################################
        
class Pathway(object):
    # This is a virtual class.
    
    def __init__(self, pName=None) :
        self._pName = pName
        self._pType = None # Pathway Type.
        
        # === Input to the pathway ===
        self._input = {"train": None, "val": None, "test": None}
        
        # === Basic architecture parameters === 
        self._layersInPathway = []
        self._subsFactor = [1,1,1]
        self._recField = None # At the end of pathway
        
        # === Output of the block ===
        self._output = {"train": None, "val": None, "test": None}
        
    def makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(self,
                                                    log,
                                                    rng,
                                                    inputTrain,
                                                    inputVal,
                                                    inputTest,
                                                    
                                                    numKernsPerLayer,
                                                    kernelDimsPerLayer,
                                                    
                                                    convWInitMethod,
                                                    useBnPerLayer, # As a flag for case that I want to apply BN on input image. I want to apply to input of FC.
                                                    movingAvForBnOverXBatches,
                                                    activFuncPerLayer,
                                                    dropoutRatesPerLayer=[],
                                                    
                                                    poolingParamsStructureForThisPathwayType = [],
                                                    
                                                    indicesOfLowerRankLayersForPathway=[],
                                                    ranksOfLowerRankLayersForPathway = [],
                                                    indicesOfLayersToConnectResidualsInOutputForPathway=[]
                                                    ) :
        log.print3("[Pathway_" + str(self.getStringType()) + "] is being built...")
        
        self._recField = self.calcRecFieldOfPathway(kernelDimsPerLayer)
        
        self._setInputAttributes(inputTrain, inputVal, inputTest)                
        log.print3("\t[Pathway_"+str(self.getStringType())+"]: Input's Shape: (Train) " + str(self._input["train"].shape) + \
                ", (Val) " + str(self._input["val"].shape) + ", (Test) " + str(self._input["test"].shape))
        
        inputToNextLayerTrain = self._input["train"]; inputToNextLayerVal = self._input["val"]; inputToNextLayerTest = self._input["test"]
        numOfLayers = len(numKernsPerLayer)
        for layer_i in range(0, numOfLayers) :
            thisLayerFilterShape = [numKernsPerLayer[layer_i], inputToNextLayerTrain.shape[1]] + kernelDimsPerLayer[layer_i]
            
            thisLayerUseBn = useBnPerLayer[layer_i]
            thisLayerActivFunc = activFuncPerLayer[layer_i]
            thisLayerDropoutRate = dropoutRatesPerLayer[layer_i] if dropoutRatesPerLayer else 0
            
            thisLayerPoolingParameters = poolingParamsStructureForThisPathwayType[layer_i]
            
            log.print3("\t[Conv.Layer_" + str(layer_i) + "], Filter Shape: " + str(thisLayerFilterShape))
            log.print3("\t[Conv.Layer_" + str(layer_i) + "], Input's Shape: (Train) " + str(inputToNextLayerTrain.shape) + \
                            ", (Val) " + str(inputToNextLayerVal.shape) + ", (Test) " + str(inputToNextLayerTest.shape))
            
            if layer_i in indicesOfLowerRankLayersForPathway :
                layer = LowRankConvBlock(ranksOfLowerRankLayersForPathway[ indicesOfLowerRankLayersForPathway.index(layer_i) ])
            else : # normal conv layer
                layer = ConvBlock()
            layer.makeLayer(rng,
                            inputToLayerTrain=inputToNextLayerTrain,
                            inputToLayerVal=inputToNextLayerVal,
                            inputToLayerTest=inputToNextLayerTest,
                            
                            filterShape=thisLayerFilterShape,
                            poolingParameters=thisLayerPoolingParameters,
                            convWInitMethod=convWInitMethod,
                            useBnFlag = thisLayerUseBn,
                            movingAvForBnOverXBatches=movingAvForBnOverXBatches,
                            activationFunc=thisLayerActivFunc,
                            dropoutRate=thisLayerDropoutRate
                            ) 
            self._layersInPathway.append(layer)
            
            if layer_i not in indicesOfLayersToConnectResidualsInOutputForPathway : #not a residual connecting here
                inputToNextLayerTrain = layer.output["train"]
                inputToNextLayerVal = layer.output["val"]
                inputToNextLayerTest = layer.output["test"]
            else : #make residual connection
                log.print3("\t[Pathway_"+str(self.getStringType())+"]: making Residual Connection between output of [Layer_"+str(layer_i)+"] to input of previous layer.")
                assert layer_i > 0 # The very first layer (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                earlierLayer = self._layersInPathway[layer_i-1]
                
                inputToNextLayerTrain = makeResidualConnection(log, layer.output["train"], earlierLayer.input["train"])
                inputToNextLayerVal = makeResidualConnection(log, layer.output["val"], earlierLayer.input["val"])
                inputToNextLayerTest = makeResidualConnection(log, layer.output["test"], earlierLayer.input["test"])
                layer.outputAfterResidualConnIfAnyAtOutp["train"] = inputToNextLayerTrain
                layer.outputAfterResidualConnIfAnyAtOutp["val"] = inputToNextLayerVal
                layer.outputAfterResidualConnIfAnyAtOutp["test"] = inputToNextLayerTest
        
        self._setOutputAttributes(inputToNextLayerTrain, inputToNextLayerVal, inputToNextLayerTest)
        
        log.print3("\t[Pathway_"+str(self.getStringType())+"]: Output's Shape: (Train) " + str(self._output["train"].shape) + \
                 		", (Val) " + str(self._output["val"].shape) + ", (Test) " + str(self._output["test"].shape))
        
        log.print3("[Pathway_" + str(self.getStringType()) + "] done.")
        
        
    # The below should be updated, and calculated in here properly with private function and per layer.
    def calcRecFieldOfPathway(self, kernelDimsPerLayer) :
        return calcRecFieldFromKernDimListPerLayerWhenStrides1(kernelDimsPerLayer)
        
    def calcInputRczDimsToProduceOutputFmsOfCompatibleDims(self, thisPathWayKernelDims, dimsOfOutputFromPrimaryPathway):
        recFieldAtEndOfPathway = self.calcRecFieldOfPathway(thisPathWayKernelDims)
        rczDimsOfInputToPathwayShouldBe = [-1,-1,-1]
        rczDimsOfOutputOfPathwayShouldBe = [-1,-1,-1]
        
        rczDimsOfOutputFromPrimaryPathway = dimsOfOutputFromPrimaryPathway[2:]
        for rcz_i in range(3) :
            rczDimsOfOutputOfPathwayShouldBe[rcz_i] = int(ceil(rczDimsOfOutputFromPrimaryPathway[rcz_i]/(1.0*self.subsFactor()[rcz_i])))
            rczDimsOfInputToPathwayShouldBe[rcz_i] = recFieldAtEndOfPathway[rcz_i] + rczDimsOfOutputOfPathwayShouldBe[rcz_i] - 1
        return rczDimsOfInputToPathwayShouldBe
        
    # Setters
    def _setInputAttributes(self, inputToLayerTrain, inputToLayerVal, inputToLayerTest):
        self._input["train"] = inputToLayerTrain; self._input["val"] = inputToLayerVal; self._input["test"] = inputToLayerTest
        
    def _setOutputAttributes(self, outputTrain, outputVal, outputTest):
        self._output["train"] = outputTrain; self._output["val"] = outputVal; self._output["test"] = outputTest
        
    # Getters
    def pName(self):
        return self._pName
    def pType(self):
        return self._pType
    def getLayers(self):
        return self._layersInPathway
    def getLayer(self, index):
        return self._layersInPathway[index]
    def subsFactor(self):
        return self._subsFactor
    def getOutput(self):
        return [ self._output["train"], self._output["val"], self._output["test"] ]
    def getShapeOfOutput(self):
        return [ self._output["train"].shape, self._output["val"].shape, self._output["test"].shape ]
    def getShapeOfInput(self, train_val_test_str):
        assert train_val_test_str in ["train", "val", "test"]
        return self._input[train_val_test_str].shape
    
    # Other API :
    def getStringType(self) : raise NotImplementedMethod() # Abstract implementation. Children classes should implement this.
    # Will be overriden for lower-resolution pathways.
    def getOutputAtNormalRes(self): return self.getOutput()
    
class NormalPathway(Pathway):
    def __init__(self, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.NORM
    # Override parent's abstract classes.
    def getStringType(self) :
        return "NORMAL"
        
class SubsampledPathway(Pathway):
    def __init__(self, subsamplingFactor, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.SUBS
        self._subsFactor = subsamplingFactor
        
        self._outputNormRes = {"train": None, "val": None, "test": None}
        
    def upsampleOutputToNormalRes(self, upsamplingScheme="repeat",
                            shapeToMatchInRczTrain=None, shapeToMatchInRczVal=None, shapeToMatchInRczTest=None):
        #should be called only once to build. Then just call getters if needed to get upsampled layer again.
        [outputTrain, outputVal, outputTest] = self.getOutput()
        
        outputNormResTrain = upsampleRcz5DimArrayAndOptionalCrop(outputTrain,
                                                                self.subsFactor(),
                                                                upsamplingScheme,
                                                                shapeToMatchInRczTrain)
        outputNormResVal = upsampleRcz5DimArrayAndOptionalCrop(	outputVal,
                                                                self.subsFactor(),
                                                                upsamplingScheme,
                                                                shapeToMatchInRczVal)
        outputNormResTest = upsampleRcz5DimArrayAndOptionalCrop(outputTest,
                                                                self.subsFactor(),
                                                                upsamplingScheme,
                                                                shapeToMatchInRczTest)
        
        self._setOutputAttributesNormRes(outputNormResTrain, outputNormResVal, outputNormResTest)
        
    def _setOutputAttributesNormRes(self, outputNormResTrain, outputNormResVal, outputNormResTest):
        #Essentially this is after the upsampling "layer"
        self._outputNormRes["train"] = outputNormResTrain; self._outputNormRes["val"] = outputNormResVal; self._outputNormRes["test"] = outputNormResTest
        
        
    # OVERRIDING parent's classes.
    def getStringType(self) :
        return "SUBSAMPLED" + str(self.subsFactor())
        
    def getOutputAtNormalRes(self):
        # upsampleOutputToNormalRes() must be called first once.
        return [ self._outputNormRes["train"], self._outputNormRes["val"], self._outputNormRes["test"] ]
            
             
class FcPathway(Pathway):
    def __init__(self, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.FC
    # Override parent's abstract classes.
    def getStringType(self) :
        return "FC"



