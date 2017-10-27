# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import numpy
import numpy as np
import copy

from math import ceil

import theano.tensor as T

from deepmedic.pathwayTypes import PathwayTypes
from deepmedic.cnnLayerTypes import ConvLayer, LowRankConvLayer
from deepmedic.cnnHelpers import calculateReceptiveFieldDimensionsFromKernelsDimListPerLayerForFullyConvCnnWithStrides1

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
    expandedR = array5Dim.repeat(factor3Dim[0], axis=2)
    expandedRC = expandedR.repeat(factor3Dim[1], axis=3)
    expandedRCZ = expandedRC.repeat(factor3Dim[2], axis=4)
    return expandedRCZ
    
def upsampleRcz5DimArrayAndOptionalCrop(array5dimToUpsample,
                                        upsamplingFactor,
                                        upsamplingScheme="repeat",
                                        dimensionsOf5DimArrayToMatchInRcz=None) :
    # array5dimToUpsample : [batch_size, numberOfFms, r, c, z].
    if upsamplingScheme == "repeat" :
        upsampledOutput = repeatRcz5DimArrayByFactor(array5dimToUpsample, upsamplingFactor)
    else :
        print("NOT IMPLEMENTED! EXITING!"); exit(1)
        
    if dimensionsOf5DimArrayToMatchInRcz != None :
        # If the central-voxels are eg 10, the susampled-part will have 4 central voxels. Which above will be repeated to 3*4 = 12.
        # I need to clip the last ones, to have the same dimension as the input from 1st pathway, which will have dimensions equal to the centrally predicted voxels (10)
        output = cropRczOf5DimArrayToMatchOther(upsampledOutput, dimensionsOf5DimArrayToMatchInRcz)
    else :
        output = upsampledOutput
        
    return output
    
def getMiddlePartOfFms(fms, listOfNumberOfCentralVoxelsToGetPerDimension) :
    # fms: a 5D tensor, [batch, fms, r, c, z]
    fmsShape = T.shape(fms) #fms.shape works too, but this is clearer theano grammar.
    # if part is of even width, one voxel to the left is the centre.
    rCentreOfPartIndex = (fmsShape[2] - 1) / 2
    rIndexToStartGettingCentralVoxels = rCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[0] - 1) / 2
    rIndexToStopGettingCentralVoxels = rIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[0]  # Excluding
    cCentreOfPartIndex = (fmsShape[3] - 1) / 2
    cIndexToStartGettingCentralVoxels = cCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[1] - 1) / 2
    cIndexToStopGettingCentralVoxels = cIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[1]  # Excluding
    
    if len(listOfNumberOfCentralVoxelsToGetPerDimension) == 2:  # the input FMs are of 2 dimensions (for future use)
        return fms[	:, :,
                    rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                    cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels]
    elif len(listOfNumberOfCentralVoxelsToGetPerDimension) == 3 :  # the input FMs are of 3 dimensions
        zCentreOfPartIndex = (fmsShape[4] - 1) / 2
        zIndexToStartGettingCentralVoxels = zCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[2] - 1) / 2
        zIndexToStopGettingCentralVoxels = zIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[2]  # Excluding
        return fms[	:, :,
                    rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                    cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels,
                    zIndexToStartGettingCentralVoxels : zIndexToStopGettingCentralVoxels]
    else :  # wrong number of dimensions!
        return -1
        
def makeResidualConnectionBetweenLayersAndReturnOutput( myLogger,
                                                        deeperLayerOutputImagesTrValTest,
                                                        deeperLayerOutputImageShapesTrValTest,
                                                        earlierLayerOutputImagesTrValTest,
                                                        earlierLayerOutputImageShapesTrValTest) :
    # Add the outputs of the two layers and return the output, as well as its dimensions.
    # Result: The result should have exactly the same shape as the output of the Deeper layer. Both #FMs and Dimensions of FMs.
    
    (deeperLayerOutputImageTrain, deeperLayerOutputImageVal, deeperLayerOutputImageTest) = deeperLayerOutputImagesTrValTest
    (deeperLayerOutputImageShapeTrain, deeperLayerOutputImageShapeVal, deeperLayerOutputImageShapeTest) = deeperLayerOutputImageShapesTrValTest
    (earlierLayerOutputImageTrain, earlierLayerOutputImageVal, earlierLayerOutputImageTest) = earlierLayerOutputImagesTrValTest
    (earlierLayerOutputImageShapeTrain, earlierLayerOutputImageShapeVal, earlierLayerOutputImageShapeTest) = earlierLayerOutputImageShapesTrValTest
    # Note: deeperLayerOutputImageShapeTrain has dimensions: [batchSize, FMs, r, c, z]    
    # The deeper FMs can be greater only when there is upsampling. But then, to do residuals, I would need to upsample the earlier FMs. Not implemented.
    if np.any(np.asarray(deeperLayerOutputImageShapeTrain[2:]) > np.asarray(earlierLayerOutputImageShapeTrain[2:])) or \
            np.any(np.asarray(deeperLayerOutputImageShapeVal[2:]) > np.asarray(earlierLayerOutputImageShapeVal[2:])) or \
                np.any(np.asarray(deeperLayerOutputImageShapeTest[2:]) > np.asarray(earlierLayerOutputImageShapeTest[2:])) :
        myLogger.print3("ERROR: In function [makeResidualConnectionBetweenLayersAndReturnOutput] the RCZ-dimensions of a deeper layer FMs were found greater than the earlier layers. Not implemented functionality. Exiting!")
        myLogger.print3("\t (train) Dimensions of Deeper Layer=" + str(deeperLayerOutputImageShapeTrain) + ". Dimensions of Earlier Layer=" + str(earlierLayerOutputImageShapeTrain) )
        myLogger.print3("\t (val) Dimensions of Deeper Layer=" + str(deeperLayerOutputImageShapeVal) + ". Dimensions of Earlier Layer=" + str(earlierLayerOutputImageShapeVal) )
        myLogger.print3("\t (test) Dimensions of Deeper Layer=" + str(deeperLayerOutputImageShapeTest) + ". Dimensions of Earlier Layer=" + str(earlierLayerOutputImageShapeTest) )
        exit(1)
        
    # get the part of the earlier layer that is of the same dimensions as the FMs of the deeper:
    partOfEarlierFmsToAddTrain = getMiddlePartOfFms(earlierLayerOutputImageTrain, deeperLayerOutputImageShapeTrain[2:])
    partOfEarlierFmsToAddVal = getMiddlePartOfFms(earlierLayerOutputImageVal, deeperLayerOutputImageShapeVal[2:])
    partOfEarlierFmsToAddTest = getMiddlePartOfFms(earlierLayerOutputImageTest, deeperLayerOutputImageShapeTest[2:])
    
    # Add the FMs, after taking care of zero padding if the deeper layer has more FMs.
    numFMsDeeper = deeperLayerOutputImageShapeTrain[1]
    numFMsEarlier = earlierLayerOutputImageShapeTrain[1]
    if numFMsDeeper >= numFMsEarlier :
        outputOfResConnTrain = T.inc_subtensor(deeperLayerOutputImageTrain[:, :numFMsEarlier, :,:,:], partOfEarlierFmsToAddTrain, inplace=False)
        outputOfResConnVal = T.inc_subtensor(deeperLayerOutputImageVal[:, :numFMsEarlier, :,:,:], partOfEarlierFmsToAddVal, inplace=False)
        outputOfResConnTest = T.inc_subtensor(deeperLayerOutputImageTest[:, :numFMsEarlier, :,:,:], partOfEarlierFmsToAddTest, inplace=False)
    else : # Deeper FMs are fewer than earlier. This should not happen in most architectures. But oh well...
        outputOfResConnTrain = deeperLayerOutputImageTrain + partOfEarlierFmsToAddTrain[:, :numFMsDeeper, :,:,:]
        outputOfResConnVal = deeperLayerOutputImageVal + partOfEarlierFmsToAddVal[:, :numFMsDeeper, :,:,:]
        outputOfResConnTest = deeperLayerOutputImageTest + partOfEarlierFmsToAddTest[:, :numFMsDeeper, :,:,:]
        
    # Dimensions of output are the same as those of the deeperLayer
    return (outputOfResConnTrain, outputOfResConnVal, outputOfResConnTest)
    
    
#################################################################
#                        Classes of Pathways                    #
#################################################################
        
class Pathway(object):
    # This is a virtual class.
    
    def __init__(self, pName=None) :
        self._pName = pName
        self._pType = None # Pathway Type.
        
        # === Input to the pathway ===
        self._inputTrain = None
        self._inputVal = None
        self._inputTest = None
        self._inputShapeTrain = None
        self._inputShapeVal = None
        self._inputShapeTest = None
        
        # === Basic architecture parameters === 
        self._layersInPathway = []
        self._subsFactor = [1,1,1]
        self._recField = None # At the end of pathway
        
        # === Output of the block ===
        self._outputTrain = None
        self._outputVal = None
        self._outputTest = None
        self._outputShapeTrain = None
        self._outputShapeVal = None
        self._outputShapeTest = None
        
    def makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(self,
                                                    myLogger,
                                                    
                                                    inputTrain,
                                                    inputVal,
                                                    inputTest,
                                                    inputDimsTrain,
                                                    inputDimsVal,
                                                    inputDimsTest,
                                                    
                                                    numKernsPerLayer,
                                                    kernelDimsPerLayer,
                                                    
                                                    initializationTechniqueClassic0orDelvingInto1,
                                                    useBnPerLayer, # As a flag for case that I want to apply BN on input image. I want to apply to input of FC.
                                                    rollingAverageForBatchNormalizationOverThatManyBatches,
                                                    activFuncPerLayer,
                                                    dropoutRatesPerLayer=[],
                                                    
                                                    poolingParamsStructureForThisPathwayType = [],
                                                    
                                                    indicesOfLowerRankLayersForPathway=[],
                                                    ranksOfLowerRankLayersForPathway = [],
                                                    
                                                    indicesOfLayersToConnectResidualsInOutputForPathway=[]
                                                    ) :
        rng = numpy.random.RandomState(55789)
        myLogger.print3("[Pathway_" + str(self.getStringType()) + "] is being built...")
        
        self._recField = self.calcRecFieldOfPathway(kernelDimsPerLayer)
        
        self._setInputAttributes(inputTrain, inputVal, inputTest, inputDimsTrain, inputDimsVal, inputDimsTest)                
        myLogger.print3("\t[Pathway_"+str(self.getStringType())+"]: Input's Shape: (Train) " + str(self._inputShapeTrain) + \
                ", (Val) " + str(self._inputShapeVal) + ", (Test) " + str(self._inputShapeTest))
        
        inputToNextLayerTrain = self._inputTrain; inputToNextLayerVal = self._inputVal; inputToNextLayerTest = self._inputTest
        inputToNextLayerShapeTrain = self._inputShapeTrain; inputToNextLayerShapeVal = self._inputShapeVal; inputToNextLayerShapeTest = self._inputShapeTest
        numOfLayers = len(numKernsPerLayer)
        for layer_i in range(0, numOfLayers) :
            thisLayerFilterShape = [numKernsPerLayer[layer_i],inputToNextLayerShapeTrain[1]] + kernelDimsPerLayer[layer_i]
            
            thisLayerUseBn = useBnPerLayer[layer_i]
            thisLayerActivFunc = activFuncPerLayer[layer_i] # 0 relu, 1 prelu, -1 linear (no activ, for 1st layer over raw input).
            thisLayerDropoutRate = dropoutRatesPerLayer[layer_i] if dropoutRatesPerLayer else 0
            
            thisLayerPoolingParameters = poolingParamsStructureForThisPathwayType[layer_i]
            
            myLogger.print3("\t[Conv.Layer_" + str(layer_i) + "], Filter Shape: " + str(thisLayerFilterShape))
            myLogger.print3("\t[Conv.Layer_" + str(layer_i) + "], Input's Shape: (Train) " + str(inputToNextLayerShapeTrain) + \
                            ", (Val) " + str(inputToNextLayerShapeVal) + ", (Test) " + str(inputToNextLayerShapeTest))
            
            if layer_i in indicesOfLowerRankLayersForPathway :
                layer = LowRankConvLayer(ranksOfLowerRankLayersForPathway[ indicesOfLowerRankLayersForPathway.index(layer_i) ])
            else : # normal conv layer
                layer = ConvLayer()
            layer.makeLayer(rng,
                            inputToLayerTrain=inputToNextLayerTrain,
                            inputToLayerVal=inputToNextLayerVal,
                            inputToLayerTest=inputToNextLayerTest,
                            inputToLayerShapeTrain=inputToNextLayerShapeTrain,
                            inputToLayerShapeVal=inputToNextLayerShapeVal,
                            inputToLayerShapeTest=inputToNextLayerShapeTest,
                            
                            filterShape=thisLayerFilterShape,
                            poolingParameters=thisLayerPoolingParameters,
                            initializationTechniqueClassic0orDelvingInto1=initializationTechniqueClassic0orDelvingInto1,
                            useBnFlag = thisLayerUseBn,
                            rollingAverageForBatchNormalizationOverThatManyBatches=rollingAverageForBatchNormalizationOverThatManyBatches,
                            activationFunctionToUseRelu0orPrelu1orMinus1ForLinear=thisLayerActivFunc,
                            dropoutRate=thisLayerDropoutRate
                            ) 
            self._layersInPathway.append(layer)
            
            if layer_i not in indicesOfLayersToConnectResidualsInOutputForPathway : #not a residual connecting here
                inputToNextLayerTrain = layer.outputTrain
                inputToNextLayerVal = layer.outputVal
                inputToNextLayerTest = layer.outputTest
            else : #make residual connection
                myLogger.print3("\t[Pathway_"+str(self.getStringType())+"]: making Residual Connection between output of [Layer_"+str(layer_i)+"] to input of previous layer.")
                deeperLayerOutputImagesTrValTest = (layer.outputTrain, layer.outputVal, layer.outputTest)
                deeperLayerOutputImageShapesTrValTest = (layer.outputShapeTrain, layer.outputShapeVal, layer.outputShapeTest)
                assert layer_i > 0 # The very first layer (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                earlierLayer = self._layersInPathway[layer_i-1]
                earlierLayerOutputImagesTrValTest = (earlierLayer.inputTrain, earlierLayer.inputVal, earlierLayer.inputTest)
                earlierLayerOutputImageShapesTrValTest = (earlierLayer.inputShapeTrain, earlierLayer.inputShapeVal, earlierLayer.inputShapeTest)
                
                (inputToNextLayerTrain,
                inputToNextLayerVal,
                inputToNextLayerTest) = makeResidualConnectionBetweenLayersAndReturnOutput( myLogger,
															                                deeperLayerOutputImagesTrValTest,
															                                deeperLayerOutputImageShapesTrValTest,
															                                earlierLayerOutputImagesTrValTest,
															                                earlierLayerOutputImageShapesTrValTest )
                layer.outputAfterResidualConnIfAnyAtOutpTrain = inputToNextLayerTrain
                layer.outputAfterResidualConnIfAnyAtOutpVal = inputToNextLayerVal
                layer.outputAfterResidualConnIfAnyAtOutpTest = inputToNextLayerTest
            # Residual connections preserve the both the number of FMs and the dimensions of the FMs, the same as in the later, deeper layer.
            inputToNextLayerShapeTrain = layer.outputShapeTrain
            inputToNextLayerShapeVal = layer.outputShapeVal
            inputToNextLayerShapeTest = layer.outputShapeTest
        
        self._setOutputAttributes(inputToNextLayerTrain, inputToNextLayerVal, inputToNextLayerTest,
                                inputToNextLayerShapeTrain, inputToNextLayerShapeVal, inputToNextLayerShapeTest)
        
        myLogger.print3("\t[Pathway_"+str(self.getStringType())+"]: Output's Shape: (Train) " + str(self._outputShapeTrain) + \
                 		", (Val) " + str(self._outputShapeVal) + ", (Test) " + str(self._outputShapeTest))
        
        myLogger.print3("[Pathway_" + str(self.getStringType()) + "] done.")
        
    # Skip connections to end of pathway.
    def makeMultiscaleConnectionsForLayerType(self, convLayersToConnectToFirstFcForMultiscaleFromThisLayerType) :
    	
        layersInThisPathway = self.getLayers()
        
        [outputOfPathwayTrain, outputOfPathwayVal, outputOfPathwayTest ] = self.getOutput()
        [outputShapeTrain, outputShapeVal, outputShapeTest] = self.getShapeOfOutput()
        numOfCentralVoxelsToGetTrain = outputShapeTrain[2:]; numOfCentralVoxelsToGetVal = outputShapeVal[2:]; numOfCentralVoxelsToGetTest = outputShapeTest[2:]
        
        for convLayer_i in convLayersToConnectToFirstFcForMultiscaleFromThisLayerType :
            thisLayer = layersInThisPathway[convLayer_i]
                    
            middlePartOfFmsTrain = getMiddlePartOfFms(thisLayer.outputTrain, numOfCentralVoxelsToGetTrain)
            middlePartOfFmsVal = getMiddlePartOfFms(thisLayer.outputVal, numOfCentralVoxelsToGetVal)
            middlePartOfFmsTest = getMiddlePartOfFms(thisLayer.outputTest, numOfCentralVoxelsToGetTest)
            
            outputOfPathwayTrain = T.concatenate([outputOfPathwayTrain, middlePartOfFmsTrain], axis=1)
            outputOfPathwayVal = T.concatenate([outputOfPathwayVal, middlePartOfFmsVal], axis=1)
            outputOfPathwayTest = T.concatenate([outputOfPathwayTest, middlePartOfFmsTest], axis=1)
            outputShapeTrain[1] += thisLayer.getNumberOfFeatureMaps(); outputShapeVal[1] += thisLayer.getNumberOfFeatureMaps(); outputShapeTest[1] += thisLayer.getNumberOfFeatureMaps(); 
            
        self._setOutputAttributes(outputOfPathwayTrain, outputOfPathwayVal, outputOfPathwayTest,
                                outputShapeTrain, outputShapeVal, outputShapeTest)
        
    # The below should be updated, and calculated in here properly with private function and per layer.
    def calcRecFieldOfPathway(self, kernelDimsPerLayer) :
        return calculateReceptiveFieldDimensionsFromKernelsDimListPerLayerForFullyConvCnnWithStrides1(kernelDimsPerLayer)
        
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
    def _setInputAttributes(self, inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain, inputToLayerShapeVal, inputToLayerShapeTest) :
        self._inputTrain = inputToLayerTrain; self._inputVal = inputToLayerVal; self._inputTest = inputToLayerTest
        self._inputShapeTrain = inputToLayerShapeTrain; self._inputShapeVal = inputToLayerShapeVal; self._inputShapeTest = inputToLayerShapeTest
        
    def _setOutputAttributes(self, outputTrain, outputVal, outputTest, outputShapeTrain, outputShapeVal, outputShapeTest) :
        self._outputTrain = outputTrain; self._outputVal = outputVal; self._outputTest = outputTest
        self._outputShapeTrain = outputShapeTrain; self._outputShapeVal = outputShapeVal; self._outputShapeTest = outputShapeTest
        
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
        return [ self._outputTrain, self._outputVal, self._outputTest ]
    def getShapeOfOutput(self):
        return [ self._outputShapeTrain, self._outputShapeVal, self._outputShapeTest ]
    def getShapeOfInput(self):
        return [ self._inputShapeTrain, self._inputShapeVal, self._inputShapeTest ]
        
    # Other API :
    def getStringType(self) : raise NotImplementedMethod() # Abstract implementation. Children classes should implement this.
    # Will be overriden for lower-resolution pathways.
    def getOutputAtNormalRes(self): return self.getOutput()
    def getShapeOfOutputAtNormalRes(self): return self.getShapeOfOutput()
    
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
        
        self._outputNormResTrain = None
        self._outputNormResVal = None
        self._outputNormResTest = None
        self._outputNormResShapeTrain = None
        self._outputNormResShapeVal = None
        self._outputNormResShapeTest = None
        
    def upsampleOutputToNormalRes(self, upsamplingScheme="repeat",
                            shapeToMatchInRczTrain=None, shapeToMatchInRczVal=None, shapeToMatchInRczTest=None):
        #should be called only once to build. Then just call getters if needed to get upsampled layer again.
        [outputTrain, outputVal, outputTest] = self.getOutput()
        [outputShapeTrain, outputShapeVal, outputShapeTest] = self.getShapeOfOutput()
        
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
        
        outputNormResShapeTrain = outputShapeTrain[:2] + shapeToMatchInRczTrain[2:]
        outputNormResShapeVal = outputShapeVal[:2] + shapeToMatchInRczVal[2:]
        outputNormResShapeTest = outputShapeTest[:2] + shapeToMatchInRczTest[2:]
        
        self._setOutputAttributesNormRes(outputNormResTrain, outputNormResVal, outputNormResTest,
                                outputNormResShapeTrain, outputNormResShapeVal, outputNormResShapeTest)
        
    def _setOutputAttributesNormRes(self, outputNormResTrain, outputNormResVal, outputNormResTest,
                                    outputNormResShapeTrain, outputNormResShapeVal, outputNormResShapeTest) :
        #Essentially this is after the upsampling "layer"
        self._outputNormResTrain = outputNormResTrain; self._outputNormResVal = outputNormResVal; self._outputNormResTest = outputNormResTest
        self._outputNormResShapeTrain = outputNormResShapeTrain; self._outputNormResShapeVal = outputNormResShapeVal; self._outputNormResShapeTest = outputNormResShapeTest
        
        
    # OVERRIDING parent's classes.
    def getStringType(self) :
        return "SUBSAMPLED" + str(self.subsFactor())
        
    def getOutputAtNormalRes(self):
        # upsampleOutputToNormalRes() must be called first once.
        return [ self._outputNormResTrain, self._outputNormResVal, self._outputNormResTest ]
        
    def getShapeOfOutputAtNormalRes(self):
        # upsampleOutputToNormalRes() must be called first once.
        return [ self._outputNormResShapeTrain, self._outputNormResShapeVal, self._outputNormResShapeTest ]
        
             
class FcPathway(Pathway):
    def __init__(self, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.FC
    # Override parent's abstract classes.
    def getStringType(self) :
        return "FC"



