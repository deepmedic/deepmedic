# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import random
from math import floor
from math import ceil

from deepmedic import cnnLayerTypes

#-----helper functions that I use in here---
def getListOfNumberOfCentralVoxelsClassifiedPerDimension(imagePartDimensions, patchDimensions) :
    numberOfCentralVoxelsClassifiedPerDimension = []
    for dimension_i in xrange(0, len(imagePartDimensions)) :
        numberOfCentralVoxelsClassifiedPerDimension.append(imagePartDimensions[dimension_i] - patchDimensions[dimension_i] + 1)
    return numberOfCentralVoxelsClassifiedPerDimension

def getMiddlePartOfFms(fms, fmsShape, listOfNumberOfCentralVoxelsToGetPerDimension) :
    #if part is of even width, one voxel to the left is the centre.
    rCentreOfPartIndex = (fmsShape[2] - 1) / 2
    rIndexToStartGettingCentralVoxels = rCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[0]-1)/2
    rIndexToStopGettingCentralVoxels = rIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[0] #Excluding
    cCentreOfPartIndex = (fmsShape[3] - 1) / 2
    cIndexToStartGettingCentralVoxels = cCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[1]-1)/2
    cIndexToStopGettingCentralVoxels = cIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[1] #Excluding
    
    if len(listOfNumberOfCentralVoxelsToGetPerDimension) == 2: #the input FMs are of 2 dimensions (for future use)
        return fms[ :,:,
                    rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                    cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels]
    elif len(listOfNumberOfCentralVoxelsToGetPerDimension) ==3 :  #the input FMs are of 3 dimensions
        zCentreOfPartIndex = (fmsShape[4] - 1) / 2
        zIndexToStartGettingCentralVoxels = zCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[2]-1)/2
        zIndexToStopGettingCentralVoxels = zIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[2] #Excluding
        return fms[ :, :,
                    rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                    cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels,
                    zIndexToStartGettingCentralVoxels : zIndexToStopGettingCentralVoxels]
    else : #wrong number of dimensions!
        return -1
    
    
def padImageWithMirroring( inputImage, inputImageDimensions, voxelsPerDimToPad ) :
    # inputImage shape: [batchSize, #channels#, r, c, z]
    # inputImageDimensions : [ batchSize, #channels, dim r, dim c, dim z ] of inputImage
    # voxelsPerDimToPad shape: [ num o voxels in r-dim to add, ...c-dim, ...z-dim ]
    # If voxelsPerDimToPad is odd, 1 more voxel is added to the right side.
    # r-axis
    assert np.all(voxelsPerDimToPad) >= 0
    padLeft = int(voxelsPerDimToPad[0]/2); padRight = int((voxelsPerDimToPad[0]+1)/2);
    paddedImage = T.concatenate([inputImage[:,:, int(voxelsPerDimToPad[0]/2)-1::-1 ,:,:], inputImage], axis=2) if padLeft >0 else inputImage
    paddedImage = T.concatenate([paddedImage, paddedImage[ :, :, -1:-1-int((voxelsPerDimToPad[0]+1)/2):-1, :, :]], axis=2) if padRight >0 else paddedImage
    # c-axis
    padLeft = int(voxelsPerDimToPad[1]/2); padRight = int((voxelsPerDimToPad[1]+1)/2);
    paddedImage = T.concatenate([paddedImage[:,:,:, padLeft-1::-1 ,:], paddedImage], axis=3) if padLeft >0 else paddedImage
    paddedImage = T.concatenate([paddedImage, paddedImage[:,:,:, -1:-1-padRight:-1,:]], axis=3) if padRight >0 else paddedImage
    # z-axis
    padLeft = int(voxelsPerDimToPad[2]/2); padRight = int((voxelsPerDimToPad[2]+1)/2)
    paddedImage = T.concatenate([paddedImage[:,:,:,:, padLeft-1::-1 ], paddedImage], axis=4) if padLeft >0 else paddedImage
    paddedImage = T.concatenate([paddedImage, paddedImage[:,:,:,:, -1:-1-padRight:-1]], axis=4) if padRight >0 else paddedImage
    
    newDimensions = [inputImageDimensions[0],
                     inputImageDimensions[1],
                     inputImageDimensions[2] + voxelsPerDimToPad[0],
                     inputImageDimensions[3] + voxelsPerDimToPad[1],
                     inputImageDimensions[4] + voxelsPerDimToPad[2] ]
    
    return (paddedImage, newDimensions)


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
    partOfEarlierFmsToAddTrain = getMiddlePartOfFms(earlierLayerOutputImageTrain, earlierLayerOutputImageShapeTrain, deeperLayerOutputImageShapeTrain[2:])
    partOfEarlierFmsToAddVal = getMiddlePartOfFms(earlierLayerOutputImageVal, earlierLayerOutputImageShapeVal, deeperLayerOutputImageShapeVal[2:])
    partOfEarlierFmsToAddTest = getMiddlePartOfFms(earlierLayerOutputImageTest, earlierLayerOutputImageShapeTest, deeperLayerOutputImageShapeTest[2:])
    
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

    
##################################################
##################################################
################ THE CNN CLASS ###################
##################################################
##################################################

# MAKE A SUB-CLASS: pathway!
class Cnn3d(object):
    def __init__(self):
        
        self.cnnModelName = None
        
        self.cnnLayers = []
        self.cnnLayersSubsampled = []
        self.fcLayers = []
        self.cnnLayersZoomed1 = []
        self.CNN_PATHWAY_NORMAL = 0; self.CNN_PATHWAY_SUBSAMPLED = 1; self.CNN_PATHWAY_FC = 2; self.CNN_PATHWAY_ZOOMED1 = 3;
        self.typesOfCnnLayers = [self.cnnLayers,
                                self.cnnLayersSubsampled,
                                self.fcLayers,
                                self.cnnLayersZoomed1
                                ];
        
        self.finalLayer = ""
        
        self.numberOfOutputClasses = None
        
        #=== Compiled Functions for API ====
        self.cnnTrainModel = ""
        self.cnnValidateModel = ""
        self.cnnTestModel = ""
        self.cnnVisualiseFmFunction = ""
        
        #=====================================
        self.sharedTrainingNiiData_x = ""
        self.sharedValidationNiiData_x = ""
        self.sharedTestingNiiData_x = ""
        self.sharedTrainingNiiLabels_y = ""
        self.sharedValidationNiiLabels_y = ""
        self.sharedTrainingCoordinates_of_patches_for_epoch = ""
        self.sharedValidationCoordinates_of_patches_for_epoch = ""
        self.sharedTestingCoordinates_of_patches_for_epoch = ""
        
        self.borrowFlag = ""
        self.imagePartDimensionsTraining = ""
        self.imagePartDimensionsValidation = ""
        self.imagePartDimensionsTesting = ""
        self.subsampledImagePartDimensionsTraining = ""
        self.subsampledImagePartDimensionsValidation = ""
        self.subsampledImagePartDimensionsTesting = ""
        
        self.batchSize = ""
        self.batchSizeValidation = ""
        self.batchSizeTesting = ""
        
        self.dataTypeX = ""
        self.subsampleFactor = ""
        self.patchDimensions = ""
        #Fully Connected Layers
        self.numberOfCentralVoxelsClassifiedPerDimension = ""
        self.numberOfCentralVoxelsClassifiedPerDimensionTesting = ""   
        
        #Automatically lower CNN's learning rate by looking at validation accuracy:
        self.topMeanValidationAccuracyAchievedInEpoch = [-1,-1]
        self.lastEpochAtTheEndOfWhichLrWasLowered = 0 #refers to CnnTrained epochs, not the epochs in the do_training loop.
        
        # Residual Learning
        self.indicesOfLayersToConnectResidualsInOutput = ""
        
        # Lower rank convolutional layers
        self.indicesOfLowerRankLayersPerPathway = ""
        self.ranksOfLowerRankLayersForEachPathway = ""
        
        
        #============= ATTRIBUTES SPECIFIC TO THE TRAINING STATE ============
        self.numberOfEpochsTrained = 0
        
        self._trainingStateAttributesInitialized = False
        
        self.layersOfLayerTypesToTrain = None
        self.costFunctionLetter = "" # "L", "D" or "J"
        #====== Learning rate and momentum ==========
        self.initialLearningRate = "" #used by exponential schedule
        self.learning_rate = theano.shared(np.cast["float32"](0.01)) #initial value, changed in make_cnn_model().compileTrainingFunction()
        self.classicMomentum0OrNesterov1 = None
        #SGD + Classic momentum: (to save the momentum)
        self.initialMomentum = "" #used by exponential schedule
        self.momentum = theano.shared(np.cast["float32"](0.))
        self.momentumTypeNONNormalized0orNormalized1 = None
        self.velocities_forMom = [] #list of shared_variables. Each of the individual Dws is a sharedVar. This whole thing isnt.
        #=== Optimizer specific =====
        self.sgd0orAdam1orRmsProp2 = None
        #ADAM:
        self.b1_adam = None
        self.b2_adam = None
        self.epsilonForAdam = None
        self.i_adam = theano.shared(np.cast["float32"](0.)) #Current iteration of adam
        self.m_listForAllParamsAdam = [] #list of mean of grads for all parameters, for ADAM optimizer.
        self.v_listForAllParamsAdam = [] #list of variances of grads for all parameters, for ADAM optimizer.
        #RMSProp
        self.rho_rmsProp = None
        self.epsilonForRmsProp = None
        self.accuGradSquare_listForAllParamsRmsProp = [] #the rolling average accumulator of the variance of the grad (grad^2)
        #Regularisation
        self.L1_reg_constant = None
        self.L2_reg_constant = None
        
        
        #======= tensors, input to the CNN. Needed to be saved for later compilation after loading =======
        # Symbolic variables, which stand for the input. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
        self.inputTensorsXToCnnInitialized = False
        self.layer0_input = None; self.layer0_inputValidation = None; self.layer0_inputTesting = None;
        self.layer0_inputSubsampled = None; self.layer0_inputSubsampledValidation = None; self.layer0_inputSubsampledTesting = None;
        
    def increaseNumberOfEpochsTrained(self):
        self.numberOfEpochsTrained += 1
        
    def change_learning_rate_of_a_cnn(self, newValueForLearningRate, myLogger=None) :
        stringToPrint = "UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) +") Changing the Cnn's Learning Rate to: "+str(newValueForLearningRate)
        if myLogger<>None :
            myLogger.print3( stringToPrint )
        else :
            print stringToPrint
        self.learning_rate.set_value(newValueForLearningRate)
        self.lastEpochAtTheEndOfWhichLrWasLowered = self.numberOfEpochsTrained
        
    def divide_learning_rate_of_a_cnn_by(self, divideLrBy, myLogger=None) :
        oldLR = self.learning_rate.get_value()
        newValueForLearningRate = oldLR*1.0/divideLrBy
        self.change_learning_rate_of_a_cnn(newValueForLearningRate, myLogger)
        
        
    def change_momentum_of_a_cnn(self, newValueForMomentum, myLogger=None):
        stringToPrint = "UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) +") Changing the Cnn's Momentum to: "+str(newValueForMomentum)
        if myLogger<>None :
            myLogger.print3( stringToPrint )
        else :
            print stringToPrint
        self.momentum.set_value(newValueForMomentum)
        
    def multiply_momentum_of_a_cnn_by(self, multiplyMomentumBy, myLogger=None) :
        oldMom = self.momentum.get_value()
        newValueForMomentum = oldMom*multiplyMomentumBy
        self.change_momentum_of_a_cnn(newValueForMomentum, myLogger)
        
    def changeB1AndB2ParametersOfAdam(self, b1ParamForAdam, b2ParamForAdam, myLogger) :
        myLogger.print3("UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) +") Changing the Cnn's B1 and B2 parameters for ADAM optimization to: B1 = "+str(b1ParamForAdam) + " || B2 = " + str(b2ParamForAdam))
        self.b1_adam = b1ParamForAdam
        self.b2_adam = b2ParamForAdam
        
    def changeRhoParameterOfRmsProp(self, rhoParamForRmsProp, myLogger) :
        myLogger.print3("UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) +") Changing the Cnn's Rho parameter for RMSProp optimization to: Rho = "+str(rhoParamForRmsProp))
        self.rho_rmsProp = rhoParamForRmsProp
        
    def checkMeanValidationAccOfLastEpochAndUpdateCnnsTopAccAchievedIfNeeded(self,
                                                                            myLogger,
                                                                            meanValidationAccuracyOfLastEpoch,
                                                                            minIncreaseInValidationAccuracyConsideredForLrSchedule) :
        #Called at the end of an epoch, right before increasing self.numberOfEpochsTrained
        highestAchievedValidationAccuracyOfCnn = self.topMeanValidationAccuracyAchievedInEpoch[0]
        if meanValidationAccuracyOfLastEpoch > highestAchievedValidationAccuracyOfCnn + minIncreaseInValidationAccuracyConsideredForLrSchedule :
            self.topMeanValidationAccuracyAchievedInEpoch[0] = meanValidationAccuracyOfLastEpoch
            self.topMeanValidationAccuracyAchievedInEpoch[1] = self.numberOfEpochsTrained
            myLogger.print3("UPDATE: In this last epoch (cnnTrained) #" + str(self.topMeanValidationAccuracyAchievedInEpoch[1]) + " the CNN achieved a new highest mean validation accuracy of: " + str(self.topMeanValidationAccuracyAchievedInEpoch[0]) )
            
            
    def freeGpuTrainingData(self) :
        self.sharedTrainingNiiData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))# = []
        self.sharedTrainingSubsampledData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))
        self.sharedTrainingNiiLabels_y.set_value(np.zeros([1,1,1,1], dtype="float32"))# = []
        
    def freeGpuValidationData(self) :
        self.sharedValidationNiiData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))# = []
        self.sharedValidationSubsampledData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))
        self.sharedValidationNiiLabels_y.set_value(np.zeros([1,1,1,1], dtype="float32"))# = []
        
    def freeGpuTestingData(self) :
        self.sharedTestingNiiData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))# = []
        self.sharedTestingSubsampledData_x.set_value(np.zeros([1,1,1,1,1], dtype="float32"))
        
    def _checkTrainingStateAttributesInitialized(self):
        return self._trainingStateAttributesInitialized
    
    #for inference with batch-normalization. Every training batch, this is called to update an internal matrix of each layer, with the last mus and vars, so that I can compute the rolling average for inference.
    def updateTheMatricesOfTheLayersWithTheLastMusAndVarsForTheMovingAverageOfTheBatchNormInference(self) :
        self._updateMatricesOfBnRollingAverageForInference()
        
    def _updateMatricesOfBnRollingAverageForInference(self):
        for layer_type_i in xrange(0, len(self.typesOfCnnLayers)) :
            for layer_i in xrange(0, len(self.typesOfCnnLayers[layer_type_i])) :
                self.typesOfCnnLayers[layer_type_i][layer_i].updateTheMatricesWithTheLastMusAndVarsForTheRollingAverageOfBNInference() # Will do nothing if no BN.
                
    def makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(self,
                                                        myLogger,
                                                        thisPathwayType,
                                                        thisPathWayNKerns,
                                                        thisPathWayKernelDimensions,
                                                        inputImageToPathway,
                                                        inputImageToPathwayInference,
                                                        inputImageToPathwayTesting,
                                                        numberOfImageChannelsToPathway,
                                                        imagePartDimensionsTraining,
                                                        imagePartDimensionsValidation,
                                                        imagePartDimensionsTesting,
                                                        applyBnToInputOfPathway, # As a flag for case that I want to apply BN on input image. I want to apply to input of FC.
                                                        rollingAverageForBatchNormalizationOverThatManyBatches,
                                                        maxPoolingParamsStructureForThisPathwayType,
                                                        initializationTechniqueClassic0orDelvingInto1,
                                                        activationFunctionToUseRelu0orPrelu1,
                                                        thisPathwayDropoutRates=[],
                                                        indicesOfLayersToConnectResidualsInOutputForPathway=[],
                                                        indicesOfLowerRankLayersForPathway=[],
                                                        ranksOfLowerRankLayersForPathway = []
                                                        ) :
        
        rng = numpy.random.RandomState(55789)
        
        shapeOfInputImageToPathway = [self.batchSize, numberOfImageChannelsToPathway] + imagePartDimensionsTraining
        shapeOfInputImageToPathwayValidation = [self.batchSizeValidation, numberOfImageChannelsToPathway] + imagePartDimensionsValidation
        shapeOfInputImageToPathwayTesting = [self.batchSizeTesting, numberOfImageChannelsToPathway] + imagePartDimensionsTesting
        
        inputImageToNextLayer = inputImageToPathway
        inputImageToNextLayerInference = inputImageToPathwayInference
        inputImageToNextLayerTesting = inputImageToPathwayTesting
        inputImageToNextLayerShape = shapeOfInputImageToPathway
        inputImageToNextLayerShapeValidation = shapeOfInputImageToPathwayValidation
        inputImageToNextLayerShapeTesting = shapeOfInputImageToPathwayTesting
        
        for layer_i in xrange(0, len(thisPathWayNKerns)) :
            thisLayerFilterShape = [thisPathWayNKerns[layer_i],
                                    inputImageToNextLayerShape[1], #number of feature maps of last layer.
                                    thisPathWayKernelDimensions[layer_i][0],
                                    thisPathWayKernelDimensions[layer_i][1],
                                    thisPathWayKernelDimensions[layer_i][2]]
            
            thisLayerDropoutRate = thisPathwayDropoutRates[layer_i] if thisPathwayDropoutRates else 0
            
            thisLayerMaxPoolingParameters = maxPoolingParamsStructureForThisPathwayType[layer_i]
            
            useBnInThisLayer = applyBnToInputOfPathway if layer_i == 0 and rollingAverageForBatchNormalizationOverThatManyBatches > 0 else rollingAverageForBatchNormalizationOverThatManyBatches > 0
            activationFunctionToUseRelu0orPrelu1orMinus1ForLinear = -1 if layer_i == 0 and thisPathwayType <> self.CNN_PATHWAY_FC else activationFunctionToUseRelu0orPrelu1
            if layer_i in indicesOfLowerRankLayersForPathway :
                layer = cnnLayerTypes.LowRankConvLayer(ranksOfLowerRankLayersForPathway[ indicesOfLowerRankLayersForPathway.index(layer_i) ])
            else : # normal conv layer
                layer = cnnLayerTypes.ConvLayer()
            layer.makeLayer(rng,
                    inputToLayerTrain=inputImageToNextLayer,
                    inputToLayerVal=inputImageToNextLayerInference,
                    inputToLayerTest=inputImageToNextLayerTesting,
                    inputToLayerShapeTrain=inputImageToNextLayerShape,
                    inputToLayerShapeVal=inputImageToNextLayerShapeValidation,
                    inputToLayerShapeTest=inputImageToNextLayerShapeTesting,
                    filterShape=thisLayerFilterShape,
                    #for batchNormalization
                    useBnFlag = useBnInThisLayer,
                    rollingAverageForBatchNormalizationOverThatManyBatches=rollingAverageForBatchNormalizationOverThatManyBatches,
                    maxPoolingParameters=thisLayerMaxPoolingParameters,
                    initializationTechniqueClassic0orDelvingInto1=initializationTechniqueClassic0orDelvingInto1,
                    activationFunctionToUseRelu0orPrelu1orMinus1ForLinear=activationFunctionToUseRelu0orPrelu1orMinus1ForLinear,
                    dropoutRate=thisLayerDropoutRate
                    ) 
            self.typesOfCnnLayers[thisPathwayType].append(layer)
            
            if layer_i not in indicesOfLayersToConnectResidualsInOutputForPathway : #not a residual connecting here
                inputImageToNextLayer = layer.outputTrain
                inputImageToNextLayerInference = layer.outputVal
                inputImageToNextLayerTesting = layer.outputTest
            else : #make residual connection
                myLogger.print3("DEBUG: [Pathway-"+str(thisPathwayType)+"], making Residual Connection between output of layer ["+str(layer_i)+"] (indexing from 0) to input of previous layer.")
                deeperLayerOutputImagesTrValTest = (layer.outputTrain, layer.outputVal, layer.outputTest)
                deeperLayerOutputImageShapesTrValTest = (layer.outputShapeTrain, layer.outputShapeVal, layer.outputShapeTest)
                assert layer_i > 0 # The very first layer (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                earlierLayer = self.typesOfCnnLayers[thisPathwayType][layer_i-1]
                earlierLayerOutputImagesTrValTest = (earlierLayer.inputTrain, earlierLayer.inputVal, earlierLayer.inputTest)
                earlierLayerOutputImageShapesTrValTest = (earlierLayer.inputShapeTrain, earlierLayer.inputShapeVal, earlierLayer.inputShapeTest)
                
                (inputImageToNextLayer,
                inputImageToNextLayerInference,
                inputImageToNextLayerTesting) = makeResidualConnectionBetweenLayersAndReturnOutput( myLogger,
                                                                                                    deeperLayerOutputImagesTrValTest,
                                                                                                    deeperLayerOutputImageShapesTrValTest,
                                                                                                    earlierLayerOutputImagesTrValTest,
                                                                                                    earlierLayerOutputImageShapesTrValTest)
                
            # Residual connections preserve the both the number of FMs and the dimensions of the FMs, the same as in the later, deeper layer.
            inputImageToNextLayerShape = layer.outputShapeTrain
            inputImageToNextLayerShapeValidation = layer.outputShapeVal
            inputImageToNextLayerShapeTesting = layer.outputShapeTest
            
        return [inputImageToNextLayer, inputImageToNextLayerInference, inputImageToNextLayerTesting, inputImageToNextLayerShape, inputImageToNextLayerShapeValidation, inputImageToNextLayerShapeTesting]
    
    
    def repeatRczTheOutputOfLayerBySubsampleFactor(self, layerOutputToRepeat):
        # Repeat FM in the three dimensions, to upsample back to the normal resolution space.
        expandedOutputR = layerOutputToRepeat.repeat(self.subsampleFactor[0], axis = 2)
        expandedOutputRC = expandedOutputR.repeat(self.subsampleFactor[1], axis = 3)
        expandedOutputRCZ = expandedOutputRC.repeat(self.subsampleFactor[2], axis = 4)
        return expandedOutputRCZ
    
    def repeatRczTheOutputOfLayerBySubsampleFactorToMatchDimensionsOfOtherFm(self,
                                                                           layerOutputToRepeat,
                                                                           dimensionsOfFmToMatch ) :
        # dimensionsOfFmToMatch should be [batch_size, numberOfFms, r, c , z]. I care for RCZ.
        expandedOutput = self.repeatRczTheOutputOfLayerBySubsampleFactor(layerOutputToRepeat)
        # If the central-voxels are eg 10, the susampled-part will have 4 central voxels. Which above will be repeated to 3*4 = 12.
        # I need to clip the last ones, to have the same dimension as the input from 1st pathway, which will have dimensions equal to the centrally predicted voxels (10)
        expandedOutput = expandedOutput[:,
                                        :,
                                        :dimensionsOfFmToMatch[2],
                                        :dimensionsOfFmToMatch[3],
                                        :dimensionsOfFmToMatch[4]]
        return expandedOutput
    
    
    def makeMultiscaleConnectionsForLayerType(self,
                                        typeOfLayers_index,
                                        convLayersToConnectToFirstFcForMultiscaleFromThisLayerType,
                                        numberOfFmsOfInputToFirstFcLayer,
                                        inputToFirstFcLayer,
                                        inputToFirstFcLayerInference,
                                        inputToFirstFcLayerTesting) :
        
        layersInThisPathway = self.typesOfCnnLayers[typeOfLayers_index]
        
        if typeOfLayers_index <> self.CNN_PATHWAY_SUBSAMPLED :
            numberOfCentralVoxelsToGet = self.numberOfCentralVoxelsClassifiedPerDimension
            numberOfCentralVoxelsToGetValidation = self.numberOfCentralVoxelsClassifiedPerDimensionValidation
            numberOfCentralVoxelsToGetTesting = self.numberOfCentralVoxelsClassifiedPerDimensionTesting
            
        else : #subsampled pathway... You get one more voxel if they do not get divited by subsampleFactor exactly.
            numberOfCentralVoxelsToGet = [  int(ceil(self.numberOfCentralVoxelsClassifiedPerDimension[0]*1.0/self.subsampleFactor[0])),
                                            int(ceil(self.numberOfCentralVoxelsClassifiedPerDimension[1]*1.0/self.subsampleFactor[1])),
                                            int(ceil(self.numberOfCentralVoxelsClassifiedPerDimension[2]*1.0/self.subsampleFactor[2]))]
            numberOfCentralVoxelsToGetValidation = [ int(ceil(self.numberOfCentralVoxelsClassifiedPerDimensionValidation[0]*1.0/self.subsampleFactor[0])),
                                            int(ceil(self.numberOfCentralVoxelsClassifiedPerDimensionValidation[1]*1.0/self.subsampleFactor[1])),
                                            int(ceil(self.numberOfCentralVoxelsClassifiedPerDimensionValidation[2]*1.0/self.subsampleFactor[2]))]
            numberOfCentralVoxelsToGetTesting = [ int(ceil(self.numberOfCentralVoxelsClassifiedPerDimensionTesting[0]*1.0/self.subsampleFactor[0])),
                                            int(ceil(self.numberOfCentralVoxelsClassifiedPerDimensionTesting[1]*1.0/self.subsampleFactor[1])),
                                            int(ceil(self.numberOfCentralVoxelsClassifiedPerDimensionTesting[2]*1.0/self.subsampleFactor[2]))
                                            ]
            
        for convLayer_i in convLayersToConnectToFirstFcForMultiscaleFromThisLayerType :
            thisLayer = layersInThisPathway[convLayer_i]
            outputOfLayer = thisLayer.outputTrain
            outputOfLayerInference = thisLayer.outputVal
            outputOfLayerTesting = thisLayer.outputTest
            
            middlePartOfFms = getMiddlePartOfFms(outputOfLayer, thisLayer.outputShapeTrain, numberOfCentralVoxelsToGet)
            middlePartOfFmsInference = getMiddlePartOfFms(outputOfLayerInference, thisLayer.outputShapeVal, numberOfCentralVoxelsToGetValidation)
            middlePartOfFmsTesting = getMiddlePartOfFms(outputOfLayerTesting, thisLayer.outputShapeTest, numberOfCentralVoxelsToGetTesting)
            
            
            if typeOfLayers_index == self.CNN_PATHWAY_SUBSAMPLED :
                shapeOfOutputOfLastLayerOf1stPathway = self.typesOfCnnLayers[self.CNN_PATHWAY_NORMAL][-1].outputShapeTrain
                shapeOfOutputOfLastLayerOf1stPathwayValidation = self.typesOfCnnLayers[self.CNN_PATHWAY_NORMAL][-1].outputShapeVal
                shapeOfOutputOfLastLayerOf1stPathwayTesting = self.typesOfCnnLayers[self.CNN_PATHWAY_NORMAL][-1].outputShapeTest
                middlePartOfFms = self.repeatRczTheOutputOfLayerBySubsampleFactorToMatchDimensionsOfOtherFm(
                                                                                                middlePartOfFms,
                                                                                                shapeOfOutputOfLastLayerOf1stPathway )
                middlePartOfFmsInference = self.repeatRczTheOutputOfLayerBySubsampleFactorToMatchDimensionsOfOtherFm(
                                                                                                middlePartOfFmsInference,
                                                                                                shapeOfOutputOfLastLayerOf1stPathwayValidation )
                middlePartOfFmsTesting = self.repeatRczTheOutputOfLayerBySubsampleFactorToMatchDimensionsOfOtherFm(
                                                                                                middlePartOfFmsTesting,
                                                                                                shapeOfOutputOfLastLayerOf1stPathwayTesting )
                
            numberOfFmsOfInputToFirstFcLayer = numberOfFmsOfInputToFirstFcLayer + thisLayer.getNumberOfFeatureMaps()
            inputToFirstFcLayer = T.concatenate([inputToFirstFcLayer, middlePartOfFms], axis=1)
            inputToFirstFcLayerInference = T.concatenate([inputToFirstFcLayerInference, middlePartOfFmsInference], axis=1)
            inputToFirstFcLayerTesting = T.concatenate([inputToFirstFcLayerTesting, middlePartOfFmsTesting], axis=1)
            
        return [numberOfFmsOfInputToFirstFcLayer, inputToFirstFcLayer, inputToFirstFcLayerInference, inputToFirstFcLayerTesting]
    
    
    #========================================OPTIMIZERS========================================
    """
    From https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617 :
    ClassicMomentum:
    (1) v_t = mu * v_t-1 - lr * gradient_f(params_t)
    (2) params_t = params_t-1 + v_t
    (3) params_t = params_t-1 + mu * v_t-1 - lr * gradient_f(params_t-1)
    
    Nesterov momentum:
    (4) v_t = mu * v_t-1 - lr * gradient_f(params_t-1 + mu * v_t-1)
    (5) params_t = params_t-1 + v_t
    
    alternate formulation for Nesterov momentum:
    (6) v_t = mu * v_t-1 - lr * gradient_f(params_t-1)
    (7) params_t = params_t-1 + mu * v_t - lr * gradient_f(params_t-1)
    (8) params_t = params_t-1 + mu**2 * v_t-1 - (1+mu) * lr * gradient_f(params_t-1)
    
    Can also find help for optimizers in Lasagne: https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
    """
    
    def getUpdatesAccordingToSgd(self, cost, paramsToOptDuringTraining) :
        # create a list of gradients for all model parameters
        grads = T.grad(cost, paramsToOptDuringTraining)
        
        #========================= Momentum ===========================
        self.velocities_forMom = []
        updates = []
        
        #The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self.momentum*self.momentumTypeNONNormalized0orNormalized1
        
        for param, grad  in zip(paramsToOptDuringTraining, grads) :
            v = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            self.velocities_forMom.append(v)
            
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum*self.learning_rate*grad
            newVelocity = self.momentum*v - stepToGradientDirection
            
            if self.classicMomentum0OrNesterov1 == 0 :
                updateToParam = newVelocity
            else : #Nesterov
                updateToParam = self.momentum*newVelocity - stepToGradientDirection
                
            updates.append((v, newVelocity)) #I can do (1-mom)*learnRate*grad.
            updates.append((param, param + updateToParam))
            
        return updates
    
    def getUpdatesAccordingToRmsProp(self, cost, params) :
        #epsilon=1e-4 in paper. I got NaN in cost function when I ran it with this value. Worked ok with epsilon=1e-6.

        #Code taken and updated (it was V2 of paper, updated to V8) from https://gist.github.com/Newmu/acb738767acb4788bac3
        grads = T.grad(cost, params)
        updates = []
        self.accuGradSquare_listForAllParamsRmsProp = []
        self.velocities_forMom = []
        
        #The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self.momentum*self.momentumTypeNONNormalized0orNormalized1
        
        for param, grad in zip(params, grads):
            accu = theano.shared(param.get_value()*0., broadcastable=param.broadcastable) #accumulates the mean of the grad's square.
            self.accuGradSquare_listForAllParamsRmsProp.append(accu)
            
            v = theano.shared(param.get_value()*0., broadcastable=param.broadcastable) #velocity
            self.velocities_forMom.append(v)
            
            accu_new = self.rho_rmsProp * accu + (1 - self.rho_rmsProp) * T.sqr(grad)
            
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum*(self.learning_rate * grad /T.sqrt(accu_new + self.epsilonForRmsProp))
            
            newVelocity = self.momentum*v - stepToGradientDirection
            
            if self.classicMomentum0OrNesterov1 == 0 :
                updateToParam = newVelocity
            else : #Nesterov
                updateToParam = self.momentum*newVelocity - stepToGradientDirection
                
            updates.append((accu, accu_new))
            updates.append((v, newVelocity)) #I can do (1-mom)*learnRate*grad.
            updates.append((param, param + updateToParam))
            
        return updates
    
    
    def getUpdatesAccordingToAdam(self, cost, params) :
        # Epsilon on paper was 10**(-8).
        # Code is on par with version V8 of Kingma's paper.
        grads = T.grad(cost, params)
        
        updates = []
        
        self.i_adam = theano.shared(np.cast["float32"](0.)) #Current iteration
        self.m_listForAllParamsAdam = [] #list of mean of grads for all parameters, for ADAM optimizer.
        self.v_listForAllParamsAdam = [] #list of variances of grads for all parameters, for ADAM optimizer.
        
        i = self.i_adam
        i_t = i + 1.
        fix1 = 1. - (self.b1_adam)**i_t
        fix2 = 1. - (self.b2_adam)**i_t
        lr_t = self.learning_rate * (T.sqrt(fix2) / fix1)
        for param, grad in zip(params, grads):
            m = theano.shared(param.get_value() * 0.)
            self.m_listForAllParamsAdam.append(m)
            v = theano.shared(param.get_value() * 0.)
            self.v_listForAllParamsAdam.append(v)
            m_t = (self.b1_adam * m) + ((1. - self.b1_adam) * grad)
            v_t = (self.b2_adam * v) + ((1. - self.b2_adam) * T.sqr(grad)) #Double check this with the paper.
            grad_t = m_t / (T.sqrt(v_t) + self.epsilonForAdam)
            param_t = param - (lr_t * grad_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, param_t))
        updates.append((i, i_t))
        return updates
    
    def _getUpdatesOfTrainableParameters(self, myLogger, cost, paramsToOptDuringTraining) :
        if self.sgd0orAdam1orRmsProp2 == 0 :
            myLogger.print3("Optimizer used: [SGD]. Momentum used: Classic0 or Nesterov1 : " + str(self.classicMomentum0OrNesterov1))
            updates = self.getUpdatesAccordingToSgd(cost, paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 1 :
            myLogger.print3("Optimizer used: [ADAM]. No momentum implemented for Adam.")
            updates = self.getUpdatesAccordingToAdam(cost, paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 2 :
            myLogger.print3("Optimizer used: [RMSProp]. Momentum used: Classic0 or Nesterov1 : " + str(self.classicMomentum0OrNesterov1))
            updates = self.getUpdatesAccordingToRmsProp(cost, paramsToOptDuringTraining)
        return updates
    
    def _initializeSharedVarsForXInputs(self):
        # ======= Initialize sharedVariables ==========
        #Create the needed shared variables. Number of dimensions should be correct (5 for x, 4 for y). But size is placeholder. Changes when shared.set_value during training.
        self.sharedTrainingNiiData_x = theano.shared(np.zeros([1,1,1,1,1], dtype="float32"), borrow = self.borrowFlag)
        self.sharedTrainingSubsampledData_x = theano.shared(np.zeros([1,1,1,1,1], dtype="float32"), borrow = self.borrowFlag)
        self.sharedValidationNiiData_x = theano.shared(np.zeros([1,1,1,1,1], dtype="float32"), borrow = self.borrowFlag)
        self.sharedValidationSubsampledData_x = theano.shared(np.zeros([1,1,1,1,1], dtype="float32"), borrow = self.borrowFlag)
        self.sharedTestingNiiData_x = theano.shared(np.zeros([1,1,1,1,1], dtype="float32"), borrow = self.borrowFlag)
        self.sharedTestingSubsampledData_x = theano.shared(np.zeros([1,1,1,1,1], dtype="float32"), borrow = self.borrowFlag)
        
    def _initializeSharedVarsForYInputs(self):
        # When storing data on the GPU it has to be stored as floats (floatX). Later this variable is cast as "int", to be used correctly in computations.
        self.sharedTrainingNiiLabels_y = theano.shared(np.zeros([1,1,1,1], dtype="float32") , borrow = self.borrowFlag)
        self.sharedValidationNiiLabels_y = theano.shared(np.zeros([1,1,1,1], dtype="float32") , borrow = self.borrowFlag)
        
    def _getTrainableParameters(self):
        paramsToOptDuringTraining = [] #Ws and Bs
        for type_of_layer_i in xrange(0, len(self.typesOfCnnLayers) ) :
            for layer_i in xrange(0, len(self.typesOfCnnLayers[type_of_layer_i]) ) :
                if self.layersOfLayerTypesToTrain == "all" or (layer_i in self.layersOfLayerTypesToTrain[type_of_layer_i]) :
                    paramsToOptDuringTraining = paramsToOptDuringTraining + self.typesOfCnnLayers[type_of_layer_i][layer_i].params
        return paramsToOptDuringTraining
    
    def _getL1RegCost(self) :
        L1 = 0
        for type_of_layer_i in xrange(0, len(self.typesOfCnnLayers) ) :
            for layer_i in xrange(0, len(self.typesOfCnnLayers[type_of_layer_i]) ) :    
                L1 += self.typesOfCnnLayers[type_of_layer_i][layer_i].getL1RegCost()
        return L1
    
    def _getL2RegCost(self) :
        L2_sqr = 0
        for type_of_layer_i in xrange(0, len(self.typesOfCnnLayers) ) :
            for layer_i in xrange(0, len(self.typesOfCnnLayers[type_of_layer_i]) ) :    
                L2_sqr += self.typesOfCnnLayers[type_of_layer_i][layer_i].getL2RegCost()
        return L2_sqr
    
    # This function should be called at least once prior to compiling train function for the first time. 
    # If I need to "resume" training, this should not be called.
    # However, if I need to use a pretrained model, and train it in a second stage, I should recall this, with the new stage's parameters, and then recompile trainFunction.
    def initializeTrainingState(self,
                                myLogger,
                                layersOfLayerTypesToTrain,
                                costFunctionLetter,
                                learning_rate,
                                sgd0orAdam1orRmsProp2,
                                classicMomentum0OrNesterov1,
                                momentum,
                                momentumTypeNONNormalized0orNormalized1,
                                b1ParamForAdam,
                                b2ParamForAdam,
                                epsilonForAdam,
                                rhoParamForRmsProp,
                                epsilonForRmsProp,
                                L1_reg_constant,
                                L2_reg_constant
                                ) :
        myLogger.print3("Setting the training-related attributes of the CNN.")
        self.numberOfEpochsTrained = 0
        
        # Layers to train (rest are left untouched, eg for pretrained models.
        self.layersOfLayerTypesToTrain = layersOfLayerTypesToTrain
        
        # Cost function
        if costFunctionLetter <> "previous" :
            self.costFunctionLetter = costFunctionLetter
            
        # Regularization
        self.L1_reg_constant = L1_reg_constant
        self.L2_reg_constant = L2_reg_constant
        
        # Learning rate and momentum
        self.initialLearningRate = learning_rate # This is important for the learning rate schedule to work.
        self.change_learning_rate_of_a_cnn(learning_rate, myLogger)
        self.classicMomentum0OrNesterov1 = classicMomentum0OrNesterov1
        self.initialMomentum = momentum
        self.momentumTypeNONNormalized0orNormalized1 = momentumTypeNONNormalized0orNormalized1
        self.change_momentum_of_a_cnn(momentum, myLogger)
        
        # Optimizer
        self.sgd0orAdam1orRmsProp2 = sgd0orAdam1orRmsProp2
        if sgd0orAdam1orRmsProp2 == 1 :
            self.changeB1AndB2ParametersOfAdam(b1ParamForAdam, b2ParamForAdam, myLogger)
            self.epsilonForAdam = epsilonForAdam
        elif sgd0orAdam1orRmsProp2 == 2 :
            self.changeRhoParameterOfRmsProp(rhoParamForRmsProp, myLogger)
            self.epsilonForRmsProp = epsilonForRmsProp
            
        self._trainingStateAttributesInitialized = True
        
    def _getUpdatesForBnRollingAverage(self) :
        #These are not the variables of the normalization of the FMs' distributions that are optimized during training. These are only the Mu and Stds that are used during inference,
        #... and here we update the sharedVariable which is used "from the outside during do_training()" to update the rolling-average-matrix for inference. Do for all layers.
        updatesForBnRollingAverage = []
        for layer_type_i in xrange( 0, len(self.typesOfCnnLayers) ) :
            for layer_i in xrange( 0, len(self.typesOfCnnLayers[layer_type_i]) ) :
                theCertainLayer = self.typesOfCnnLayers[layer_type_i][layer_i]
                updatesForBnRollingAverage.extend( theCertainLayer.getUpdatesForBnRollingAverage() ) #CAREFUL: WARN, PROBLEM, THEANO BUG! If a layer has only 1FM, the .newMu_B ends up being of type (true,) instead of vector!!! Error!!!
        return updatesForBnRollingAverage
    
    #NOTE: compileTrainFunction() changes the self.initialLearningRate. Which is used for the exponential schedule!
    def compileTrainFunction(self, myLogger) :
        myLogger.print3("...Building the training function...")
        
        if not self._checkTrainingStateAttributesInitialized() :
            myLogger.print3("ERROR: Prior to compiling the training function, training state attributes need to be initialized via a call of [Cnn3d.setTrainingStateAttributes(...)]. Exiting!"); exit(1)
            
        self._initializeSharedVarsForXInputs()
        self._initializeSharedVarsForYInputs()
        
        #symbolic variables needed:
        index = T.lscalar()
        x = self.layer0_input
        xSubsampled = self.layer0_inputSubsampled
        y = T.itensor4('y') # Input of the theano-compiled-function. Dimensions of y labels: [batchSize, r, c, z]
        # When storing data on the GPU it has to be stored as floats (floatX). Thus the sharedVariable is FloatX/32. Here this variable is cast as "int", to be used correctly in computations.
        intCastSharedTrainingNiiLabels_y = T.cast( self.sharedTrainingNiiLabels_y, 'int32')
        inputVectorWeightsOfClassesInCostFunction = T.fvector() #These two were added to counter class imbalance by changing the weights in the cost function
        weightPerClass = T.fvector() # a vector with 1 element per class.
        
        # ======= Create List Of Trained Parameters to be fit by gradient descent=======
        paramsToOptDuringTraining = self._getTrainableParameters()
        
        #==========================COST FUNCTION=======================
        #The cost Function to use.
        if self.costFunctionLetter == "L" :
            costFromLastLayer = self.finalLayer.negativeLogLikelihood(y, weightPerClass)
        else :
            myLogger.print3("ERROR: Problem in make_cnn_model(). The parameter self.costFunctionLetter did not have an acceptable value( L,D,J ). Exiting."); exit(1)
            
        cost = (costFromLastLayer
                + self.L1_reg_constant * self._getL1RegCost()
                + self.L2_reg_constant * self._getL2RegCost())
        
        #============================OPTIMIZATION=============================
        updates = self._getUpdatesOfTrainableParameters(myLogger, cost, paramsToOptDuringTraining)
        
        #================BATCH NORMALIZATION ROLLING AVERAGE UPDATES======================
        updates = updates + self._getUpdatesForBnRollingAverage()
        
        #========================COMPILATION OF FUNCTIONS =================
        
        if not self.usingSubsampledPathway : # This is to avoid warning from theano for unused input (xSubsampled), in case I am not using the pathway.
            givensSet = {   x: self.sharedTrainingNiiData_x[index * self.batchSize: (index + 1) * self.batchSize],
                            y: intCastSharedTrainingNiiLabels_y[index * self.batchSize: (index + 1) * self.batchSize],
                            weightPerClass: inputVectorWeightsOfClassesInCostFunction }
        else :
            givensSet = {   x: self.sharedTrainingNiiData_x[index * self.batchSize: (index + 1) * self.batchSize],
                            xSubsampled: self.sharedTrainingSubsampledData_x[index * self.batchSize: (index + 1) * self.batchSize],
                            y: intCastSharedTrainingNiiLabels_y[index * self.batchSize: (index + 1) * self.batchSize],
                            weightPerClass: inputVectorWeightsOfClassesInCostFunction }
        myLogger.print3("...Compiling the function for training... (This may take a few minutes...)")
        self.cnnTrainModel = theano.function(
                                [index, inputVectorWeightsOfClassesInCostFunction],
                                [cost] + self.finalLayer.getRpRnTpTnForTrain0OrVal1(y,0),
                                updates=updates,
                                givens = givensSet
                                )
        myLogger.print3("The training function was compiled.")
        
    def compileValidationFunction(self, myLogger) :
        myLogger.print3("...Building the validation function...")
        
        #symbolic variables needed:
        index = T.lscalar()
        x = self.layer0_inputValidation
        xSubsampled = self.layer0_inputSubsampledValidation
        y = T.itensor4('y') # Input of the theano-compiled-function. Dimensions of y labels: [batchSize, r, c, z]
        # When storing data on the GPU it has to be stored as floats (floatX). Thus the sharedVariable is FloatX/32. Here this variable is cast as "int", to be used correctly in computations.
        intCastSharedValidationNiiLabels_y = T.cast( self.sharedValidationNiiLabels_y, 'int32')
        
        if not self.usingSubsampledPathway : # This is to avoid warning from theano for unused input (xSubsampled), in case I am not using the pathway.
            givensSet = {   x: self.sharedValidationNiiData_x[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation],
                            y: intCastSharedValidationNiiLabels_y[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation] }
        else :
            givensSet = {   x: self.sharedValidationNiiData_x[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation],
                            xSubsampled: self.sharedValidationSubsampledData_x[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation],
                            y: intCastSharedValidationNiiLabels_y[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation] }
            
        myLogger.print3("...Compiling the function for validation... (This may take a few minutes...)")
        self.cnnValidateModel = theano.function(
                                    [index],
                                    self.finalLayer.getRpRnTpTnForTrain0OrVal1(y,1),
                                    givens = givensSet
                                    )
        myLogger.print3("The validation function was compiled.")
        
        
    def compileTestAndVisualisationFunction(self, myLogger) :
        myLogger.print3("...Building the function for testing and visualisation of FMs...")
        
        #symbolic variables needed:
        index = T.lscalar()
        x = self.layer0_inputTesting
        xSubsampled = self.layer0_inputSubsampledTesting
        
        listToReturnWithAllTheFmActivationsAndPredictionsAppended = []
        for type_of_layer_i in xrange(0,len(self.typesOfCnnLayers)) : #0=simple pathway, 1 = subsampled pathway, 2 = fc layers, 3 = zoomedIn1.
            for layer_i in xrange(0, len(self.typesOfCnnLayers[type_of_layer_i])) : #each layer that this pathway/fc has.
                listToReturnWithAllTheFmActivationsAndPredictionsAppended.append(self.typesOfCnnLayers[type_of_layer_i][layer_i].fmsActivations([0,9999]))
                
        listToReturnWithAllTheFmActivationsAndPredictionsAppended.append(self.finalLayer.predictionProbabilities())
        
        if not self.usingSubsampledPathway : # This is to avoid warning from theano for unused input (xSubsampled), in case I am not using the pathway.
            givensSet = {   x: self.sharedTestingNiiData_x[index * self.batchSizeTesting: (index + 1) * self.batchSizeTesting] }
        else :
            givensSet = {   x: self.sharedTestingNiiData_x[index * self.batchSizeTesting: (index + 1) * self.batchSizeTesting],
                            xSubsampled: self.sharedTestingSubsampledData_x[index * self.batchSizeTesting: (index + 1) * self.batchSizeTesting] }
            
        myLogger.print3("...Compiling the function for testing and visualisation of FMs... (This may take a few minutes...)")
        self.cnnTestAndVisualiseAllFmsFunction = theano.function(
                                                        [index],
                                                        listToReturnWithAllTheFmActivationsAndPredictionsAppended,
                                                        givens = givensSet
                                                        )
        myLogger.print3("The function for testing and visualisation of FMs was compiled.")
        
    def _getInputTensorsXToCnn(self):
        if not self.inputTensorsXToCnnInitialized :
            # Symbolic variables, which stand for the input to the CNN. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
            tensor5 = T.TensorType(dtype='float32', broadcastable=(False, False, False, False, False))
            self.layer0_input = tensor5() # Actually, for these 3, a single tensor5() could be used, as long as I reshape it separately for each afterwards. The actual value is loaded by the compiled functions.
            self.layer0_inputValidation = tensor5() # myTensor.reshape(inputImageShapeValidation)
            self.layer0_inputTesting = tensor5()
            self.layer0_inputSubsampled = tensor5() # Actually, for these 3, a single tensor5() could be used.
            self.layer0_inputSubsampledValidation = tensor5() # myTensor.reshape(inputImageSubsampledShapeValidation)
            self.layer0_inputSubsampledTesting = tensor5()
            self.inputTensorsXToCnnInitialized = True
            
        return (self.layer0_input, self.layer0_inputValidation, self.layer0_inputTesting,
                self.layer0_inputSubsampled, self.layer0_inputSubsampledValidation, self.layer0_inputSubsampledTesting)
        
    def _getClassificationLayer(self):
        return cnnLayerTypes.ConvLayerWithSoftmax()
        
    def make_cnn_model( self,
                        myLogger,
                        cnnModelName,
                        #=== Model Parameters ===
                        numberOfOutputClasses,
                        numberOfImageChannelsPath1,
                        numberOfImageChannelsPath2,
                        
                        #=== Normal Pathway ===
                        nkerns,
                        kernelDimensions,
                        patchDimensions, # Should be automatically calculate it in here
                        #=== Subsampled Pathway ===
                        nkernsSubsampled,
                        kernelDimensionsSubsampled,
                        subsampleFactor,
                        #=== zoomed-in pathway === # Deprecated.
                        zoomedInPatchDimensions,
                        nkernsZoomedIn1,
                        kernelDimensionsZoomedIn1,
                        #=== FC Layers ===
                        fcLayersFMs,
                        kernelDimensionsFirstFcLayer,
                        softmaxTemperature,
                        
                        #=== Other Architectural params ===
                        activationFunctionToUseRelu0orPrelu1,
                        #---Residual Connections----
                        indicesOfLayersToConnectResidualsInOutput,
                        #--Lower Rank Layer Per Pathway---
                        indicesOfLowerRankLayersPerPathway,
                        ranksOfLowerRankLayersForEachPathway,
                        #---Pooling---
                        maxPoolingParamsStructure,
                        #--- Skip Connections --- #Deprecated, not used/supported
                        convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes,
                        
                        #===Size of Image Segments ===
                        imagePartDimensionsTraining ,
                        imagePartDimensionsValidation,
                        imagePartDimensionsTesting,
                        subsampledImagePartDimensionsTraining, # Should be automatically calculate it in here
                        subsampledImagePartDimensionsValidation,
                        subsampledImagePartDimensionsTesting,
                        
                        #=== Batch Sizes ===
                        batch_size,
                        batch_size_validation,
                        batch_size_testing,
                        
                        #=== Others ===
                        #Dropout
                        dropoutRatesForAllPathways, #list of sublists, one for each pathway. Each either empty or full with the dropout rates of all the layers in the path.
                        #Initialization
                        initializationTechniqueClassic0orDelvingInto1,
                        #Batch Normalization
                        applyBnToInputOfPathways, # one Boolean flag per pathway type. Placeholder for the FC pathway.
                        rollingAverageForBatchNormalizationOverThatManyBatches,
                        
                        #=== various ====
                        borrowFlag,
                        dataTypeX = 'float32',
                        ):
        """
        maxPoolingParamsStructure: The padding of the function further below adds zeros. Zeros are not good, especially if I use PreLus. So I made mine, that pads by mirroring.
        Be careful that, without this, if I have ds=2, str=1 and ignoreBorder=False, it still reduces the dimension of the image by 1. That's why I need this. To keep the dimensions stable.
        It mirrors the last elements of each dimension as many times as it is given as arg.
        """
        self.cnnModelName = cnnModelName
        
        # ============= Model Parameters Passed as arguments ================
        self.numberOfOutputClasses = numberOfOutputClasses
        self.numberOfImageChannelsPath1 = numberOfImageChannelsPath1
        self.numberOfImageChannelsPath2 = numberOfImageChannelsPath2
        # === Architecture ===
        self.usingSubsampledPathway = len(nkernsSubsampled) > 0
        self.subsampleFactor = subsampleFactor
        self.patchDimensions = patchDimensions # receptive field
        self.zoomedInPatchDimensions = zoomedInPatchDimensions
        
        #== Other Architectural Params ==
        self.indicesOfLayersToConnectResidualsInOutput = indicesOfLayersToConnectResidualsInOutput
        self.indicesOfLowerRankLayersPerPathway = indicesOfLowerRankLayersPerPathway
        #pooling?
        #== Size of Image Segments ==
        self.imagePartDimensionsTraining = imagePartDimensionsTraining
        self.imagePartDimensionsValidation = imagePartDimensionsValidation
        self.imagePartDimensionsTesting = imagePartDimensionsTesting
        self.subsampledImagePartDimensionsTraining = subsampledImagePartDimensionsTraining
        self.subsampledImagePartDimensionsValidation = subsampledImagePartDimensionsValidation
        self.subsampledImagePartDimensionsTesting = subsampledImagePartDimensionsTesting
        # == Batch Sizes ==
        self.batchSize = batch_size
        self.batchSizeValidation = batch_size_validation
        self.batchSizeTesting = batch_size_testing
        #== Others ==
        self.dropoutRatesForAllPathways = dropoutRatesForAllPathways
        self.initializationTechniqueClassic0orDelvingInto1 = initializationTechniqueClassic0orDelvingInto1
        #== various ==
        self.borrowFlag = borrowFlag
        self.dataTypeX = dataTypeX
        
        # ======== Calculated Attributes =========
        self.numberOfCentralVoxelsClassifiedPerDimension = getListOfNumberOfCentralVoxelsClassifiedPerDimension(imagePartDimensionsTraining, patchDimensions) # I should get rid of this.
        self.numberOfCentralVoxelsClassifiedPerDimensionValidation = getListOfNumberOfCentralVoxelsClassifiedPerDimension(imagePartDimensionsValidation, patchDimensions) # I should get rid of this.
        self.numberOfCentralVoxelsClassifiedPerDimensionTesting = getListOfNumberOfCentralVoxelsClassifiedPerDimension(imagePartDimensionsTesting, patchDimensions) # I should get rid of this.
        
        #==============================
        rng = numpy.random.RandomState(23455)
        
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        myLogger.print3("...Building the CNN model...")
        
        # Symbolic variables, which stand for the input. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
        (layer0_input, layer0_inputValidation, layer0_inputTesting,
        layer0_inputSubsampled, layer0_inputSubsampledValidation, layer0_inputSubsampledTesting) = self._getInputTensorsXToCnn()
        
        inputImageShape = (self.batchSize, numberOfImageChannelsPath1, imagePartDimensionsTraining[0], imagePartDimensionsTraining[1], imagePartDimensionsTraining[2])
        inputImageShapeValidation = (self.batchSizeValidation, numberOfImageChannelsPath1, imagePartDimensionsValidation[0], imagePartDimensionsValidation[1], imagePartDimensionsValidation[2])
        inputImageShapeTesting = (self.batchSizeTesting, numberOfImageChannelsPath1, imagePartDimensionsTesting[0], imagePartDimensionsTesting[1], imagePartDimensionsTesting[2])
        
        if self.usingSubsampledPathway : #Using subsampled pathway.
            inputImageSubsampledShape = (self.batchSize, numberOfImageChannelsPath2, subsampledImagePartDimensionsTraining[0], subsampledImagePartDimensionsTraining[1], subsampledImagePartDimensionsTraining[2])
            inputImageSubsampledShapeValidation = (self.batchSizeValidation, numberOfImageChannelsPath2, subsampledImagePartDimensionsValidation[0], subsampledImagePartDimensionsValidation[1], subsampledImagePartDimensionsValidation[2])
            inputImageSubsampledShapeTesting = (self.batchSizeTesting, numberOfImageChannelsPath2, subsampledImagePartDimensionsTesting[0], subsampledImagePartDimensionsTesting[1], subsampledImagePartDimensionsTesting[2])
            
        #=======================Make the FIRST (NORMAL) PATHWAY of the CNN=======================
        thisPathwayType = self.CNN_PATHWAY_NORMAL
        thisPathWayNKerns = nkerns
        thisPathWayKernelDimensions = kernelDimensions
        inputImageToPathway =  layer0_input
        inputImageToPathwayInference = layer0_inputValidation
        inputImageToPathwayTesting = layer0_inputTesting
        
        [outputNormalPathTrain,
        outputNormalPathVal,
        outputNormalPathTest,
        dimensionsOfOutputFrom1stPathway,
        dimensionsOfOutputFrom1stPathwayValidation,
        dimensionsOfOutputFrom1stPathwayTesting] = self.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                                                            thisPathwayType,
                                                                                                            thisPathWayNKerns,
                                                                                                            thisPathWayKernelDimensions,
                                                                                                            inputImageToPathway,
                                                                                                            inputImageToPathwayInference,
                                                                                                            inputImageToPathwayTesting,
                                                                                                            numberOfImageChannelsPath1,
                                                                                                            imagePartDimensionsTraining,
                                                                                                            imagePartDimensionsValidation,
                                                                                                            imagePartDimensionsTesting,
                                                                                                            applyBnToInputOfPathways[thisPathwayType],
                                                                                                            rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                                                            maxPoolingParamsStructure[thisPathwayType],
                                                                                                            initializationTechniqueClassic0orDelvingInto1,
                                                                                                            activationFunctionToUseRelu0orPrelu1,
                                                                                                            dropoutRatesForAllPathways[thisPathwayType],
                                                                                                            indicesOfLayersToConnectResidualsInOutput[thisPathwayType],
                                                                                                            indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                                                            ranksOfLowerRankLayersForEachPathway[thisPathwayType]
                                                                                                            )
        myLogger.print3("DEBUG: Shape of output of the FIRST PATH: (Train) "+str(dimensionsOfOutputFrom1stPathway)+\
                         ", (Val) "+str(dimensionsOfOutputFrom1stPathwayValidation)+", (Test) "+str(dimensionsOfOutputFrom1stPathwayTesting))
        
        inputToFirstFcLayer = outputNormalPathTrain
        inputToFirstFcLayerInference = outputNormalPathVal
        inputToFirstFcLayerTesting = outputNormalPathTest
        numberOfFmsOfInputToFirstFcLayer = dimensionsOfOutputFrom1stPathway[1]
        
        #====================== Make the Multi-scale-in-net connections for Path-1 ===========================        
        [numberOfFmsOfInputToFirstFcLayer, #updated after concatenations
        inputToFirstFcLayer,
        inputToFirstFcLayerInference,
        inputToFirstFcLayerTesting] = self.makeMultiscaleConnectionsForLayerType(self.CNN_PATHWAY_NORMAL,
                                                                                convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes[self.CNN_PATHWAY_NORMAL],
                                                                                numberOfFmsOfInputToFirstFcLayer,
                                                                                inputToFirstFcLayer,
                                                                                inputToFirstFcLayerInference,
                                                                                inputToFirstFcLayerTesting)
        
        #=======================Make the SECOND (SUBSAMPLED) PATHWAY of the CNN=============================
        
        if self.usingSubsampledPathway : #If there is actually a 2nd pathway in this model...
            thisPathwayType = self.CNN_PATHWAY_SUBSAMPLED
            thisPathWayNKerns = nkernsSubsampled
            thisPathWayKernelDimensions = kernelDimensionsSubsampled
            inputImageToPathway =  layer0_inputSubsampled
            inputImageToPathwayInference = layer0_inputSubsampledValidation
            inputImageToPathwayTesting = layer0_inputSubsampledTesting
            
            [outputSubsampledPathTrain,
            outputSubsampledPathVal,
            outputSubsampledPathTest,
            dimensionsOfOutputFrom2ndPathway,
            dimensionsOfOutputFrom2ndPathwayValidation,
            dimensionsOfOutputFrom2ndPathwayTesting] = self.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                                                                thisPathwayType,
                                                                                                                thisPathWayNKerns,
                                                                                                                thisPathWayKernelDimensions,
                                                                                                                inputImageToPathway,
                                                                                                                inputImageToPathwayInference,
                                                                                                                inputImageToPathwayTesting,
                                                                                                                numberOfImageChannelsPath2,
                                                                                                                subsampledImagePartDimensionsTraining,
                                                                                                                subsampledImagePartDimensionsValidation,
                                                                                                                subsampledImagePartDimensionsTesting,
                                                                                                                applyBnToInputOfPathways[thisPathwayType],
                                                                                                                rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                                                                maxPoolingParamsStructure[thisPathwayType],
                                                                                                                initializationTechniqueClassic0orDelvingInto1,
                                                                                                                activationFunctionToUseRelu0orPrelu1,
                                                                                                                dropoutRatesForAllPathways[thisPathwayType],
                                                                                                                indicesOfLayersToConnectResidualsInOutput[thisPathwayType],
                                                                                                                indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                                                                ranksOfLowerRankLayersForEachPathway[thisPathwayType]
                                                                                                                )
            myLogger.print3("DEBUG: Shape of output of the SECOND PATH: (Train) "+str(dimensionsOfOutputFrom2ndPathway)+\
                    ", (Val) "+str(dimensionsOfOutputFrom2ndPathwayValidation)+", (Test) "+str(dimensionsOfOutputFrom2ndPathwayTesting))
            
            expandedOutputOfLastLayerOfSecondCnnPathway = self.repeatRczTheOutputOfLayerBySubsampleFactorToMatchDimensionsOfOtherFm(
                                                                                                                            outputSubsampledPathTrain,
                                                                                                                            dimensionsOfOutputFrom1stPathway)
            #For Validation with Subsampled pathway:
            expandedOutputOfLastLayerOfSecondCnnPathwayInference = self.repeatRczTheOutputOfLayerBySubsampleFactorToMatchDimensionsOfOtherFm(
                                                                                                                            outputSubsampledPathVal,
                                                                                                                            dimensionsOfOutputFrom1stPathwayValidation )
            #For Testing with Subsampled pathway:
            expandedOutputOfLastLayerOfSecondCnnPathwayTesting = self.repeatRczTheOutputOfLayerBySubsampleFactorToMatchDimensionsOfOtherFm(
                                                                                                                            outputSubsampledPathTest,
                                                                                                                            dimensionsOfOutputFrom1stPathwayTesting )
            
            #====================================CONCATENATE the output of the 2 cnn-pathways=============================
            inputToFirstFcLayer = T.concatenate([inputToFirstFcLayer, expandedOutputOfLastLayerOfSecondCnnPathway], axis=1)
            inputToFirstFcLayerInference = T.concatenate([inputToFirstFcLayerInference, expandedOutputOfLastLayerOfSecondCnnPathwayInference], axis=1)
            inputToFirstFcLayerTesting = T.concatenate([inputToFirstFcLayerTesting, expandedOutputOfLastLayerOfSecondCnnPathwayTesting], axis=1)
            numberOfFmsOfInputToFirstFcLayer = numberOfFmsOfInputToFirstFcLayer + dimensionsOfOutputFrom2ndPathway[1]
            
            #====================== Make the Multi-scale-in-net connections for Path-2 ===========================        
            [numberOfFmsOfInputToFirstFcLayer, #updated after concatenations
            inputToFirstFcLayer,
            inputToFirstFcLayerInference,
            inputToFirstFcLayerTesting] = self.makeMultiscaleConnectionsForLayerType(self.CNN_PATHWAY_SUBSAMPLED,
                                                                                    convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes[self.CNN_PATHWAY_SUBSAMPLED],
                                                                                    numberOfFmsOfInputToFirstFcLayer,
                                                                                    inputToFirstFcLayer,
                                                                                    inputToFirstFcLayerInference,
                                                                                    inputToFirstFcLayerTesting)
            
            
        #=======================Make the THIRD (ZOOMED IN) PATHWAY of the CNN=======================
        if len(nkernsZoomedIn1)>0 :        
            thisPathwayType = self.CNN_PATHWAY_ZOOMED1
            thisPathWayNKerns = nkernsZoomedIn1
            thisPathWayKernelDimensions = kernelDimensionsZoomedIn1
            
            imagePartDimensionsForZoomedPatch1Training = []; imagePartDimensionsForZoomedPatch1Validation=[]; imagePartDimensionsForZoomedPatch1Testing=[];
            for dim_i in xrange(0, 3) :
                imagePartDimensionsForZoomedPatch1Training.append(self.zoomedInPatchDimensions[dim_i] + self.numberOfCentralVoxelsClassifiedPerDimension[dim_i] -1)
                imagePartDimensionsForZoomedPatch1Validation.append(self.zoomedInPatchDimensions[dim_i] + self.numberOfCentralVoxelsClassifiedPerDimensionValidation[dim_i] -1)
                imagePartDimensionsForZoomedPatch1Testing.append(self.zoomedInPatchDimensions[dim_i] + self.numberOfCentralVoxelsClassifiedPerDimensionTesting[dim_i] -1)
                
            inputImageToPathway =  getMiddlePartOfFms(layer0_input, inputImageShape, imagePartDimensionsForZoomedPatch1Training)
            inputImageToPathwayInference = getMiddlePartOfFms(layer0_inputValidation, inputImageShapeValidation, imagePartDimensionsForZoomedPatch1Validation)
            inputImageToPathwayTesting = getMiddlePartOfFms(layer0_inputTesting, inputImageShapeTesting, imagePartDimensionsForZoomedPatch1Testing)
            
            [outputZoomedPathTrain,
            outputZoomedPathVal,
            outputZoomedPathTest,
            dimensionsOfOutputFromZoomed1Pathway,
            dimensionsOfOutputFromZoomed1PathwayValidation,
            dimensionsOfOutputFromZoomed1PathwayTesting] = self.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                                                                    thisPathwayType,
                                                                                                                    thisPathWayNKerns,
                                                                                                                    thisPathWayKernelDimensions,
                                                                                                                    inputImageToPathway,
                                                                                                                    inputImageToPathwayInference,
                                                                                                                    inputImageToPathwayTesting,
                                                                                                                    numberOfImageChannelsPath1,
                                                                                                                    imagePartDimensionsForZoomedPatch1Training,
                                                                                                                    imagePartDimensionsForZoomedPatch1Validation,
                                                                                                                    imagePartDimensionsForZoomedPatch1Testing,
                                                                                                                    applyBnToInputOfPathways[thisPathwayType],
                                                                                                                    rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                                                                    maxPoolingParamsStructure[thisPathwayType],
                                                                                                                    initializationTechniqueClassic0orDelvingInto1,
                                                                                                                    activationFunctionToUseRelu0orPrelu1,
                                                                                                                    dropoutRatesForAllPathways[thisPathwayType],
                                                                                                                    indicesOfLayersToConnectResidualsInOutput[thisPathwayType],
                                                                                                                    indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                                                                    ranksOfLowerRankLayersForEachPathway[thisPathwayType]
                                                                                                                    )
            #====================================CONCATENATE the output of the ZoomedIn1 pathway=============================
            inputToFirstFcLayer = T.concatenate([inputToFirstFcLayer, outputZoomedPathTrain], axis=1)
            inputToFirstFcLayerInference = T.concatenate([inputToFirstFcLayerInference, outputZoomedPathVal], axis=1)
            inputToFirstFcLayerTesting = T.concatenate([inputToFirstFcLayerTesting, outputZoomedPathTest], axis=1)
            numberOfFmsOfInputToFirstFcLayer = numberOfFmsOfInputToFirstFcLayer + dimensionsOfOutputFromZoomed1Pathway[1]
            
            
        #======================= Make the Fully Connected Layers =======================
        # This is the shape of the kernel in the first FC layer.
        # NOTE: If there is no hidden FC layer, this kernel is used in the Classification layer then.
        # Originally it was 1x1x1 only. The pathways themselves where taking care of the receptive field.
        # However I can now define it larger (eg 3x3x3), in case it helps combining the multiresolution features better/smoother.
        # The convolution is seamless, ie same shape output/input, by mirror padding the input.
        firstFcLayerAfterConcatenationKernelShape = kernelDimensionsFirstFcLayer
        myLogger.print3("DEBUG: Shape of the kernel of the first FC layer is : " + str(firstFcLayerAfterConcatenationKernelShape) )
        inputOfFcPathwayImagePartDimsTrain = [self.batchSize, numberOfFmsOfInputToFirstFcLayer] + dimensionsOfOutputFrom1stPathway[2:5]
        inputOfFcPathwayImagePartDimsVal = [self.batchSizeValidation, numberOfFmsOfInputToFirstFcLayer] + dimensionsOfOutputFrom1stPathwayValidation[2:5]
        inputOfFcPathwayImagePartDimsTest = [self.batchSizeTesting, numberOfFmsOfInputToFirstFcLayer] + dimensionsOfOutputFrom1stPathwayTesting[2:5]
        voxelsToPadPerDim = [ kernelDim -1 for kernelDim in firstFcLayerAfterConcatenationKernelShape ]
        (inputImageToPathway,
        inputOfFcPathwayImagePartDimsTrain) = padImageWithMirroring( inputToFirstFcLayer, inputOfFcPathwayImagePartDimsTrain, voxelsToPadPerDim )
        (inputImageToPathwayInference,
        inputOfFcPathwayImagePartDimsVal) = padImageWithMirroring( inputToFirstFcLayerInference, inputOfFcPathwayImagePartDimsVal, voxelsToPadPerDim )
        (inputImageToPathwayTesting,
        inputOfFcPathwayImagePartDimsTest) = padImageWithMirroring( inputToFirstFcLayerTesting, inputOfFcPathwayImagePartDimsTest, voxelsToPadPerDim )
        myLogger.print3("DEBUG: Shape of input to the FC PATH: (Train) "+str(inputOfFcPathwayImagePartDimsTrain)+\
                         ", (Val) "+str(inputOfFcPathwayImagePartDimsVal)+", (Test) "+str(inputOfFcPathwayImagePartDimsTest))
        
        if len(fcLayersFMs)>0 : 
            thisPathwayType = self.CNN_PATHWAY_FC
            thisPathWayNKerns = fcLayersFMs
            thisPathWayKernelDimensions = [firstFcLayerAfterConcatenationKernelShape]
            for fcLayer_i in xrange(1,len(fcLayersFMs)) :
                thisPathWayKernelDimensions.append([1,1,1])
            
            [inputImageForFinalClassificationLayer,
            inputImageForFinalClassificationLayerInference,
            inputImageForFinalClassificationLayerTesting,
            shapeOfInputImageForFinalClassificationLayer,
            shapeOfInputImageForFinalClassificationLayerValidation,
            shapeOfInputImageForFinalClassificationLayerTesting] = self.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                                                                    thisPathwayType,
                                                                                                                    thisPathWayNKerns,
                                                                                                                    thisPathWayKernelDimensions,
                                                                                                                    inputImageToPathway,
                                                                                                                    inputImageToPathwayInference,
                                                                                                                    inputImageToPathwayTesting,
                                                                                                                    numberOfFmsOfInputToFirstFcLayer,
                                                                                                                    inputOfFcPathwayImagePartDimsTrain[2:5],
                                                                                                                    inputOfFcPathwayImagePartDimsVal[2:5],
                                                                                                                    inputOfFcPathwayImagePartDimsTest[2:5],
                                                                                                                    True, # This should always be true for FC.
                                                                                                                    rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                                                                    maxPoolingParamsStructure[thisPathwayType],
                                                                                                                    initializationTechniqueClassic0orDelvingInto1,
                                                                                                                    activationFunctionToUseRelu0orPrelu1,
                                                                                                                    dropoutRatesForAllPathways[thisPathwayType],
                                                                                                                    indicesOfLayersToConnectResidualsInOutput[thisPathwayType],
                                                                                                                    indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                                                                    ranksOfLowerRankLayersForEachPathway[thisPathwayType]
                                                                                                                    )

            filterShapeForFinalClassificationLayer = [self.numberOfOutputClasses, fcLayersFMs[-1], 1, 1, 1]
        else : #there is no extra FC layer, just the final FC-softmax.
            inputImageForFinalClassificationLayer = inputImageToPathway
            inputImageForFinalClassificationLayerInference = inputImageToPathwayInference
            inputImageForFinalClassificationLayerTesting = inputImageToPathwayTesting
            shapeOfInputImageForFinalClassificationLayer = inputOfFcPathwayImagePartDimsTrain
            shapeOfInputImageForFinalClassificationLayerValidation = inputOfFcPathwayImagePartDimsVal
            shapeOfInputImageForFinalClassificationLayerTesting = inputOfFcPathwayImagePartDimsTest
            filterShapeForFinalClassificationLayer = [self.numberOfOutputClasses, numberOfFmsOfInputToFirstFcLayer] + firstFcLayerAfterConcatenationKernelShape
            
        #======================Make the FINAL FC CLASSIFICATION LAYER ===================================
        myLogger.print3("DEBUG: Filter Shape of the final-FC-classification Layer: " + str(filterShapeForFinalClassificationLayer))
        myLogger.print3("DEBUG: Shape of input to the Classification layer: (Train) "+str(shapeOfInputImageForFinalClassificationLayer)+\
                         ", (Val) "+str(shapeOfInputImageForFinalClassificationLayerValidation)+", (Test) "+str(shapeOfInputImageForFinalClassificationLayerTesting))
        #The last classification FC layer + softmax.
        dropoutRateForClassificationLayer = 0 if dropoutRatesForAllPathways[self.CNN_PATHWAY_FC] == [] else dropoutRatesForAllPathways[self.CNN_PATHWAY_FC][len(fcLayersFMs)]
        classificationLayer = self._getClassificationLayer()
        classificationLayer.makeLayer(
            rng,
            inputToLayerTrain = inputImageForFinalClassificationLayer,
            inputToLayerVal = inputImageForFinalClassificationLayerInference,
            inputToLayerTest = inputImageForFinalClassificationLayerTesting,                    
            inputToLayerShapeTrain = shapeOfInputImageForFinalClassificationLayer,
            inputToLayerShapeVal = shapeOfInputImageForFinalClassificationLayerValidation,
            inputToLayerShapeTest = shapeOfInputImageForFinalClassificationLayerTesting,
            filterShape = filterShapeForFinalClassificationLayer,
            useBnFlag = rollingAverageForBatchNormalizationOverThatManyBatches > 0,
            rollingAverageForBatchNormalizationOverThatManyBatches = rollingAverageForBatchNormalizationOverThatManyBatches,
            maxPoolingParameters = maxPoolingParamsStructure[self.CNN_PATHWAY_FC][len(fcLayersFMs)],
            initializationTechniqueClassic0orDelvingInto1 = initializationTechniqueClassic0orDelvingInto1,
            activationFunctionToUseRelu0orPrelu1orMinus1ForLinear = activationFunctionToUseRelu0orPrelu1,
            dropoutRate = dropoutRateForClassificationLayer,
            softmaxTemperature = softmaxTemperature
        )
        self.fcLayers.append(classificationLayer)
        self.finalLayer = classificationLayer
        
        myLogger.print3("Finished building the CNN's model.")
        
        