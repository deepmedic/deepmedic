# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import os

from deepmedic.cnnHelpers import calculateReceptiveFieldDimensionsFromKernelsDimListPerLayerForFullyConvCnnWithStrides1
from deepmedic.cnnHelpers import checkReceptiveFieldFineInComparisonToSegmentSize
from deepmedic.cnnHelpers import calculateSubsampledImagePartDimensionsFromImagePartSizePatchSizeAndSubsampleFactor
from deepmedic.cnnHelpers import checkKernDimPerLayerCorrect3dAndNumLayers
from deepmedic.cnnHelpers import checkSubsampleFactorEven

class CreateModelSessionParameters(object) :
    #THE LOGIC WHETHER I GOT A PARAMETER THAT I NEED SHOULD BE IN HERE!
    #Checks for whether needed parameters and types were passed correctly
    def checkLayersForResidualsGivenDoNotInclude1st(self,
                                                    residConnAtLayersNormal,
                                                    residConnAtLayersSubsampled,
                                                    residConnAtLayersFc) :
        if 1 in residConnAtLayersNormal : self.errorResLayer1("Normal")
        if 1 in residConnAtLayersSubsampled : self.errorResLayer1("Subsampled")
        if 1 in residConnAtLayersFc : self.errorResLayer1("Fully Connected")
        
    #To be called from outside too.
    @staticmethod
    def getDefaultSessionName() :
        return getDefaultModelName()
    @staticmethod
    def getDefaultModelName() :
        return "cnnModel"
    @staticmethod
    def defaultDropFcList(numFMsInExtraFcs) :
        numberOfExtraFcs = len(numFMsInExtraFcs)
        dropoutRatesForFcsIncludingClassLayer = []
        if numberOfExtraFcs > 0:
            dropoutForExtraFcs = [0.0] + [0.5]*(numberOfExtraFcs-1) + [0.5]
        else :
            dropoutForExtraFcs = [0.5]
        return dropoutForExtraFcs
    
    #ERRORS
    @staticmethod
    def errorSegmDimensionsSmallerThanReceptiveF(receptiveFieldNormal, segmentDimensions, tr0_val1_inf2) :
        stringsTrainValInference = ["TRAINING", "VALIDATION", "INFERENCE"]
        print "ERROR: The segment-size (input) should be at least as big as the receptive field of the model! The network was made with a receptive field of dimensions: ", receptiveFieldNormal, ". But in the case of :", stringsTrainValInference[tr0_val1_inf2], " the dimensions of the input segment were specified smaller: ", segmentDimensions, ". Please fix this by adjusting number of layer and kernel dimensions! Exiting!"; exit(1)
    @staticmethod
    def errorRequireNumberOfClasses() :
        print "ERROR: Number of classses not specified in the config file, which is required. Please specify in the format: numberOfOutputClasses = 3 (any integer). This number should be including the background class! For instance if the class is binary, set this to 2! Exiting!"; exit(1)
    errReqNumClasses = errorRequireNumberOfClasses
    
    
    @staticmethod
    def errorRequireNumberOfChannels() :
        print "ERROR: Parameter \"numberOfInputChannels\" not specified or specified smaller than 1. Please specify the number of input channels that will be used as input to the CNN, in the format: numberOfInputChannels = number (an integer > 0). Exiting!"; exit(1)
    errReqNumChannels = errorRequireNumberOfChannels
    @staticmethod
    def errorRequireFMsNormalPathwayGreaterThanNothing() :
        print "ERROR: The required parameter \"numberFMsPerLayerNormal\" was either not given, or given an empty list. This parameter should be given in the format: numberFMsPerLayerNormal = [number-of-FMs-layer1, ..., number-of-FMs-layer-N], where each number is an integer greater than zero. It specifies the number of layers (specified by the number of entries in the list) and the number of Feature Maps at each layer of the normal-scale pathway. Please provide and retry. Exiting!"; exit(1)                
    errReqFMsNormal = errorRequireFMsNormalPathwayGreaterThanNothing
    @staticmethod
    def errorRequireKernelDimensionsPerLayerNormal() :
        print "ERROR: The required parameter \"kernelDimPerLayerNormal\" was not provided, or provided incorrectly. It should be provided in the format: kernelDimPerLayerNormal = [ [dim1-of-kernels-in-layer-1, dim2-of-kernels-in-layer-1, dim3-of-kernels-in-layer-1], ..., [dim1-of-kernels-in-layer-N, dim2-of-kernels-in-layer-N, dim3-of-kernels-in-layer-N] ]. It is a list of sublists. One sublist should be provided per layer of the Normal pathway. Thus it should have as many entries as the entries in parameter \"numberFMsPerLayerNormal\". Each sublist should contain 3 integer ODD numbers greater than zero, which should specify the dimensions of the 3-dimensional kernels. For instace: kernelDimPerLayerNormal = [[5,5,5],[3,3,3]] for a pathway with 2 layers, the first of which has 5x5x5 kernels and the second 3x3x3 kernels. Please fix and retry \n WARN: The kernel dimensions should be ODD-NUMBERS. System was not thoroughly tested for kernels of even dimensions! Exiting!"; exit(1)
    errReqKernDimNormal = errorRequireKernelDimensionsPerLayerNormal
    @staticmethod
    def errorRequireKernelDimensionsSubsampled(numFMsPerLayerNormal, numFMsPerLayerSubsampled) :
        print "ERROR: It was requrested to use the 2-scale architecture, with a subsampled pathway. Because of limitations to the developed system, the two pathways must have the save size of receptive field. By default, if \"useSubsampledPathway\" = True, and the parameters \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" are not specified, the second pathway will be constructed symmetrical to the first. However, in this case, \"numberFMsPerLayerSubsampled\" was specified. It was found to have ", len(numFMsPerLayerSubsampled)," entries, which specified this amount of layers in the subsampled pathway. This is different than the number of layers in the Normal pathway, specified to be: ", len(numFMsPerLayerNormal),". In this case, we require you to also provide the parameter \"numberFMsPerLayerSubsampled\", specifying kernel dimensions in the subsampled pathway, in a fashion that results in same size of receptive field as the normal pathway."
        CreateModelSessionParameters.warnForSameReceptiveField()
        print "Exiting!"; exit(1)
    @staticmethod
    def errorRequireKernelDimensionsPerLayerSubsampledCorrect() :
        print "ERROR: The parameter \"kernelDimPerLayerSubsampled\" was not provided, or provided incorrectly. It should be provided in the format: kernelDimPerLayerSubsampled = [ [dim1-of-kernels-in-layer-1, dim2-of-kernels-in-layer-1, dim3-of-kernels-in-layer-1], ..., [dim1-of-kernels-in-layer-N, dim2-of-kernels-in-layer-N, dim3-of-kernels-in-layer-N] ]. It is a list of sublists. One sublist should be provided per layer of the SUBSAMPLED pathway. Thus it should have as many entries as the entries in parameter \"numberFMsPerLayerSubsampled\". (WARN: if the latter is not provided, it is by default taken equal to what specified for \"numberFMsPerLayerNormal\", in order to make the pathways symmetrical). Each sublist should contain 3 integer ODD numbers greater than zero, which should specify the dimensions of the 3-dimensional kernels. For instace: kernelDimPerLayerNormal = [[5,5,5],[3,3,3]] for a pathway with 2 layers, the first of which has 5x5x5 kernels and the second 3x3x3 kernels. Please fix and retry. (WARN: The kernel dimensions should be ODD-NUMBERS. System was not thoroughly tested for kernels of even dimensions!)"
        CreateModelSessionParameters.warnForSameReceptiveField()
        print "Exiting!"; exit(1)
    errReqKernDimNormalCorr = errorRequireKernelDimensionsPerLayerSubsampledCorrect
    @staticmethod
    def errorReceptiveFieldsOfNormalAndSubsampledDifferent(kernDimPerLayerNormal, receptiveFieldSubsampled) :
        print "ERROR: The receptive field of the normal pathway was calculated = ", len(kernDimPerLayerNormal), " while the receptive field of the subsampled pathway was calculated=", len(receptiveFieldSubsampled), ". Because of limitations to the developed system, the two pathways must have the save size of receptive field. Please provide a combination of \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" that gives the same size of field as the normal pathway. If unsure of how to proceed, please ommit specifying \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" in the config file, and the second subsampled pathway will be automatically created to mirror the normal. Else, if you want to just specify the number of Feature Maps in the subsampled, provide \"numberFMsPerLayerSubsampled\" = [num-FMs-layer1, ..., num-FMs-layerN], with N the same number as the normal pathway, and we will then use the same kernel-sizes as the normal pathway. Exiting!"; exit(1)
    @staticmethod
    def errorReqInitializationMethod01() :
        print "ERROR: Parameter \"initializeClassic0orDelving1\" must be given equal to 0 or 1. Omit for default (=1). Exiting!"; exit(1)
    @staticmethod
    def errorReqActivFunction01() :
        print "ERROR: Parameter \"relu0orPrelu1\" must be given equal to 0 or 1. Omit for default (=1). Exiting!"; exit(1)
        
    @staticmethod
    def errorSubFactor3d() :
        print "ERROR: The parameter \"subsampleFactor\" must have 3 entries, one for each of the 3 dimensions. Please provide it in the format: subsampleFactor = [subFactor-dim1, subFactor-dim2, subFactor-dim3]. Each of the entries should be an integer, eg [3,3,3]."
        CreateModelSessionParameters.warnSubFactorOdd()
        print "Exiting!"; exit(1)

    @staticmethod
    def errorRequireSegmentDimensionsTrain() :
        print "ERROR: The parameter \"segmentsDimTrain\" was is required but not given. It specifies the size of the 3D segment that is given as input to the network. It should be at least as large as the receptive field of the network in each dimension. Please specify it in the format: segmentsDimTrain = [dim-1, dim-2, dim-3]. Exiting!"; exit(1)
    errReqSegmDimTrain = errorRequireSegmentDimensionsTrain
    
    @staticmethod
    def errorRequireOptimizer012() :
        print "ERROR: The parameter \"sgd0orAdam1orRms2\" must be given 0,1 or 2. Omit for default. Exiting!"; exit(1)
    @staticmethod
    def errorRequireMomentumClass0Nestov1() :
        print "ERROR: The parameter \"classicMom0OrNesterov1\" must be given 0 or 1. Omit for default. Exiting!"; exit(1)
    @staticmethod
    def errorRequireMomValueBetween01() :
        print "ERROR: The parameter \"momentumValue\" must be given between 0.0 and 1.0 Omit for default. Exiting!"; exit(1)
    @staticmethod
    def errorRequireMomNonNorm0Norm1() :
        print "ERROR: The parameter \"momNonNorm0orNormalized1\" must be given 0 or 1. Omit for default. Exiting!"; exit(1)
    @staticmethod
    def errorRequireBatchSizeTrain() :
        print "ERROR: The parameter \"batchSizeTrain\" was not specified, although required. This parameter specifies how many training-samples (segments) to use to form a batch, on which a single training iteration is performed. The bigger the better, but larger batches add to the memory and computational burden. Depending on the segment-size, the batch size should be smaller (if big segment sizes are used) or larger (if small segment sizes are used). A number between 10 to 100 is suggested. Please specify in the format: batchSizeTrain = 10 (a number). Exiting!"; exit(1)
    errReqBatchSizeTr = errorRequireBatchSizeTrain
    @staticmethod
    def errorResLayer1(strPathwayType) :
        print "ERROR: The parameter \"layersWithResidualConn\" for the [", strPathwayType, "] pathway was specified to include the number 1, ie the 1st layer."
        print "\t This is not an acceptable value, as a residual connection is made between the output of the specified layer and the input of the previous layer. There is no layer before the 1st!"
        print "\t Provide a list that does not iinclude the first layer, eg layersWithResidualConnNormal = [4,6,8], or an empty list [] for no such connections. Exiting!"; exit(1)
        
    @staticmethod
    def warnForSameReceptiveField() :
        print "WARN: Because of limitations to the developed system, the two pathways must have the save size of receptive field. If unsure of how to proceed, please ommit specifying \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" in the config file, and the second subsampled pathway will be automatically created to mirror the normal. Else, if you want to just specify the number of Feature Maps in the subsampled, provide \"numberFMsPerLayerSubsampled\" = [num-FMs-layer1, ..., num-FMs-layerN], with N the same number as the normal pathway, and we will then use the same kernel-sizes as the normal pathway."
    @staticmethod
    def warnSubFactorOdd() :
        print "WARN: The system was only thoroughly tested for ODD subsampling factor! (Eg subsampleFactor = [3,3,3])."
        
    def __init__(self,
                cnnModelName,
                sessionLogger,
                mainOutputAbsFolder,
                folderForSessionCnnModels,
                #===MODEL PARAMETERS===
                numberClasses,
                numberOfInputChannelsNormal,
                #===Normal pathway===
                numFMsNormal,
                kernDimNormal,
                residConnAtLayersNormal,
                lowerRankLayersNormal,
                #==Subsampled pathway==
                useSubsampledBool,
                numFMsSubsampled,
                kernDimSubsampled,
                subsampleFactor,
                residConnAtLayersSubsampled,
                lowerRankLayersSubsampled,
                #==FC Layers====
                numFMsFc,
                kernelDimensionsFirstFcLayer,
                residConnAtLayersFc,
                
                #==Size of Image Segments ==
                segmDimTrain,
                segmDimVal,
                segmDimInfer,
                #== Batch Sizes ==
                batchSizeTrain,
                batchSizeVal,
                batchSizeInfer,
                
                #===Other Architectural Parameters ===
                activationFunction,
                #==Dropout Rates==
                dropNormal,
                dropSubsampled,
                dropFc,
                #==Regularization==
                l1Reg,
                l2Reg,
                #== Weight Initialization==
                initialMethod,

                #== Batch Normalization ==
                bnRollingAverOverThatManyBatches,

                #====Optimization=====
                learningRate,
                optimizerSgd0Adam1Rms2,
                classicMom0Nesterov1,
                momentumValue,
                momNonNormalized0Normalized1,
                #Adam
                b1Adam,
                b2Adam,
                eAdam,
                #Rms
                rhoRms,
                eRms
                ):
        
        #Importants for running session.
        
        self.cnnModelName = cnnModelName if cnnModelName else getDefaultModelName()
        self.sessionLogger = sessionLogger
        self.mainOutputAbsFolder = mainOutputAbsFolder
        self.pathAndFilenameToSaveModel = os.path.abspath(folderForSessionCnnModels + "/" + self.cnnModelName)
        
        #===========MODEL PARAMETERS==========
        self.numberClasses = numberClasses if numberClasses <> None else self.errReqNumClasses()
        self.numberOfInputChannelsNormal = numberOfInputChannelsNormal if numberOfInputChannelsNormal <> None or\
                                                numberOfInputChannelsNormal<1 else self.errReqNumChannels()
                                                
        #===Normal pathway===
        self.numFMsPerLayerNormal = numFMsNormal if numFMsNormal <> None and numFMsNormal > 0 else self.errReqFMsNormal()
        numOfLayers = len(self.numFMsPerLayerNormal)
        self.kernDimPerLayerNormal = kernDimNormal if checkKernDimPerLayerCorrect3dAndNumLayers(kernDimNormal, numOfLayers) else self.errReqKernDimNormal()
        self.receptiveFieldNormal = calculateReceptiveFieldDimensionsFromKernelsDimListPerLayerForFullyConvCnnWithStrides1(self.kernDimPerLayerNormal)
        residConnAtLayersNormal = residConnAtLayersNormal if residConnAtLayersNormal <> None else [] #layer number, starting from 1 for 1st layer. NOT indices.
        lowerRankLayersNormal = lowerRankLayersNormal if lowerRankLayersNormal <> None else [] #layer number, starting from 1 for 1st layer. NOT indices.
        
        #==Subsampled pathway==
        self.useSubsampledBool = useSubsampledBool if useSubsampledBool <> None else False
        if not self.useSubsampledBool :
            self.numFMsPerLayerSubsampled = []
            self.kernDimPerLayerSubsampled = []
            self.receptiveFieldSubsampled = []
            self.subsampleFactor = []
            residConnAtLayersSubsampled = []
            lowerRankLayersSubsampled = []
            
        else :
            self.numFMsPerLayerSubsampled = numFMsSubsampled if numFMsSubsampled <> None else self.numFMsPerLayerNormal
            if kernDimSubsampled == None and\
                                        len(self.numFMsPerLayerSubsampled) == len(self.numFMsPerLayerNormal) :
                self.kernDimPerLayerSubsampled = self.kernDimPerLayerNormal
                self.receptiveFieldSubsampled = self.receptiveFieldNormal
            elif kernDimSubsampled == None and\
                                        len(self.numFMsPerLayerSubsampled) <> len(self.numFMsPerLayerNormal) : #user specified subsampled layers.
                self.errorRequireKernelDimensionsSubsampled(self.kernDimPerLayerNormal, numFMsSubsampled)
            # kernDimSubsampled was specified. Now it's going to be tricky to make sure everything alright.
            elif not checkKernDimPerLayerCorrect3dAndNumLayers(kernDimSubsampled, len(self.numFMsPerLayerSubsampled)) :
                self.errReqKernDimNormalCorr()
            else : #kernel dimensions specified and are correct (3d, same number of layers as subsampled specified). Need to check the two receptive fields and make sure they are correct.
                self.kernDimPerLayerSubsampled = kernDimSubsampled
                self.receptiveFieldSubsampled = calculateReceptiveFieldDimensionsFromKernelsDimListPerLayerForFullyConvCnnWithStrides1(self.kernDimPerLayerSubsampled)
                if self.receptiveFieldNormal <> self.receptiveFieldSubsampled :
                    self.errorReceptiveFieldsOfNormalAndSubsampledDifferent(self.receptiveFieldNormal, self.receptiveFieldSubsampled)
                #Everything alright, finally. Proceed safely...
            self.subsampleFactor = subsampleFactor if subsampleFactor <> None else [3,3,3]
            if len(self.subsampleFactor) <> 3 :
                self.errorSubFactor3d()
            if not checkSubsampleFactorEven(self.subsampleFactor) :
                self.warnSubFactorOdd()
            residConnAtLayersSubsampled = residConnAtLayersSubsampled if residConnAtLayersSubsampled <> None else residConnAtLayersNormal
            lowerRankLayersSubsampled = lowerRankLayersSubsampled if lowerRankLayersSubsampled <> None else lowerRankLayersNormal
            
        #==FC Layers==
        self.numFMsInExtraFcs = numFMsFc if numFMsFc <> None else []
        self.kernelDimensionsFirstFcLayer = kernelDimensionsFirstFcLayer if kernelDimensionsFirstFcLayer <> None else [1,1,1]
        assert len(self.kernelDimensionsFirstFcLayer) == 3 and (False not in [ dim > 0 for dim in self.kernelDimensionsFirstFcLayer] )
        residConnAtLayersFc = residConnAtLayersFc if residConnAtLayersFc <> None else []
                                        
        #==Size of Image Segments ==
        self.segmDimNormalTrain = segmDimTrain if segmDimTrain <> None else self.errReqSegmDimTrain()
        self.segmDimNormalVal = segmDimVal if segmDimVal <> None else self.receptiveFieldNormal
        self.segmDimNormalInfer = segmDimInfer if segmDimInfer <> None else self.segmDimNormalTrain
        for (tr0_val1_inf2, segmentDimensions) in [ (0,self.segmDimNormalTrain), (1,self.segmDimNormalVal), (2,self.segmDimNormalInfer) ] :
            if not checkReceptiveFieldFineInComparisonToSegmentSize(self.receptiveFieldNormal, segmentDimensions) :
                self.errorSegmDimensionsSmallerThanReceptiveF(self.receptiveFieldNormal, segmentDimensions, tr0_val1_inf2)
                
        #=== Batch Sizes ===
        self.batchSizeTrain = batchSizeTrain if batchSizeTrain <> None else self.errReqBatchSizeTr()
        self.batchSizeVal = batchSizeVal if batchSizeVal <> None else self.batchSizeTrain
        self.batchSizeInfer = batchSizeInfer if batchSizeInfer <> None else self.batchSizeTrain
        
        #=== Dropout rates ===
        self.dropNormal = dropNormal if dropNormal <> None else []
        self.dropSubsampled = dropSubsampled if dropSubsampled <> None else []
        self.dropFc = dropFc if dropFc <> None else self.defaultDropFcList(self.numFMsInExtraFcs) #default = [0.0, 0.5, ..., 0.5]
        self.dropoutRatesForAllPathways = [self.dropNormal, self.dropSubsampled, self.dropFc, []]
        
        #==Regularization==
        self.l1Reg = l1Reg if l1Reg <> None else 0.000001
        self.l2Reg = l2Reg if l2Reg <> None else 0.0001
        
        #== Weight Initialization==
        self.initialMethodClassic0Delving1 = initialMethod if initialMethod <> None else 1
        if not self.initialMethodClassic0Delving1 in [0,1]:
            self.errorReqInitializationMethod01()
        #== Activation Function ==
        self.activationFunctionRelu0Prelu1 = activationFunction if activationFunction <> None else 1
        if not self.activationFunctionRelu0Prelu1 in [0,1]:
            self.errorReqActivFunction01()
            
        #==BATCH NORMALIZATION==
        self.applyBnToInputOfPathways = [False, False, "Placeholder", False] # the 3 entry, for FC, is always True internally.
        self.bnRollingAverOverThatManyBatches = bnRollingAverOverThatManyBatches if bnRollingAverOverThatManyBatches <> None else 60
        
        #====Optimization=====
        self.learningRate = learningRate if learningRate <> None else 0.001
        self.optimizerSgd0Adam1Rms2 = optimizerSgd0Adam1Rms2 if optimizerSgd0Adam1Rms2 <> None else 2
        if self.optimizerSgd0Adam1Rms2 == 0 :
            self.b1Adam = "placeholder"; self.b2Adam = "placeholder"; self.eAdam = "placeholder";
            self.rhoRms = "placeholder"; self.eRms = "placeholder";
        elif self.optimizerSgd0Adam1Rms2 == 1 :
            self.b1Adam = b1Adam if b1Adam <> None else 0.9 #default in paper and seems good
            self.b2Adam = b2Adam if b2Adam <> None else 0.999 #default in paper and seems good
            self.eAdam = eAdam if eAdam else 10**(-8)
            self.rhoRms = "placeholder"; self.eRms = "placeholder";
        elif self.optimizerSgd0Adam1Rms2 == 2 :
            self.b1Adam = "placeholder"; self.b2Adam = "placeholder"; self.eAdam = "placeholder";
            self.rhoRms = rhoRms if rhoRms <> None else 0.9 #default in paper and seems good
            self.eRms = eRms if eRms <> None else 10**(-4) # 1e-6 was the default in the paper, but blew up the gradients in first try. Never tried 1e-5 yet.
        else :
            self.errorRequireOptimizer012()
            
        self.classicMom0Nesterov1 = classicMom0Nesterov1 if classicMom0Nesterov1 <> None else 1
        if self.classicMom0Nesterov1 not in [0,1]:
            self.errorRequireMomentumClass0Nestov1()
        self.momNonNormalized0Normalized1 = momNonNormalized0Normalized1 if momNonNormalized0Normalized1 <> None else 1
        if self.momNonNormalized0Normalized1 not in [0,1] :
            self.errorRequireMomNonNorm0Norm1()
        self.momentumValue = momentumValue if momentumValue <> None else 0.6
        if self.momentumValue < 0. or self.momentumValue > 1:
            self.errorRequireMomValueBetween01()
            
        #==============CALCULATED=====================
        if self.useSubsampledBool :
            self.segmDimSubsampledTrain = calculateSubsampledImagePartDimensionsFromImagePartSizePatchSizeAndSubsampleFactor(self.segmDimNormalTrain, self.receptiveFieldNormal, self.subsampleFactor)
            self.segmDimSubsampledVal = calculateSubsampledImagePartDimensionsFromImagePartSizePatchSizeAndSubsampleFactor(self.segmDimNormalVal, self.receptiveFieldNormal, self.subsampleFactor)
            self.segmDimSubsampledInfer = calculateSubsampledImagePartDimensionsFromImagePartSizePatchSizeAndSubsampleFactor(self.segmDimNormalInfer, self.receptiveFieldNormal, self.subsampleFactor)
        else :
            self.segmDimSubsampledTrain = []; self.segmDimSubsampledVal = []; self.segmDimSubsampledInfer = [];
            
        # Residual Connections backwards, per pathway type :
        self.checkLayersForResidualsGivenDoNotInclude1st(residConnAtLayersNormal, residConnAtLayersSubsampled, residConnAtLayersFc)
        # The following variable passed to the system takes indices, ie number starts from 0. User specifies from 1.
        self.indicesOfLayersToConnectResidualsInOutput = [  [ layerNum - 1 for layerNum in residConnAtLayersNormal ],
                                                            [ layerNum - 1 for layerNum in residConnAtLayersSubsampled ],
                                                            [ layerNum - 1 for layerNum in residConnAtLayersFc ],
                                                            []
                                                        ]
        
        self.indicesOfLowerRankLayersPerPathway = [[ layerNum - 1 for layerNum in lowerRankLayersNormal ],
                                                   [ layerNum - 1 for layerNum in lowerRankLayersSubsampled ],
                                                   [], #FC doesn't make sense to be lower rank. It's 1x1x1 anyway.
                                                   []
                                                   ]
        self.ranksOfLowerRankLayersForEachPathway = [[ 2 for layer_i in self.indicesOfLowerRankLayersPerPathway[0] ],
                                                     [ 2 for layer_i in self.indicesOfLowerRankLayersPerPathway[1] ],
                                                     [],
                                                     []
                                                     ]
        #============= HIDDENS ======================
        self.costFunctionLetter = "L"
        
        self.numberOfInputChannelsSubsampled = self.numberOfInputChannelsNormal
        
        #----for the zoomed-in pathway----
        self.zoomedInPatchDimensions = [9, 9, 9]
        self.nkernsZoomedIn1 = [] #[15,15,15,30]
        self.kernelDimensionsZoomedIn1 = [[3,3,3], [3,3,3], [3,3,3], [3,3,3]]
        
        #MultiscaleConnections:
        self.convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes = [ [], [] ] #a sublist for each pathway. Starts from 0 index. Give a sublist, even empty for no connections.
        #... It's ok if I dont have a 2nd path but still give a 2nd sublist, it's controlled by nkernsSubsampled.
        
        #-------POOLING---------- (not fully supported currently)
        #One entry per pathway-type. leave [] if the pathway does not exist or there is no mp there AT ALL.
        #Inside each entry, put a list FOR EACH LAYER. It should be [] for the layer if no mp there. But FOR EACH LAYER.
        #MP is applied >>AT THE INPUT of the layer<<. To use mp to a layer, put a list of [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode] which give the dimensions of the mp window, the stride, how many times to mirror the last slot at each dimension for padding (give 0 for none), the mode (usually 'max' pool). Eg [[2,2,2],[1,1,1]] or [[2,2,2],[2,2,2]] usually.
        self.maxPoolingParamsStructure = [ #If a pathway is not used, put an empty list in the first dimension entry. 
                                        [ [] for layeri in xrange(len(self.numFMsPerLayerNormal)) ], #[[[2,2,2], [1,1,1], [1,1,1], 'max'], [],[],[],[],[],[], []], #first pathway
                                        [ [] for layeri in xrange(len(self.numFMsPerLayerSubsampled)) ], #second pathway
                                        [ [] for layeri in xrange(len(self.numFMsInExtraFcs) + 1) ], #FC. This should NEVER be used for segmentation. Possible for classification though.
                                        [[],[],[],[]] #zoomed in pathway.
                                        ]
        
        self.softmaxTemperature = 1.0 #Higher temperatures make the probabilities LESS distinctable. Actions have more similar probabilities. 
        
    #Public
    def getPathAndFilenameToSaveModel(self) :
        return self.pathAndFilenameToSaveModel
    
    def printParametersOfThisSession(self) :
        logPrint = self.sessionLogger.print3
        logPrint("=============================================================")
        logPrint("=============== PARAMETERS FOR MODEL CREATION ===============")
        logPrint("=============================================================")
        logPrint("CNN model's name = " + str(self.cnnModelName))
        logPrint("Main output folder = " + str(self.mainOutputAbsFolder))
        logPrint("Path and filename to save model = " + str(self.pathAndFilenameToSaveModel))
        
        logPrint("~~~~~~~~~~~~~~~~~~Model parameters~~~~~~~~~~~~~~~~")
        logPrint("Number of Classes (including background) = " + str(self.numberClasses))
        logPrint("~~Normal Pathway~~")
        logPrint("Number of Input Channels = " + str(self.numberOfInputChannelsNormal))
        logPrint("Number of Layers = " + str(len(self.numFMsPerLayerNormal)))
        logPrint("Number of Feature Maps per layer = " + str(self.numFMsPerLayerNormal))
        logPrint("Kernel Dimensions per layer = " + str(self.kernDimPerLayerNormal))
        logPrint("Receptive Field = " + str(self.receptiveFieldNormal))
        logPrint("Residual connections added at the output of layers (indices from 0) = " + str(self.indicesOfLayersToConnectResidualsInOutput[0]))
        logPrint("Layers that will be made of Lower Rank (indices from 0) = " + str(self.indicesOfLowerRankLayersPerPathway[0]))
        logPrint("Lower Rank layers will be made of rank = " + str(self.ranksOfLowerRankLayersForEachPathway[0]))
        #logPrint("Parameters for pooling before convolutions in this pathway = " +  + str(self.maxPoolingParamsStructure[0]))
        
        logPrint("~~Subsampled Pathway~~")
        logPrint("Use subsampled Pathway = " + str(self.useSubsampledBool))
        logPrint("Number of Layers = " + str(len(self.numFMsPerLayerSubsampled)))
        logPrint("Number of Feature Maps per layer = " + str(self.numFMsPerLayerSubsampled))
        logPrint("Kernel Dimensions per layer = " + str(self.kernDimPerLayerSubsampled))
        logPrint("Receptive Field = " + str(self.receptiveFieldSubsampled))
        logPrint("Subsampling Factor = " + str(self.subsampleFactor))
        logPrint("Residual connections added at the output of layers (indices from 0) = " + str(self.indicesOfLayersToConnectResidualsInOutput[1]))
        logPrint("Layers that will be made of Lower Rank (indices from 0) = " + str(self.indicesOfLowerRankLayersPerPathway[1]))
        logPrint("Lower Rank layers will be made of rank = " + str(self.ranksOfLowerRankLayersForEachPathway[1]))
        #logPrint("Parameters for pooling before convolutions in this pathway = " +  + str(self.maxPoolingParamsStructure[1]))
        
        logPrint("~~Fully Connected Pathway~~")
        logPrint("Number of additional FC layers (Excluding the Classif. Layer) = " + str(len(self.numFMsInExtraFcs)))
        logPrint("Number of Feature Maps in the additional FC layers = " + str(self.numFMsInExtraFcs))
        logPrint("Residual connections added at the output of layers (indices from 0) = " + str(self.indicesOfLayersToConnectResidualsInOutput[2]))
        logPrint("Layers that will be made of Lower Rank (indices from 0) = " + str(self.indicesOfLowerRankLayersPerPathway[2]))
        #logPrint("Parameters for pooling before convolutions in this pathway = " +  + str(self.maxPoolingParamsStructure[2]))
        logPrint("Dimensions of Kernels in the 1st FC layer (Classif. layer if no hidden FCs used) = " + str(self.kernelDimensionsFirstFcLayer))
        
        logPrint("~~Size Of Image Segments~~")
        logPrint("Size of Segments for Training = " + str(self.segmDimNormalTrain))
        logPrint("Size of Segments for Validation = " + str(self.segmDimNormalVal))
        logPrint("Size of Segments for Testing = " + str(self.segmDimNormalInfer))
        logPrint("~~Size Of Image Segments (Subsampled, auto-calculated)~~")
        logPrint("Size of Segments for Training (Subsampled) = " + str(self.segmDimSubsampledTrain))
        logPrint("Size of Segments for Validation (Subsampled) = " + str(self.segmDimSubsampledVal))
        logPrint("Size of Segments for Testing (Subsampled) = " + str(self.segmDimSubsampledInfer))
        
        logPrint("~~Batch Sizes~~")
        logPrint("Batch Size for Training = " + str(self.batchSizeTrain))
        logPrint("Batch Size for Validation = " + str(self.batchSizeVal))
        logPrint("Batch Size for Testing = " + str(self.batchSizeInfer))
        
        logPrint("~~Dropout Rates~~")
        logPrint("Drop.R. for each layer in Normal Pathway = " + str(self.dropoutRatesForAllPathways[0]))
        logPrint("Drop.R. for each layer in Subsampled Pathway = " + str(self.dropoutRatesForAllPathways[1]))
        logPrint("Drop.R. for each layer in FC Pathway (additional FC layers + Classific.Layer at end) = " + str(self.dropoutRatesForAllPathways[2]))
        
        logPrint("~~L1/L2 Regularization~~")
        logPrint("L1 Regularization term = " + str(self.l1Reg))
        logPrint("L2 Regularization term = " + str(self.l2Reg))
        
        logPrint("~~Weight Initialization~~")
        logPrint("Classic random N(0,0.01) initialization (0), or ala \"Delving Into Rectifier\" (1) = " + str(self.initialMethodClassic0Delving1))
        
        logPrint("~~Activation Function~~")
        logPrint("ReLU (0), or PReLU (1) = " + str(self.activationFunctionRelu0Prelu1))
        
        logPrint("~~Batch Normalization~~")
        logPrint("Apply BN straight on pathways' inputs (eg straight on segments) = " + str(self.applyBnToInputOfPathways))
        logPrint("Batch Normalization uses a rolling average for inference, over this many batches = " + str(self.bnRollingAverOverThatManyBatches))
        
        logPrint("~~Optimization~~")
        logPrint("Initial Learning rate = " + str(self.learningRate))
        logPrint("Optimizer to use: SGD(0), Adam(1), RmsProp(2) = " + str(self.optimizerSgd0Adam1Rms2))
        logPrint("Parameters for Adam: b1= " + str(self.b1Adam) + ", b2=" + str(self.b2Adam) + ", e= " + str(self.eAdam) )
        logPrint("Parameters for RmsProp: rho= " + str(self.rhoRms) + ", e= " + str(self.eRms) )
        logPrint("Momentum Type: Classic (0) or Nesterov (1) = " + str(self.classicMom0Nesterov1))
        logPrint("Momentum Non-Normalized (0) or Normalized (1) = " + str(self.momNonNormalized0Normalized1))
        logPrint("Momentum Value = " + str(self.momentumValue))
        
        logPrint("========== Done with printing session's parameters ==========")
        logPrint("=============================================================")
        
    def getTupleForCnnCreation(self) :
        borrowFlag = True
        dataTypeX = 'float32'
        
        cnnCreationTuple = (
                        self.sessionLogger,
                        self.cnnModelName,
                        #=== Model Parameters ===
                        self.numberClasses,
                        self.numberOfInputChannelsNormal,
                        self.numberOfInputChannelsSubsampled,
                        #=== Normal Pathway ===
                        self.numFMsPerLayerNormal, #ONLY for the convolutional layers, NOT the final convFCSoftmaxLayer!
                        self.kernDimPerLayerNormal,
                        self.receptiveFieldNormal, # Should be automatically calculate it in the CNN.
                        #=== Subsampled Pathway ===
                        self.numFMsPerLayerSubsampled,
                        self.kernDimPerLayerSubsampled,
                        self.subsampleFactor,
                        
                        #=== Zoomed-in pathway === # Deprecated
                        self.zoomedInPatchDimensions,
                        self.nkernsZoomedIn1,
                        self.kernelDimensionsZoomedIn1,
                        
                        #=== FC Layers ===
                        self.numFMsInExtraFcs,
                        self.kernelDimensionsFirstFcLayer,
                        self.softmaxTemperature,
                        
                        #=== Other Architectural params ===
                        self.activationFunctionRelu0Prelu1,
                        #---Residual Connections----
                        self.indicesOfLayersToConnectResidualsInOutput,
                        #--Lower Rank Layer Per Pathway---
                        self.indicesOfLowerRankLayersPerPathway,
                        self.ranksOfLowerRankLayersForEachPathway,
                        #---Pooling---
                        self.maxPoolingParamsStructure,
                        #--- Skip Connections --- #Deprecated, not used/supported
                        self.convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes,
                        
                        #==Size of Image Segments ==
                        self.segmDimNormalTrain,
                        self.segmDimNormalVal,
                        self.segmDimNormalInfer,
                        self.segmDimSubsampledTrain, # Should be automatically calculate it in the CNN.
                        self.segmDimSubsampledVal,
                        self.segmDimSubsampledInfer,
                        
                        #=== Batch Sizes ===
                        self.batchSizeTrain,
                        self.batchSizeVal,
                        self.batchSizeInfer,
                        
                        #=== Others ====
                        #Dropout
                        self.dropoutRatesForAllPathways,
                        #Initialization
                        self.initialMethodClassic0Delving1,
                        #Batch Normalization
                        self.applyBnToInputOfPathways,
                        self.bnRollingAverOverThatManyBatches,
                        
                        #=== various ===
                        borrowFlag,
                        dataTypeX
                        )
        
        return cnnCreationTuple
    
    def getTupleForInitializingTrainingState(self) :

        initializingTrainingStateTuple = (
                        self.sessionLogger,
                        
                        "all",
                        #=====COST FUNCTION=====
                        self.costFunctionLetter,
                        
                        #=====OPTIMIZATION=====
                        self.learningRate,
                        self.optimizerSgd0Adam1Rms2,
                        self.classicMom0Nesterov1, 
                        self.momentumValue,
                        self.momNonNormalized0Normalized1,
                        self.b1Adam,
                        self.b2Adam,
                        self.eAdam,
                        self.rhoRms,
                        self.eRms,
                        # Regularisation
                        self.l1Reg,
                        self.l2Reg
                        )
        
        return initializingTrainingStateTuple
    
    def getTupleForCompilationOfTrainFunc(self) :
        trainFunctionCompilationTuple = ( self.sessionLogger, )
        return trainFunctionCompilationTuple
    
    def getTupleForCompilationOfValFunc(self) :
        valFunctionCompilationTuple = ( self.sessionLogger, )
        return valFunctionCompilationTuple
    
    def getTupleForCompilationOfTestFunc(self) :
        testFunctionCompilationTuple = ( self.sessionLogger, )
        return testFunctionCompilationTuple
