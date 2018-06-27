# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

from deepmedic.neuralnet.utils import calcRecFieldFromKernDimListPerLayerWhenStrides1, checkRecFieldVsSegmSize, checkKernDimPerLayerCorrect3dAndNumLayers, checkSubsampleFactorEven


class ModelParameters(object) :
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
        return "deepmedic"
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
        print("ERROR: The segment-size (input) should be at least as big as the receptive field of the model! The network was made with a receptive field of dimensions: ", receptiveFieldNormal, ". But in the case of :", stringsTrainValInference[tr0_val1_inf2], " the dimensions of the input segment were specified smaller: ", segmentDimensions, ". Please fix this by adjusting number of layer and kernel dimensions! Exiting!"); exit(1)
    @staticmethod
    def errorRequireNumberOfClasses() :
        print("ERROR: Number of classses not specified in the config file, which is required. Please specify in the format: numberOfOutputClasses = 3 (any integer). This number should be including the background class! For instance if the class is binary, set this to 2! Exiting!"); exit(1)
    errReqNumClasses = errorRequireNumberOfClasses
    
    
    @staticmethod
    def errorRequireNumberOfChannels() :
        print("ERROR: Parameter \"numberOfInputChannels\" not specified or specified smaller than 1. Please specify the number of input channels that will be used as input to the CNN, in the format: numberOfInputChannels = number (an integer > 0). Exiting!"); exit(1)
    errReqNumChannels = errorRequireNumberOfChannels
    @staticmethod
    def errorRequireFMsNormalPathwayGreaterThanNothing() :
        print("ERROR: The required parameter \"numberFMsPerLayerNormal\" was either not given, or given an empty list. This parameter should be given in the format: numberFMsPerLayerNormal = [number-of-FMs-layer1, ..., number-of-FMs-layer-N], where each number is an integer greater than zero. It specifies the number of layers (specified by the number of entries in the list) and the number of Feature Maps at each layer of the normal-scale pathway. Please provide and retry. Exiting!"); exit(1)                
    errReqFMsNormal = errorRequireFMsNormalPathwayGreaterThanNothing
    @staticmethod
    def errorRequireKernelDimensionsPerLayerNormal() :
        print("ERROR: The required parameter \"kernelDimPerLayerNormal\" was not provided, or provided incorrectly. It should be provided in the format: kernelDimPerLayerNormal = [ [dim1-of-kernels-in-layer-1, dim2-of-kernels-in-layer-1, dim3-of-kernels-in-layer-1], ..., [dim1-of-kernels-in-layer-N, dim2-of-kernels-in-layer-N, dim3-of-kernels-in-layer-N] ]. It is a list of sublists. One sublist should be provided per layer of the Normal pathway. Thus it should have as many entries as the entries in parameter \"numberFMsPerLayerNormal\". Each sublist should contain 3 integer ODD numbers greater than zero, which should specify the dimensions of the 3-dimensional kernels. For instace: kernelDimPerLayerNormal = [[5,5,5],[3,3,3]] for a pathway with 2 layers, the first of which has 5x5x5 kernels and the second 3x3x3 kernels. Please fix and retry \n WARN: The kernel dimensions should be ODD-NUMBERS. System was not thoroughly tested for kernels of even dimensions! Exiting!"); exit(1)
    errReqKernDimNormal = errorRequireKernelDimensionsPerLayerNormal
    @staticmethod
    def errorRequireKernelDimensionsSubsampled(numFMsPerLayerNormal, numFMsPerLayerSubsampled) :
        print("ERROR: It was requested to use the 2-scale architecture, with a subsampled pathway. Because of limitations to the developed system, the two pathways must have the save size of receptive field. By default, if \"useSubsampledPathway\" = True, and the parameters \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" are not specified, the second pathway will be constructed symmetrical to the first. However, in this case, \"numberFMsPerLayerSubsampled\" was specified. It was found to have ", len(numFMsPerLayerSubsampled)," entries, which specified this amount of layers in the subsampled pathway. This is different than the number of layers in the Normal pathway, specified to be: ", len(numFMsPerLayerNormal),". In this case, we require you to also provide the parameter \"numberFMsPerLayerSubsampled\", specifying kernel dimensions in the subsampled pathway, in a fashion that results in same size of receptive field as the normal pathway.")
        ArchitectureParameters.warnForSameReceptiveField()
        print("Exiting!"); exit(1)
    @staticmethod
    def errorRequireKernelDimensionsPerLayerSubsampledCorrect() :
        print("ERROR: The parameter \"kernelDimPerLayerSubsampled\" was not provided, or provided incorrectly. It should be provided in the format: kernelDimPerLayerSubsampled = [ [dim1-of-kernels-in-layer-1, dim2-of-kernels-in-layer-1, dim3-of-kernels-in-layer-1], ..., [dim1-of-kernels-in-layer-N, dim2-of-kernels-in-layer-N, dim3-of-kernels-in-layer-N] ]. It is a list of sublists. One sublist should be provided per layer of the SUBSAMPLED pathway. Thus it should have as many entries as the entries in parameter \"numberFMsPerLayerSubsampled\". (WARN: if the latter is not provided, it is by default taken equal to what specified for \"numberFMsPerLayerNormal\", in order to make the pathways symmetrical). Each sublist should contain 3 integer ODD numbers greater than zero, which should specify the dimensions of the 3-dimensional kernels. For instace: kernelDimPerLayerNormal = [[5,5,5],[3,3,3]] for a pathway with 2 layers, the first of which has 5x5x5 kernels and the second 3x3x3 kernels. Please fix and retry. (WARN: The kernel dimensions should be ODD-NUMBERS. System was not thoroughly tested for kernels of even dimensions!)")
        ArchitectureParameters.warnForSameReceptiveField()
        print("Exiting!"); exit(1)
    errReqKernDimNormalCorr = errorRequireKernelDimensionsPerLayerSubsampledCorrect
    @staticmethod
    def errorReceptiveFieldsOfNormalAndSubsampledDifferent(kernDimPerLayerNormal, receptiveFieldSubsampled) :
        print("ERROR: The receptive field of the normal pathway was calculated = ", len(kernDimPerLayerNormal), " while the receptive field of the subsampled pathway was calculated=", len(receptiveFieldSubsampled), ". Because of limitations to the developed system, the two pathways must have the save size of receptive field. Please provide a combination of \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" that gives the same size of field as the normal pathway. If unsure of how to proceed, please ommit specifying \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" in the config file, and the second subsampled pathway will be automatically created to mirror the normal. Else, if you want to just specify the number of Feature Maps in the subsampled, provide \"numberFMsPerLayerSubsampled\" = [num-FMs-layer1, ..., num-FMs-layerN], with N the same number as the normal pathway, and we will then use the same kernel-sizes as the normal pathway. Exiting!"); exit(1)
    @staticmethod
    def errorReqInitializationMethod() :
        print("ERROR: Parameter \"convWeightsInit\" has been given invalid value. Exiting!"); exit(1)
    @staticmethod
    def errorReqActivFunction() :
        print("ERROR: Parameter \"activationFunction\" has been given invalid value. Exiting!"); exit(1)
        
    @staticmethod
    def errReqSameNumOfLayersPerSubPathway():
        print("ERROR: Parameter \"numberFMsPerLayerSubsampled\" has been given as a list of sublists of integers. This triggers the construction of multiple low-scale pathways.")
        print("\tHowever currently this functionality requires that the same number of layers are used in both pathways (limitation in the code).")
        print("\tUser specified in \"numberFMsPerLayerSubsampled\" sublists of different length. Each list should have the same lenght, as many as the wanted number of layers. Please adress this.")
        print("Exiting!"); exit(1)
    
    @staticmethod
    def errorSubFactor3d() :
        print("ERROR: The parameter \"subsampleFactor\" must have 3 entries, one for each of the 3 dimensions. Please provide it in the format: subsampleFactor = [subFactor-dim1, subFactor-dim2, subFactor-dim3]. Each of the entries should be an integer, eg [3,3,3].")
        ArchitectureParameters.warnSubFactorOdd()
        print("Exiting!"); exit(1)

    @staticmethod
    def errorRequireSegmentDimensionsTrain() :
        print("ERROR: The parameter \"segmentsDimTrain\" was is required but not given. It specifies the size of the 3D segment that is given as input to the network. It should be at least as large as the receptive field of the network in each dimension. Please specify it in the format: segmentsDimTrain = [dim-1, dim-2, dim-3]. Exiting!"); exit(1)
    errReqSegmDimTrain = errorRequireSegmentDimensionsTrain
    
    @staticmethod
    def errorRequireBatchSizeTrain() :
        print("ERROR: The parameter \"batchSizeTrain\" was not specified, although required. This parameter specifies how many training-samples (segments) to use to form a batch, on which a single training iteration is performed. The bigger the better, but larger batches add to the memory and computational burden. Depending on the segment-size, the batch size should be smaller (if big segment sizes are used) or larger (if small segment sizes are used). A number between 10 to 100 is suggested. Please specify in the format: batchSizeTrain = 10 (a number). Exiting!"); exit(1)
    errReqBatchSizeTr = errorRequireBatchSizeTrain
    @staticmethod
    def errorResLayer1(strPathwayType) :
        print("ERROR: The parameter \"layersWithResidualConn\" for the [", strPathwayType, "] pathway was specified to include the number 1, ie the 1st layer.")
        print("\t This is not an acceptable value, as a residual connection is made between the output of the specified layer and the input of the previous layer. There is no layer before the 1st!")
        print("\t Provide a list that does not iinclude the first layer, eg layersWithResidualConnNormal = [4,6,8], or an empty list [] for no such connections. Exiting!"); exit(1)
        
    @staticmethod
    def warnForSameReceptiveField() :
        print("WARN: Because of limitations to the developed system, the two pathways must have the save size of receptive field. If unsure of how to proceed, please ommit specifying \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" in the config file, and the second subsampled pathway will be automatically created to mirror the normal. Else, if you want to just specify the number of Feature Maps in the subsampled, provide \"numberFMsPerLayerSubsampled\" = [num-FMs-layer1, ..., num-FMs-layerN], with N the same number as the normal pathway, and we will then use the same kernel-sizes as the normal pathway.")
    @staticmethod
    def warnSubFactorOdd() :
        print("WARN: The system was only thoroughly tested for ODD subsampling factor! (Eg subsampleFactor = [3,3,3]).")
        
    # OTHERS
    @staticmethod
    def changeDatastructureToListOfListsForSecondaryPathwaysIfNeeded(listFromConfig) :
        # subsampleFactorFromConfig: whatever given in the config by the user (except None). 
        if not isinstance(listFromConfig, list) :
            print("ERROR: variable \"", listFromConfig, "\" given in modelConfig.cfg should be either a list of integers, or a list of lists of integers, in case multiple lower-scale pathways are wanted. Please correct it. Exiting."); exit(1)
        allElementsAreLists = True
        noElementIsList = True
        for element in listFromConfig :
            if isinstance(element, list) :
                noElementIsList = False
            else :
                allElementsAreLists = False
        if not (allElementsAreLists or noElementIsList) : #some are lists and some are not
            print("ERROR: variable \"", listFromConfig, "\" given in modelConfig.cfg should be either a list of integers, or a list of lists of integers, in case multiple lower-scale pathways are wanted. Please correct it. Exiting."); exit(1)
        elif noElementIsList :
            #Seems ok, but the structure is not a list of lists. It's probably the old type, eg [3,3,3]. Lets change to list of lists.
            return [ listFromConfig ]
        else :
            return listFromConfig
           
    def checkThatSublistsHaveSameLength(self, listWithSublists):
        if len(listWithSublists) == 0 :
            return True
        lengthOfFirst = len(listWithSublists[0])
        for subList_i in range( len(listWithSublists) ) :
            if lengthOfFirst != len(listWithSublists[subList_i]) :
                return False
        return True
           
           
    def __init__(   self,
                    log,
                    cfg
                    ):
        
        self.log = log        
        self.cnnModelName = cfg[cfg.MODEL_NAME] if cfg[cfg.MODEL_NAME] is not None else getDefaultModelName()
        
        #===========MODEL PARAMETERS==========
        self.numberClasses = cfg[cfg.NUM_CLASSES] if cfg[cfg.NUM_CLASSES] is not None else self.errReqNumClasses()
        self.numberOfInputChannelsNormal = cfg[cfg.NUM_INPUT_CHANS] if cfg[cfg.NUM_INPUT_CHANS] is not None else self.errReqNumChannels()
        assert self.numberOfInputChannelsNormal > 0
        #===Normal pathway===
        self.numFMsPerLayerNormal = cfg[cfg.N_FMS_NORM] if cfg[cfg.N_FMS_NORM] is not None and len(cfg[cfg.N_FMS_NORM]) > 0 else self.errReqFMsNormal()
        numOfLayers = len(self.numFMsPerLayerNormal)
        self.kernDimPerLayerNormal = cfg[cfg.KERN_DIM_NORM] if checkKernDimPerLayerCorrect3dAndNumLayers(cfg[cfg.KERN_DIM_NORM], numOfLayers) else self.errReqKernDimNormal()
        self.receptiveFieldNormal = calcRecFieldFromKernDimListPerLayerWhenStrides1(self.kernDimPerLayerNormal) # Just for COMPATIBILITY CHECKS!
        residConnAtLayersNormal = cfg[cfg.RESID_CONN_LAYERS_NORM] if cfg[cfg.RESID_CONN_LAYERS_NORM] is not None else [] #layer number, starting from 1 for 1st layer. NOT indices.
        lowerRankLayersNormal = cfg[cfg.LOWER_RANK_LAYERS_NORM] if cfg[cfg.LOWER_RANK_LAYERS_NORM] is not None else [] #layer number, starting from 1 for 1st layer. NOT indices.
        
        #==Subsampled pathway==
        self.useSubsampledBool = cfg[cfg.USE_SUBSAMPLED] if cfg[cfg.USE_SUBSAMPLED] is not None else False
        if not self.useSubsampledBool :
            self.numFMsPerLayerSubsampled = []
            self.kernDimPerLayerSubsampled = []
            self.receptiveFieldSubsampled = []
            self.subsampleFactor = []
            residConnAtLayersSubsampled = []
            lowerRankLayersSubsampled = []
            
        else :
            self.numFMsPerLayerSubsampled = cfg[cfg.N_FMS_SUBS] if cfg[cfg.N_FMS_SUBS] is not None else self.numFMsPerLayerNormal
            self.numFMsPerLayerSubsampled = self.changeDatastructureToListOfListsForSecondaryPathwaysIfNeeded(self.numFMsPerLayerSubsampled)
            # check that all subsampled pathways have the same number of layers. Limitation in the code currently, because I use kernDimSubsampled for all of them.
            if not self.checkThatSublistsHaveSameLength(self.numFMsPerLayerSubsampled):
                self.errReqSameNumOfLayersPerSubPathway()

            numOfLayersInEachSubPath = len(self.numFMsPerLayerSubsampled[0])
            if cfg[cfg.KERN_DIM_SUBS] == None and numOfLayersInEachSubPath == len(self.numFMsPerLayerNormal) :
                self.kernDimPerLayerSubsampled = self.kernDimPerLayerNormal
                self.receptiveFieldSubsampled = self.receptiveFieldNormal
            elif cfg[cfg.KERN_DIM_SUBS] == None and numOfLayersInEachSubPath != len(self.numFMsPerLayerNormal) : #user specified subsampled layers.
                self.errorRequireKernelDimensionsSubsampled(self.kernDimPerLayerNormal, cfg[cfg.N_FMS_SUBS])
            # kernDimSubsampled was specified. Now it's going to be tricky to make sure everything alright.
            elif not checkKernDimPerLayerCorrect3dAndNumLayers(cfg[cfg.KERN_DIM_SUBS], numOfLayersInEachSubPath) :
                self.errReqKernDimNormalCorr()
            else : #kernel dimensions specified and are correct (3d, same number of layers as subsampled specified). Need to check the two receptive fields and make sure they are correct.
                self.kernDimPerLayerSubsampled = cfg[cfg.KERN_DIM_SUBS]
                self.receptiveFieldSubsampled = calcRecFieldFromKernDimListPerLayerWhenStrides1(self.kernDimPerLayerSubsampled)
                if self.receptiveFieldNormal != self.receptiveFieldSubsampled :
                    self.errorReceptiveFieldsOfNormalAndSubsampledDifferent(self.receptiveFieldNormal, self.receptiveFieldSubsampled)
                #Everything alright, finally. Proceed safely...
            self.subsampleFactor = cfg[cfg.SUBS_FACTOR] if cfg[cfg.SUBS_FACTOR] is not None else [3,3,3]
            self.subsampleFactor = self.changeDatastructureToListOfListsForSecondaryPathwaysIfNeeded(self.subsampleFactor)
            for secondaryPathway_i in range(len(self.subsampleFactor)) : #It should now be a list of lists, one sublist per secondary pathway. This is what is currently defining how many pathways to use.
                if len(self.subsampleFactor[secondaryPathway_i]) != 3 :
                    self.errorSubFactor3d()
                if not checkSubsampleFactorEven(self.subsampleFactor[secondaryPathway_i]) :
                    self.warnSubFactorOdd()
            #---For multiple lower-scale pathways, via the numFMsPerLayerSubsampled and subsampleFactor config ----
            numOfSubsPaths = max(len(self.numFMsPerLayerSubsampled), len(self.subsampleFactor))
            # Default behaviour:
            # If less sublists in numFMsPerLayerSubsampled were given than numOfSubsPaths, add more sublists of numFMsPerLayerSubsampled, for the extra subpaths.
            for _ in range( numOfSubsPaths - len(self.numFMsPerLayerSubsampled) ) :
                numFmsForLayersOfLastSubPath = self.numFMsPerLayerSubsampled[-1]
                self.numFMsPerLayerSubsampled.append( [ max(1, int(numFmsInLayer_i)) for numFmsInLayer_i in numFmsForLayersOfLastSubPath ] )
            # If less sublists in subsampleFactor were given than numOfSubsPaths, add more sublists of subsampleFactors, for the extra subpaths.
            for _ in range( numOfSubsPaths - len(self.subsampleFactor) ) :
                self.subsampleFactor.append( [ subFactorInDim_i + 2 for subFactorInDim_i in self.subsampleFactor[-1] ] ) # Adds one more sublist, eg [5,5,5], which is the last subFactor, increased by +2 in all rcz dimensions.
            
            # Residuals and lower ranks.
            residConnAtLayersSubsampled = cfg[cfg.RESID_CONN_LAYERS_SUBS] if cfg[cfg.RESID_CONN_LAYERS_SUBS] is not None else residConnAtLayersNormal
            lowerRankLayersSubsampled = cfg[cfg.LOWER_RANK_LAYERS_SUBS] if cfg[cfg.LOWER_RANK_LAYERS_SUBS] is not None else lowerRankLayersNormal
            
        #==FC Layers==
        self.numFMsInExtraFcs = cfg[cfg.N_FMS_FC] if cfg[cfg.N_FMS_FC] is not None else []
        self.kernelDimensionsFirstFcLayer = cfg[cfg.KERN_DIM_1ST_FC] if cfg[cfg.KERN_DIM_1ST_FC] is not None else [1,1,1]
        assert len(self.kernelDimensionsFirstFcLayer) == 3 and (False not in [ dim > 0 for dim in self.kernelDimensionsFirstFcLayer] )
        residConnAtLayersFc = cfg[cfg.RESID_CONN_LAYERS_FC] if cfg[cfg.RESID_CONN_LAYERS_FC] is not None else []
                                        
        #==Size of Image Segments ==
        self.segmDimNormalTrain = cfg[cfg.SEG_DIM_TRAIN] if cfg[cfg.SEG_DIM_TRAIN] is not None else self.errReqSegmDimTrain()
        self.segmDimNormalVal = cfg[cfg.SEG_DIM_VAL] if cfg[cfg.SEG_DIM_VAL] is not None else self.receptiveFieldNormal
        self.segmDimNormalInfer = cfg[cfg.SEG_DIM_INFER] if cfg[cfg.SEG_DIM_INFER] is not None else self.segmDimNormalTrain
        for (tr0_val1_inf2, segmentDimensions) in [ (0,self.segmDimNormalTrain), (1,self.segmDimNormalVal), (2,self.segmDimNormalInfer) ] :
            if not checkRecFieldVsSegmSize(self.receptiveFieldNormal, segmentDimensions) :
                self.errorSegmDimensionsSmallerThanReceptiveF(self.receptiveFieldNormal, segmentDimensions, tr0_val1_inf2)
                
        #=== Batch Sizes ===
        self.batchSizeTrain = cfg[cfg.BATCH_SIZE_TR] if cfg[cfg.BATCH_SIZE_TR] is not None else self.errReqBatchSizeTr()
        self.batchSizeVal = cfg[cfg.BATCH_SIZE_VAL] if cfg[cfg.BATCH_SIZE_VAL] is not None else self.batchSizeTrain
        self.batchSizeInfer = cfg[cfg.BATCH_SIZE_INFER] if cfg[cfg.BATCH_SIZE_INFER] is not None else self.batchSizeTrain
        
        #=== Dropout rates ===
        self.dropNormal = cfg[cfg.DROP_NORM] if cfg[cfg.DROP_NORM] is not None else []
        self.dropSubsampled = cfg[cfg.DROP_SUBS] if cfg[cfg.DROP_SUBS] is not None else []
        self.dropFc = cfg[cfg.DROP_FC] if cfg[cfg.DROP_FC] is not None else self.defaultDropFcList(self.numFMsInExtraFcs) #default = [0.0, 0.5, ..., 0.5]
        self.dropoutRatesForAllPathways = [self.dropNormal, self.dropSubsampled, self.dropFc, []]
        
        #== Weight Initialization==
        self.convWInitMethod = cfg[cfg.CONV_W_INIT] if cfg[cfg.CONV_W_INIT] is not None else ["fanIn", 2]
        if not self.convWInitMethod[0] in ["normal", "fanIn"]:
            self.errorReqInitializationMethod()
        #== Activation Function ==
        self.activationFunc = cfg[cfg.ACTIV_FUNC] if cfg[cfg.ACTIV_FUNC] is not None else "prelu"
        if not self.activationFunc in ["linear", "relu", "prelu", "elu", "selu"]:
            self.errorReqActivFunction()
            
        #==BATCH NORMALIZATION==
        self.applyBnToInputOfPathways = [False, False, True] # Per pathway type. The 3rd entry, for FC, should always be True.
        self.bnRollAverOverThatManyBatches = cfg[cfg.BN_ROLL_AV_BATCHES] if cfg[cfg.BN_ROLL_AV_BATCHES] is not None else 60
        
        #==============CALCULATED=====================
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
        self.numberOfInputChannelsSubsampled = self.numberOfInputChannelsNormal
        
        #MultiscaleConnections:
        self.convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes = [ [], [] ] #a sublist for each pathway. Starts from 0 index. Give a sublist, even empty for no connections.
        #... It's ok if I dont have a 2nd path but still give a 2nd sublist, it's controlled by nkernsSubsampled.
        
        #-------POOLING---------- (not fully supported currently)
        #One entry per pathway-type. leave [] if the pathway does not exist or there is no mp there AT ALL.
        #Inside each entry, put a list FOR EACH LAYER. It should be [] for the layer if no mp there. But FOR EACH LAYER.
        #MP is applied >>AT THE INPUT of the layer<<. To use mp to a layer, put a list of [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode] which give the dimensions of the mp window, the stride, how many times to mirror the last slot at each dimension for padding (give 0 for none), the mode (usually 'max' pool). Eg [[2,2,2],[1,1,1]] or [[2,2,2],[2,2,2]] usually.
        #If a pathway is not used (eg subsampled), put an empty list in the first dimension entry. 
        mpParamsNorm = [ [] for layeri in range(len(self.numFMsPerLayerNormal)) ] #[[[2,2,2], [1,1,1], [1,1,1], 'MAX'], [],[],[],[],[],[], []], #first pathway
        mpParamsSubs = [ [] for layeri in range(len(self.numFMsPerLayerSubsampled[0])) ] if self.useSubsampledBool else [] # CAREFUL about the [0]. Only here till this structure is made different per pathway and not pathwayType.
        mpParamsFc = [ [] for layeri in range(len(self.numFMsInExtraFcs) + 1) ] #FC. This should NEVER be used for segmentation. Possible for classification though.
        self.maxPoolingParamsStructure = [ mpParamsNorm, mpParamsSubs, mpParamsFc]
        
        self.softmaxTemperature = 1.0 #Higher temperatures make the probabilities LESS distinctable. Actions have more similar probabilities. 
        
    
    def print_params(self) :
        logPrint = self.log.print3
        logPrint("=============================================================")
        logPrint("========== PARAMETERS FOR MAKING THE ARCHITECTURE ===========")
        logPrint("=============================================================")
        logPrint("CNN model's name = " + str(self.cnnModelName))
        
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
        logPrint("Number of subsampled pathways that will be built = " + str(len(self.subsampleFactor)) )
        logPrint("Number of Layers (per sub-pathway) = " + str([ len(numFmsPerLayerForSubPath) for numFmsPerLayerForSubPath in self.numFMsPerLayerSubsampled ]) )
        logPrint("Number of Feature Maps per layer (per sub-pathway) = " + str(self.numFMsPerLayerSubsampled))
        logPrint("Kernel Dimensions per layer = " + str(self.kernDimPerLayerSubsampled))
        logPrint("Receptive Field = " + str(self.receptiveFieldSubsampled))
        logPrint("Subsampling Factor per dimension (per sub-pathway) = " + str(self.subsampleFactor))
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
        
        logPrint("~~Batch Sizes~~")
        logPrint("Batch Size for Training = " + str(self.batchSizeTrain))
        logPrint("Batch Size for Validation = " + str(self.batchSizeVal))
        logPrint("Batch Size for Testing = " + str(self.batchSizeInfer))
        
        logPrint("~~Dropout Rates~~")
        logPrint("Drop.R. for each layer in Normal Pathway = " + str(self.dropoutRatesForAllPathways[0]))
        logPrint("Drop.R. for each layer in Subsampled Pathway = " + str(self.dropoutRatesForAllPathways[1]))
        logPrint("Drop.R. for each layer in FC Pathway (additional FC layers + Classific.Layer at end) = " + str(self.dropoutRatesForAllPathways[2]))
        
        logPrint("~~Weight Initialization~~")
        logPrint("Initialization method and params for the conv kernel weights = " + str(self.convWInitMethod))
        
        logPrint("~~Activation Function~~")
        logPrint("Activation function to use = " + str(self.activationFunc))
        
        logPrint("~~Batch Normalization~~")
        logPrint("Apply BN straight on pathways' inputs (eg straight on segments) = " + str(self.applyBnToInputOfPathways))
        logPrint("Batch Normalization uses a rolling average for inference, over this many batches = " + str(self.bnRollAverOverThatManyBatches))
        
        logPrint("========== Done with printing session's parameters ==========")
        logPrint("=============================================================")
        
    def get_args_for_arch(self) :
        
        args = [
                        self.log,
                        self.cnnModelName,
                        #=== Model Parameters ===
                        self.numberClasses,
                        self.numberOfInputChannelsNormal,
                        self.numberOfInputChannelsSubsampled,
                        #=== Normal Pathway ===
                        self.numFMsPerLayerNormal, #ONLY for the convolutional layers, NOT the final convFCSoftmaxLayer!
                        self.kernDimPerLayerNormal,
                        #=== Subsampled Pathway ===
                        self.numFMsPerLayerSubsampled,
                        self.kernDimPerLayerSubsampled,
                        self.subsampleFactor,
                        
                        #=== FC Layers ===
                        self.numFMsInExtraFcs,
                        self.kernelDimensionsFirstFcLayer,
                        self.softmaxTemperature,
                        
                        #=== Other Architectural params ===
                        self.activationFunc,
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
                        
                        #=== Batch Sizes ===
                        self.batchSizeTrain,
                        self.batchSizeVal,
                        self.batchSizeInfer,
                        
                        #=== Others ====
                        #Dropout
                        self.dropoutRatesForAllPathways,
                        #Initialization
                        self.convWInitMethod,
                        #Batch Normalization
                        self.applyBnToInputOfPathways,
                        self.bnRollAverOverThatManyBatches

                        ]
        
        return args

