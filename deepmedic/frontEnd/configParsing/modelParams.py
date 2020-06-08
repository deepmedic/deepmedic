# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

from deepmedic.neuralnet.utils import calc_rec_field_of_path_assuming_strides_1, check_rec_field_vs_inp_dims, \
    check_kern_dims_per_l_correct_3d_and_n_layers, subsample_factor_is_even


class ModelParameters(object):
        
    # To be called from outside too.
    @staticmethod
    def get_default_model_name():
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
    def errorSegmDimensionsSmallerThanReceptiveF(rec_field_norm, segmentDimensions, train_val_test) :
        print("ERROR: The segment-size (input) should be at least as big as the receptive field of the model! The network was made with a receptive field of dimensions: ", rec_field_norm, ". But in the case of: [", train_val_test, "] the dimensions of the input segment were specified smaller: ", segmentDimensions, ". Please fix this by adjusting number of layer and kernel dimensions! Exiting!"); exit(1)
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
    def errorRequireKernelDimensionsSubsampled(n_fms_per_l_norm, n_fms_per_l_subs) :
        print("ERROR: It was requested to use the 2-scale architecture, with a subsampled pathway. Because of limitations to the developed system, the two pathways must have the save size of receptive field. By default, if \"useSubsampledPathway\" = True, and the parameters \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" are not specified, the second pathway will be constructed symmetrical to the first. However, in this case, \"numberFMsPerLayerSubsampled\" was specified. It was found to have ", len(n_fms_per_l_subs)," entries, which specified this amount of layers in the subsampled pathway. This is different than the number of layers in the Normal pathway, specified to be: ", len(n_fms_per_l_norm),". In this case, we require you to also provide the parameter \"numberFMsPerLayerSubsampled\", specifying kernel dimensions in the subsampled pathway, in a fashion that results in same size of receptive field as the normal pathway.")
        ArchitectureParameters.warnForSameReceptiveField()
        print("Exiting!"); exit(1)
    @staticmethod
    def errorRequireKernelDimensionsPerLayerSubsampledCorrect() :
        print("ERROR: The parameter \"kernelDimPerLayerSubsampled\" was not provided, or provided incorrectly. It should be provided in the format: kernelDimPerLayerSubsampled = [ [dim1-of-kernels-in-layer-1, dim2-of-kernels-in-layer-1, dim3-of-kernels-in-layer-1], ..., [dim1-of-kernels-in-layer-N, dim2-of-kernels-in-layer-N, dim3-of-kernels-in-layer-N] ]. It is a list of sublists. One sublist should be provided per layer of the SUBSAMPLED pathway. Thus it should have as many entries as the entries in parameter \"numberFMsPerLayerSubsampled\". (WARN: if the latter is not provided, it is by default taken equal to what specified for \"numberFMsPerLayerNormal\", in order to make the pathways symmetrical). Each sublist should contain 3 integer ODD numbers greater than zero, which should specify the dimensions of the 3-dimensional kernels. For instace: kernelDimPerLayerNormal = [[5,5,5],[3,3,3]] for a pathway with 2 layers, the first of which has 5x5x5 kernels and the second 3x3x3 kernels. Please fix and retry. (WARN: The kernel dimensions should be ODD-NUMBERS. System was not thoroughly tested for kernels of even dimensions!)")
        ArchitectureParameters.warnForSameReceptiveField()
        print("Exiting!"); exit(1)
    errReqKernDimNormalCorr = errorRequireKernelDimensionsPerLayerSubsampledCorrect
    @staticmethod
    def errorReceptiveFieldsOfNormalAndSubsampledDifferent(kern_dims_per_l_norm, rec_field_subs) :
        print("ERROR: The receptive field of the normal pathway was calculated = ", len(kern_dims_per_l_norm), " while the receptive field of the subsampled pathway was calculated=", len(rec_field_subs), ". Because of limitations to the developed system, the two pathways must have the save size of receptive field. Please provide a combination of \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" that gives the same size of field as the normal pathway. If unsure of how to proceed, please ommit specifying \"numberFMsPerLayerSubsampled\" and \"kernelDimPerLayerSubsampled\" in the config file, and the second subsampled pathway will be automatically created to mirror the normal. Else, if you want to just specify the number of Feature Maps in the subsampled, provide \"numberFMsPerLayerSubsampled\" = [num-FMs-layer1, ..., num-FMs-layerN], with N the same number as the normal pathway, and we will then use the same kernel-sizes as the normal pathway. Exiting!"); exit(1)
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
        print("ERROR: The parameter \"subsample_factors\" must have 3 entries, one for each of the 3 dimensions. Please provide it in the format: subsample_factors = [subFactor-dim1, subFactor-dim2, subFactor-dim3]. Each of the entries should be an integer, eg [3,3,3].")
        ArchitectureParameters.warnSubFactorOdd()
        print("Exiting!"); exit(1)

    @staticmethod
    def errorRequireSegmentDimensionsTrain() :
        print("ERROR: The parameter \"segmentsDimTrain\" was is required but not given. It specifies the size of the 3D segment that is given as input to the network. It should be at least as large as the receptive field of the network in each dimension. Please specify it in the format: segmentsDimTrain = [dim-1, dim-2, dim-3]. Exiting!"); exit(1)
    errReqSegmDimTrain = errorRequireSegmentDimensionsTrain

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
        print("WARN: The system was only thoroughly tested for ODD subsampling factor! (Eg subsample_factors = [3,3,3]).")

    # OTHERS
    @staticmethod
    def _to_list_of_lists_if_needed(structure_from_cfg):
        # structure_from_cfg: whatever given in the config by the user (except None).
        if not isinstance(structure_from_cfg, list):
            print("ERROR: variable \"", structure_from_cfg, "\" given in modelConfig.cfg should be either a list of "
                  "integers, or a list of lists of integers, in case multiple lower-scale pathways are wanted. "
                  "Please correct it. Exiting.")
            exit(1)
        all_elements_are_lists = True
        no_element_is_list = True
        for element in structure_from_cfg:
            if isinstance(element, list):
                no_element_is_list = False
            else :
                all_elements_are_lists = False
        if not (all_elements_are_lists or no_element_is_list):  # some are lists and some are not
            print("ERROR: variable \"", structure_from_cfg, "\" given in modelConfig.cfg should be either a list of "
                  "integers, or a list of lists of integers, in case multiple lower-scale pathways are wanted. "
                  "Please correct it. Exiting.")
            exit(1)
        elif no_element_is_list:
            # Seems ok, but structure is not a list of lists, but probably the old type, eg [3,3,3].
            # Change to list of lists.
            return [structure_from_cfg]
        else:
            return structure_from_cfg

    def _check_sublists_have_same_length(self, list_of_lists):
        if len(list_of_lists) == 0:
            return True
        len_of_first = len(list_of_lists[0])
        for subList_i in range( len(list_of_lists)):
            if len_of_first != len(list_of_lists[subList_i]):
                return False
        return True

    def _check_no_res_conn_at_1st_layer(self, res_conn_at_layers_norm, res_conn_at_layers_subs, res_conn_at_layers_fc):
        if 1 in res_conn_at_layers_norm:
            self.errorResLayer1("Normal")
        if 1 in res_conn_at_layers_subs:
            self.errorResLayer1("Subsampled")
        if 1 in res_conn_at_layers_fc:
            self.errorResLayer1("Fully Connected")

    def __init__(self, log, cfg):
        self.log = log
        self.model_name = cfg[cfg.MODEL_NAME] if cfg[cfg.MODEL_NAME] is not None else self.get_default_model_name()
        
        # =========== MODEL PARAMETERS ==========
        self.n_classes = cfg[cfg.NUM_CLASSES] if cfg[cfg.NUM_CLASSES] is not None else self.errReqNumClasses()
        self.n_in_chans = cfg[cfg.NUM_INPUT_CHANS] if cfg[cfg.NUM_INPUT_CHANS] is not None else self.errReqNumChannels()
        assert self.n_in_chans > 0, "Number of input channels should be greater than 0."
        # === Normal pathway ===
        self.n_fms_per_l_norm = cfg[cfg.N_FMS_NORM] if cfg[cfg.N_FMS_NORM] is not None and len(cfg[cfg.N_FMS_NORM]) > 0\
            else self.errReqFMsNormal()
        n_layers_norm = len(self.n_fms_per_l_norm)
        self.kern_dims_per_l_norm = cfg[cfg.KERN_DIM_NORM] \
            if check_kern_dims_per_l_correct_3d_and_n_layers(cfg[cfg.KERN_DIM_NORM], n_layers_norm) \
            else self.errReqKernDimNormal()
        # The below rec_field is ONLY for checking correctness of the passed parameters. TODO: Remove
        rec_field_norm = calc_rec_field_of_path_assuming_strides_1(self.kern_dims_per_l_norm)
        self.pad_mode_per_l_norm = cfg[cfg.PAD_MODE_NORM] if cfg[cfg.PAD_MODE_NORM] is not None \
            else ['VALID'] * n_layers_norm
        # The below are layer numbers, starting from 1 for 1st layer. NOT indices starting from 0.
        res_conn_at_layers_norm = cfg[cfg.RESID_CONN_LAYERS_NORM] if cfg[cfg.RESID_CONN_LAYERS_NORM] is not None else []
        lower_rank_layers_norm = cfg[cfg.LOWER_RANK_LAYERS_NORM] if cfg[cfg.LOWER_RANK_LAYERS_NORM] is not None else []
        
        # == Subsampled pathway ==
        self.use_subs_paths = cfg[cfg.USE_SUBSAMPLED] if cfg[cfg.USE_SUBSAMPLED] is not None else False
        if not self.use_subs_paths:
            self.n_fms_per_l_subs = []
            self.kern_dims_per_l_subs = []
            self.subsample_factors = []  # Per pathway, per dimension. E.g.: [[3,3,3], [5,5,5]]
            self.pad_mode_per_l_subs = []
            res_conn_at_layers_subs = []
            lower_rank_layers_subs = []
            rec_field_subs = []

        else:
            self.n_fms_per_l_subs = cfg[cfg.N_FMS_SUBS] if cfg[cfg.N_FMS_SUBS] is not None else self.n_fms_per_l_norm
            self.n_fms_per_l_subs = self._to_list_of_lists_if_needed(self.n_fms_per_l_subs)
            # Check that all subsampled pathways have the same number of layers.
            # Limitation in the code currently, because I use kern_dims_per_l_subs for all of them.
            if not self._check_sublists_have_same_length(self.n_fms_per_l_subs):
                self.errReqSameNumOfLayersPerSubPathway()

            n_layers_subs = len(self.n_fms_per_l_subs[0])
            if cfg[cfg.KERN_DIM_SUBS] is None and n_layers_subs == n_layers_norm:
                self.kern_dims_per_l_subs = self.kern_dims_per_l_norm
                rec_field_subs = rec_field_norm
            elif cfg[cfg.KERN_DIM_SUBS] is None and n_layers_subs != n_layers_norm:
                self.errorRequireKernelDimensionsSubsampled(self.kern_dims_per_l_norm, cfg[cfg.N_FMS_SUBS])
            # KERN_DIM_SUBS was specified. Now it's going to be tricky to make sure everything alright.
            elif not check_kern_dims_per_l_correct_3d_and_n_layers(cfg[cfg.KERN_DIM_SUBS], n_layers_subs):
                self.errReqKernDimNormalCorr()
            else:  # kernel dimensions specified and are correct. Check the two receptive fields and ensure correctness.
                self.kern_dims_per_l_subs = cfg[cfg.KERN_DIM_SUBS]
                rec_field_subs = calc_rec_field_of_path_assuming_strides_1(self.kern_dims_per_l_subs)
                if rec_field_norm != rec_field_subs:
                    self.errorReceptiveFieldsOfNormalAndSubsampledDifferent(rec_field_norm, rec_field_subs)
                # Everything alright, finally. Proceed safely...
            self.pad_mode_per_l_subs = cfg[cfg.PAD_MODE_SUBS] if cfg[cfg.PAD_MODE_SUBS] is not None \
                else ['VALID'] * n_layers_subs
            self.subsample_factors = cfg[cfg.SUBS_FACTOR] if cfg[cfg.SUBS_FACTOR] is not None else [3, 3, 3]
            self.subsample_factors = self._to_list_of_lists_if_needed(self.subsample_factors)
            # self.subsample_factors: Should now be a list of lists, one per subsmpled pathway. Defines number of paths.
            n_subs_paths = len(self.subsample_factors)
            for subs_path_i in range(n_subs_paths):
                if len(self.subsample_factors[subs_path_i]) != 3:
                    self.errorSubFactor3d()
                if not subsample_factor_is_even(self.subsample_factors[subs_path_i]):
                    self.warnSubFactorOdd()
            # Default behaviour:
            # If less sublists in n_fms_per_l_subs were given than n_subs_paths, add one for each subsampled pathway.
            for _ in range(n_subs_paths - len(self.n_fms_per_l_subs)):
                n_fms_per_l_in_prev_path = self.n_fms_per_l_subs[-1]
                self.n_fms_per_l_subs.append([max(1, int(n_fms_in_l_i)) for n_fms_in_l_i in n_fms_per_l_in_prev_path])

            # Residuals and lower ranks.
            res_conn_at_layers_subs = cfg[cfg.RESID_CONN_LAYERS_SUBS] if cfg[cfg.RESID_CONN_LAYERS_SUBS] is not None \
                else res_conn_at_layers_norm
            lower_rank_layers_subs = cfg[cfg.LOWER_RANK_LAYERS_SUBS] if cfg[cfg.LOWER_RANK_LAYERS_SUBS] is not None \
                else lower_rank_layers_norm
            
        #==FC Layers==
        self.numFMsInExtraFcs = cfg[cfg.N_FMS_FC] if cfg[cfg.N_FMS_FC] is not None else []
        self.kernelDimensionsFc = cfg[cfg.KERN_DIM_FC] if cfg[cfg.KERN_DIM_FC] is not None else [[1,1,1]]
        assert len(self.kernelDimensionsFc) == (len(self.numFMsInExtraFcs) + 1), 'Need one Kernel-Dimensions per layer of FC path, equal to length of number-of-FMs-in-FC +1 (for classif layer)'
        self.pad_mode_per_l_fc = cfg[cfg.PAD_MODE_FC] if cfg[cfg.PAD_MODE_FC] is not None else ['VALID']*(len(self.numFMsInExtraFcs)+1)
        res_conn_at_layers_fc = cfg[cfg.RESID_CONN_LAYERS_FC] if cfg[cfg.RESID_CONN_LAYERS_FC] is not None else []
                                        
        #==Size of Image Segments ==
        self._inp_dims_hr_path = {'train': None, 'val': None, 'test': None}
        self._inp_dims_hr_path['train'] = cfg[cfg.SEG_DIM_TRAIN] if cfg[cfg.SEG_DIM_TRAIN] is not None else self.errReqSegmDimTrain()
        self._inp_dims_hr_path['val'] = cfg[cfg.SEG_DIM_VAL] if cfg[cfg.SEG_DIM_VAL] is not None else rec_field_norm
        self._inp_dims_hr_path['test'] = cfg[cfg.SEG_DIM_INFER] if cfg[cfg.SEG_DIM_INFER] is not None else self._inp_dims_hr_path['train']
        self.segmDimNormalTrain = 1
        self.segmDimNormalVal = 2
        self.segmDimNormalInfer = 3
        for train_val_test in ['train', 'val', 'test']:
            if not check_rec_field_vs_inp_dims(rec_field_norm, self._inp_dims_hr_path[train_val_test]):
                self.errorSegmDimensionsSmallerThanReceptiveF(rec_field_norm, self._inp_dims_hr_path, train_val_test)

        
        #=== Dropout rates ===
        self.dropNormal = cfg[cfg.DROP_NORM] if cfg[cfg.DROP_NORM] is not None else []
        self.dropSubsampled = cfg[cfg.DROP_SUBS] if cfg[cfg.DROP_SUBS] is not None else []
        self.dropFc = cfg[cfg.DROP_FC] if cfg[cfg.DROP_FC] is not None else self.defaultDropFcList(self.numFMsInExtraFcs) #default = [0.0, 0.5, ..., 0.5]
        self.dropoutRatesForAllPathways = [self.dropNormal, self.dropSubsampled, self.dropFc]
        
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
        self._check_no_res_conn_at_1st_layer(res_conn_at_layers_norm, res_conn_at_layers_subs, res_conn_at_layers_fc)
        # The following variable passed to the system takes indices, ie number starts from 0. User specifies from 1.
        self.indicesOfLayersToConnectResidualsInOutput = [[layerNum - 1 for layerNum in res_conn_at_layers_norm],
                                                          [layerNum - 1 for layerNum in res_conn_at_layers_subs],
                                                          [layerNum - 1 for layerNum in res_conn_at_layers_fc],
                                                          []
                                                        ]
        
        self.indicesOfLowerRankLayersPerPathway = [[layerNum - 1 for layerNum in lower_rank_layers_norm],
                                                   [layerNum - 1 for layerNum in lower_rank_layers_subs],
                                                   [], #FC doesn't make sense to be lower rank. It's 1x1x1 anyway.
                                                   []
                                                   ]
        self.ranksOfLowerRankLayersForEachPathway = [[2 for layer_i in self.indicesOfLowerRankLayersPerPathway[0]],
                                                     [2 for layer_i in self.indicesOfLowerRankLayersPerPathway[1]],
                                                     [],
                                                     []
                                                     ]
        #============= HIDDENS ======================
        
        #-------POOLING---------- (not fully supported currently)
        #One entry per pathway-type. leave [] if the pathway does not exist or there is no mp there AT ALL.
        #Inside each entry, put a list FOR EACH LAYER. It should be [] for the layer if no mp there. But FOR EACH LAYER.
        #MP is applied >>AT THE INPUT of the layer<<. To use mp to a layer, put a list of [[dsr,dsc,dsz], [strr,strc,strz], [mirrorPad-r,-c,-z], mode] which give the dimensions of the mp window, the stride, how many times to mirror the last slot at each dimension for padding (give 0 for none), the mode (usually 'max' pool). Eg [[2,2,2],[1,1,1]] or [[2,2,2],[2,2,2]] usually.
        #If a pathway is not used (eg subsampled), put an empty list in the first dimension entry. 
        mpParamsNorm = [ [] for layeri in range(len(self.n_fms_per_l_norm)) ] #[[[2,2,2], [1,1,1], 'MIRROR', 'MAX'], [],[],[],[],[],[], []], #first pathway
        mpParamsSubs = [ [] for layeri in range(len(self.n_fms_per_l_subs[0])) ] if self.use_subs_paths else [] # CAREFUL about the [0]. Only here till this structure is made different per pathway and not pathwayType.
        mpParamsFc = [ [] for layeri in range(len(self.numFMsInExtraFcs) + 1) ] #FC. This should NEVER be used for segmentation. Possible for classification though.
        self.maxPoolingParamsStructure = [ mpParamsNorm, mpParamsSubs, mpParamsFc]
        
        self.softmax_tempertature = 1.0  # Higher temperatures make the probabilities LESS distinctable.
        
    
    def print_params(self) :
        logPrint = self.log.print3
        logPrint("=============================================================")
        logPrint("========== PARAMETERS FOR MAKING THE ARCHITECTURE ===========")
        logPrint("=============================================================")
        logPrint("CNN model's name = " + str(self.model_name))
        
        logPrint("~~~~~~~~~~~~~~~~~~Model parameters~~~~~~~~~~~~~~~~")
        logPrint("Number of Classes (including background) = " + str(self.n_classes))
        logPrint("~~Normal Pathway~~")
        logPrint("Number of Input Channels = " + str(self.n_in_chans))
        logPrint("Number of Layers = " + str(len(self.n_fms_per_l_norm)))
        logPrint("Number of Feature Maps per layer = " + str(self.n_fms_per_l_norm))
        logPrint("Kernel Dimensions per layer = " + str(self.kern_dims_per_l_norm))
        logPrint("Padding mode of convs per layer = " + str(self.pad_mode_per_l_norm))
        logPrint("Residual connections added at the output of layers (indices from 0) = " + str(self.indicesOfLayersToConnectResidualsInOutput[0]))
        logPrint("Layers that will be made of Lower Rank (indices from 0) = " + str(self.indicesOfLowerRankLayersPerPathway[0]))
        logPrint("Lower Rank layers will be made of rank = " + str(self.ranksOfLowerRankLayersForEachPathway[0]))
        # logPrint("Parameters for pooling in this pathway = " +  + str(self.maxPoolingParamsStructure[0]))
        
        logPrint("~~Subsampled Pathway~~")
        logPrint("Use subsampled Pathway = " + str(self.use_subs_paths))
        logPrint("Number of subsampled pathways that will be built = " + str(len(self.subsample_factors)) )
        logPrint("Number of Layers (per sub-pathway) = " + str([len(n_fms_subs_i) for n_fms_subs_i in self.n_fms_per_l_subs]))
        logPrint("Number of Feature Maps per layer (per sub-pathway) = " + str(self.n_fms_per_l_subs))
        logPrint("Kernel Dimensions per layer = " + str(self.kern_dims_per_l_subs))
        logPrint("Padding mode of convs per layer = " + str(self.pad_mode_per_l_subs))
        logPrint("Subsampling Factor per dimension (per sub-pathway) = " + str(self.subsample_factors))
        logPrint("Residual connections added at the output of layers (indices from 0) = " + str(self.indicesOfLayersToConnectResidualsInOutput[1]))
        logPrint("Layers that will be made of Lower Rank (indices from 0) = " + str(self.indicesOfLowerRankLayersPerPathway[1]))
        logPrint("Lower Rank layers will be made of rank = " + str(self.ranksOfLowerRankLayersForEachPathway[1]))
        #logPrint("Parameters for pooling before convolutions in this pathway = " +  + str(self.maxPoolingParamsStructure[1]))
        
        logPrint("~~Fully Connected Pathway~~")
        logPrint("Number of additional FC layers (Excluding the Classif. Layer) = " + str(len(self.numFMsInExtraFcs)))
        logPrint("Number of Feature Maps in the additional FC layers = " + str(self.numFMsInExtraFcs))
        logPrint("Padding mode of convs per layer = " + str(self.pad_mode_per_l_fc))
        logPrint("Residual connections added at the output of layers (indices from 0) = " + str(self.indicesOfLayersToConnectResidualsInOutput[2]))
        logPrint("Layers that will be made of Lower Rank (indices from 0) = " + str(self.indicesOfLowerRankLayersPerPathway[2]))
        #logPrint("Parameters for pooling before convolutions in this pathway = " +  + str(self.maxPoolingParamsStructure[2]))
        logPrint("Dimensions of Kernels in final FC path before classification = " + str(self.kernelDimensionsFc))
        
        logPrint("~~Size Of Image Segments~~")
        logPrint("Size of Segments for Training = " + str(self._inp_dims_hr_path['train']))
        logPrint("Size of Segments for Validation = " + str(self._inp_dims_hr_path['val']))
        logPrint("Size of Segments for Testing = " + str(self._inp_dims_hr_path['test']))
        
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
        
    def get_args_for_arch(self):
        
        args = [
                        self.log,
                        self.model_name,
                        #=== Model Parameters ===
                        self.n_classes,
                        self.n_in_chans,
                        #=== Normal Pathway ===
                        self.n_fms_per_l_norm, #ONLY for the convolutional layers, NOT the final convFCSoftmaxLayer!
                        self.kern_dims_per_l_norm,
                        self.pad_mode_per_l_norm,
                        #=== Subsampled Pathway ===
                        self.n_fms_per_l_subs,
                        self.kern_dims_per_l_subs,
                        self.pad_mode_per_l_subs,
                        self.subsample_factors,
                        
                        #=== FC Layers ===
                        self.numFMsInExtraFcs,
                        self.kernelDimensionsFc,
                        self.pad_mode_per_l_fc,
                        self.softmax_tempertature,
                        
                        #=== Other Architectural params ===
                        self.activationFunc,
                        #---Residual Connections----
                        self.indicesOfLayersToConnectResidualsInOutput,
                        #--Lower Rank Layer Per Pathway---
                        self.indicesOfLowerRankLayersPerPathway,
                        self.ranksOfLowerRankLayersForEachPathway,
                        #---Pooling---
                        self.maxPoolingParamsStructure,
                        
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


    def get_inp_dims_hr_path(self, train_val_test): # TODO: Move config in train/test cfg.
        #==Size of Image Segments ==
        assert train_val_test in ['train', 'val', 'test']
        return self._inp_dims_hr_path[train_val_test]
        
    def get_n_classes(self):
        return self.n_classes

    def get_model_name(self):
        return self.model_name
