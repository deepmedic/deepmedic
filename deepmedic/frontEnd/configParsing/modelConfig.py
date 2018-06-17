# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

from deepmedic.frontEnd.configParsing.utils import *
from deepmedic.frontEnd.configParsing.config import Config


class ModelConfig(Config):
    
    #Optional but highly suggested.
    MODEL_NAME = "modelName"
    #[REQUIRED] Output:
    FOLDER_OUTP = "folderForOutput" #MUST BE GIVEN
    
    #================ MODEL PARAMETERS =================
    NUM_CLASSES = "numberOfOutputClasses"
    NUM_INPUT_CHANS = "numberOfInputChannels"
    
    #===Normal pathway===
    N_FMS_NORM = "numberFMsPerLayerNormal"
    KERN_DIM_NORM = "kernelDimPerLayerNormal"
    RESID_CONN_LAYERS_NORM = "layersWithResidualConnNormal"
    LOWER_RANK_LAYERS_NORM = "lowerRankLayersNormal"
    
    #==Subsampled pathway==
    USE_SUBSAMPLED = "useSubsampledPathway"
    #The below should be mirroring the pathway, otherwise let them specify them but throw warning all around that receptive field should stay the same!
    N_FMS_SUBS = "numberFMsPerLayerSubsampled"
    KERN_DIM_SUBS = "kernelDimPerLayerSubsampled"
    SUBS_FACTOR = "subsampleFactor"
    RESID_CONN_LAYERS_SUBS = "layersWithResidualConnSubsampled"
    LOWER_RANK_LAYERS_SUBS = "lowerRankLayersSubsampled"
    
    #==Extra hidden FC Layers. Final Classification layer is not included in here.
    N_FMS_FC = "numberFMsPerLayerFC"
    KERN_DIM_1ST_FC = "kernelDimFor1stFcLayer"
    RESID_CONN_LAYERS_FC = "layersWithResidualConnFC"
    
    #Size of Image Segments
    SEG_DIM_TRAIN = "segmentsDimTrain"
    SEG_DIM_VAL = "segmentsDimVal"
    SEG_DIM_INFER = "segmentsDimInference"
    
    #==Batch Sizes===
    #Required.
    BATCH_SIZE_TR = "batchSizeTrain"
    BATCH_SIZE_VAL = "batchSizeVal"
    BATCH_SIZE_INFER = "batchSizeInfer"
    
    #Dropout Rates:
    DROP_NORM = "dropoutRatesNormal"
    DROP_SUBS = "dropoutRatesSubsampled"
    DROP_FC = "dropoutRatesFc"
    
    #Initialization method of the kernel weights.
    CONV_W_INIT = "convWeightsInit"
    #Activation Function for all convolutional layers:
    ACTIV_FUNC = "activationFunction"
    
    #Batch Normalization
    BN_ROLL_AV_BATCHES = "rollAverageForBNOverThatManyBatches"
    

    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)

    # Called from parent constructor
    def _check_for_deprecated_cfg(self):
        msg_part1 = "ERROR: Deprecated input to the config: ["
        msg_part2 = "]. Please update config and use the new corresponding variable "
        msg_part3 = "]. Exiting."
        if self.get("initializeClassic0orDelving1") is not None:
            print(msg_part1 + "initializeClassic0orDelving1" + msg_part2 + "convWeightsInit" + msg_part3); exit(1)
        if self.get("relu0orPrelu1") is not None:
            print(msg_part1 + "relu0orPrelu1" + msg_part2 + "activationFunction" + msg_part3); exit(1)
    
    

