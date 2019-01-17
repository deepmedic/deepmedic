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


class TestConfig(Config):
    
    #Optional but highly suggested.
    SESSION_NAME = "sessionName"
    #[REQUIRED]
    FOLDER_OUTP = "folderForOutput" #MUST BE GIVEN
    SAVED_MODEL = "cnnModelFilePath" #MUST BE GIVEN
    CHANNELS = "channels" #MUST BE GIVEN
    
    NAMES_FOR_PRED_PER_CASE = "namesForPredictionsPerCase"
    
    
    #[OPTIONALS]
    PAD_INPUT = "padInputImagesBool"
    
    ROI_MASKS = "roiMasks"
    
    GT_LABELS = "gtLabels"
    
    SAVE_SEGM = "saveSegmentation" # Default True
    SAVE_PROBMAPS_PER_CLASS = "saveProbMapsForEachClass" # Default True
    SUFFIX_SEGM_PROB = "suffixForSegmAndProbsDict"
    
    #optionals, cause default is False.
    SAVE_INDIV_FMS = "saveIndividualFms"
    SAVE_4DIM_FMS = "saveAllFmsIn4DimImage"
    
    INDICES_OF_FMS_TO_SAVE_NORMAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathway"
    INDICES_OF_FMS_TO_SAVE_SUBSAMPLED = "minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathway"
    INDICES_OF_FMS_TO_SAVE_FC = "minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathway"
    

    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)

    #If certain config args are given in command line, completely override the corresponding ones in the config files.
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        if args.saved_model:
            abs_path_model_cmd_line = getAbsPathEvenIfRelativeIsGiven(args.saved_model, os.getcwd())

            if self.get( self.SAVED_MODEL ) is not None:
                log.print3("WARN: A model to load was specified both in the command line and in the test-config file!\n"+\
                            "\t The input by the command line will be used: " + str(abs_path_model_cmd_line) )
            
            self._configStruct[ self.SAVED_MODEL ] = abs_path_model_cmd_line
    
