# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

from deepmedic.frontEnd.configParsing.utils import abs_from_rel_path
from deepmedic.frontEnd.configParsing.config import Config


class TestConfig(Config):
    
    #Optional but highly suggested.
    SESSION_NAME = "sessionName"
    #[REQUIRED]
    FOLDER_OUTP = "folderForOutput"
    SAVED_MODEL = "cnnModelFilePath"
    DATAFRAME = "dataframe"
    CHANNELS = "channels"
    FNAMES_PREDS = "namesForPredictionsPerCase"
    
    #[OPTIONALS]
    ROIS = "roiMasks"
    GT_LBLS = "gtLabels"
    
    SAVE_SEGM = "saveSegmentation" # Default True
    SAVE_PROBMAPS_PER_CLASS = "saveProbMapsForEachClass" # Default True
    SUFFIX_SEGM_PROB = "suffixForSegmAndProbsDict"
    
    BATCHSIZE = "batchsize"
    
    # optionals, cause default is False.
    SAVE_INDIV_FMS = "saveIndividualFms"
    
    INDICES_OF_FMS_TO_SAVE_NORMAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathway"
    INDICES_OF_FMS_TO_SAVE_SUBSAMPLED = "minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathway"
    INDICES_OF_FMS_TO_SAVE_FC = "minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathway"
    
    # ========= GENERICS =========
    # ~~~~ Data compabitiliby checks ~~~
    RUN_INP_CHECKS = "run_input_checks"
    # ~~~~~ Preprocessing ~~~~~~~~
    PAD_INPUT = "padInputImagesBool"
    NORM_VERB_LVL = "norm_verbosity_lvl"
    NORM_ZSCORE_PRMS = "norm_zscore_prms"
    

    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)

    #If certain config args are given in command line, completely override the corresponding ones in the config files.
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        if args.saved_model:
            abs_path_model_cmd_line = abs_from_rel_path(args.saved_model, os.getcwd())

            if self.get(self.SAVED_MODEL) is not None:
                log.print3("WARN: A model to load was specified both in command line and in the test-config file!\n" +
                           "\t The input by the command line will be used: " + str(abs_path_model_cmd_line))
            
            self._configStruct[self.SAVED_MODEL] = abs_path_model_cmd_line
