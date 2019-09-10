# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

from deepmedic.frontEnd.configParsing.utils import *
from deepmedic.frontEnd.configParsing.config import Config, ConfigData


class TestConfig(Config):
    config_data = ConfigData()

    config_data.set_curr_section('session', "Session Parameters")

    SESSION_NAME = \
        config_data.add_elem("sessionName", elem_type='String', description="Session Name",
                             info="[Optional but highly suggested] The name will be used for naming "
                                  "folders to save the results in.",
                             default="testSession")  # Optional but highly recommended.
    FOLDER_OUTP = \
        config_data.add_elem("folderForOutput", elem_type='String', required=True,
                             description="Output Folder",
                             info="The main folder that the output will be placed.")
    SAVED_MODEL = \
        config_data.add_elem("cnnModelFilePath", elem_type='String',
                             description="Saved Model Checkpoint",
                             info="Path to a saved model, to load parameters from in the beginning of the "
                                  "session. If one is also specified using the GUI/command line, the latter "
                                  "will be used.")

    config_data.set_curr_section('input', "Input")

    CHANNELS = \
        config_data.add_elem("channels", required=True, description="Input Channels",
                             info="A list that should contain as many entries as the channels of the input "
                                  "image (eg multi-modal MRI). The entries should be paths to files. "
                                  "Those files should be listing the paths to the corresponding channels "
                                  "for each test-case. (see example files).")
    NAMES_FOR_PRED_PER_CASE = \
        config_data.add_elem("namesForPredictionsPerCase", elem_type='String', required=True,
                             description="Names of Predictions per Case",
                             info="The path to a file, which should list names to give to "
                                  "the results for each testing case. (see example file).")
    ROI_MASKS = \
        config_data.add_elem("roiMasks", elem_type='String', description="RoI Masks",
                             info="The path to a file, which should list paths to the Region-Of-Interest "
                                  "masks for each testing case.\n"
                                  "If ROI masks are provided, inference will only be performed within them "
                                  "(faster). If not specified, inference will be performed on whole volume.")
    GT_LABELS = \
        config_data.add_elem("gtLabels", elem_type='String', description="Ground Truth Labels",
                             info="The path to a file which should list paths to the Ground Truth labels of "
                                  "each testing case. If provided, DSC metrics will be reported")
    BATCHSIZE = \
        config_data.add_elem("batchsize", description="Batch Size", info="Batch Size.", default=10)
    
    config_data.set_curr_section('pred', "Predictions")

    SAVE_SEGM = \
        config_data.add_elem("saveSegmentation", elem_type='Bool', default=True,
                             description="Save Segmentation",
                             info="Specify whether to save segmentation maps.")
    SAVE_PROBMAPS_PER_CLASS = \
        config_data.add_elem("saveProbMapsForEachClass",
                             description="Save per Class Probability Maps",
                             info="Provide a list with as many entries as there are classes. "
                                  "True/False to save/not the probability map for the corresponding class.\n"
                                  "(default: [True,True...for all classes])")
    SUFFIX_SEGM_PROB = \
        config_data.add_elem("suffixForSegmAndProbsDict",
                             description="Suffix Dictionary for Segmentation and Probability Maps",
                             default={"segm": "Segm", "prob": "ProbMapClass"},
                             advanced=True)

    config_data.set_curr_section('fms', "Feature Maps")

    SAVE_INDIV_FMS = \
        config_data.add_elem("saveIndividualFms", elem_type='Bool', default=True,
                             description="Save Individual Feature Maps",
                             info="Specify whether to save the feature maps in separate files.")
    SAVE_4DIM_FMS = \
        config_data.add_elem("saveAllFmsIn4DimImage", elem_type='Bool', default=False,
                             description="Save All Feature Maps in 4D Image",
                             info="Specify whether to save the feature maps combined in a 4D image.")

    INDICES_OF_FMS_TO_SAVE_NORMAL = \
        config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathway",
                             description="Indices of FMs to Save per Layer (Normal)",
                             info="A model may have too many feature maps, and some may "
                                  "not be needed. For this, we allow specifying which FMs to save.\n"
                                  "Provide, for each the normal pathway, a list with as"
                                  " many sublists as there are layers of the pathway.\n"
                                  "Each sublist (one for each layer) should have 2 "
                                  "numbers. These are the minimum (inclusive) and "
                                  "maximum (exclusive) indices of the Feature Maps to "
                                  "save from the layer. See FC for example.",
                             default=[])
    INDICES_OF_FMS_TO_SAVE_SUBSAMPLED = \
        config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathway",
                             description="Indices of FMs to Save per Layer (Subsampled)",
                             info="A model may have too many feature maps, and some "
                                  "may not be needed. For this, we allow specifying which FMs to save.\n"
                                  "Provide, for each the subsampled pathway, a list "
                                  "with as many sublists as there are pathway layers.\n"
                                  "Each sublist (one for each layer) should have 2 "
                                  "numbers. These are the minimum (inclusive) and "
                                  "maximum (exclusive) indices of the Feature Maps to "
                                  "save from the layer. See FC for example.",
                             default=[])
    INDICES_OF_FMS_TO_SAVE_FC = \
        config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathway",
                             description="Indicies of FMs to Save per Layer (FC)",
                             info="A model may have too many feature maps, and some may "
                                  "not be needed. For this, we allow specifying which FMs to save.\n"
                                  "Provide, for each the normal pathway, a list with as"
                                  " many sublists as there are layers of the pathway.\n"
                                  "Each sublist (one for each layer) should have 2 numbers. These are the"
                                  " minimum (inclusive) and maximum (exclusive) indices of the Feature Maps to "
                                  "save from the layer.\n"
                                  "Example: [[],[0,150],[]] saves the Feature Maps from "
                                  "index 0 (first FM) to 150 of the last hidden FC layer, "
                                  "before the classification layer.",
                             default=[])

    config_data.set_curr_section('generic', "Generic")

    PAD_INPUT = \
        config_data.add_elem("padInputImagesBool", elem_type='Bool', description="Pad Input Images",
                             info="Pad images to fully convolve.", default=True)

    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)

    # If certain config args are given in command line, completely override the corresponding ones in the config files.
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        if args.saved_model:
            abs_path_model_cmd_line = getAbsPathEvenIfRelativeIsGiven(args.saved_model, os.getcwd())

            if self.get( self.SAVED_MODEL ) is not None:
                log.print3("WARN: A model to load was specified both in the command line and in the test-config file!\n"+\
                            "\t The input by the command line will be used: " + str(abs_path_model_cmd_line) )
            
            self._configStruct[ self.SAVED_MODEL ] = abs_path_model_cmd_line
