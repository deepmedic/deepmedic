# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

from deepmedic.frontEnd.configParsing.utils import *
from deepmedic.frontEnd.configParsing.config import Config, ConfigData, ConfigElem


class TestConfig(Config):
    config_data = ConfigData()

    config_data.set_curr_section('session', "Session Parameters")

    SESSION_NAME = \
        config_data.add_elem("sessionName", elem_type='String', description="Session Name",
                             info="[Optional but highly suggested] The name will be used for naming "
                                  "folders to save the results in.",
                             default="testSession")  # Optional but highly recommended.
    FOLDER_OUTP = \
        config_data.add_elem("folderForOutput", elem_type='Folder', required=True,
                             description="Output Folder",
                             info="The main folder that the output will be placed.")
    SAVED_MODEL = \
        config_data.add_elem("cnnModelFilePath", elem_type='File',
                             description="Saved Model Checkpoint",
                             info="Path to a saved model, to load parameters from in the beginning of the "
                                  "session. If one is also specified using the GUI/command line, the latter "
                                  "will be used.")

    config_data.set_curr_section('input', "Input")

    CHANNELS = \
        config_data.add_elem("channels", required=True, description="Input Channels", elem_type='Files',
                             info="A list that should contain as many entries as the channels of the input "
                                  "image (eg multi-modal MRI). The entries should be paths to files. "
                                  "Those files should be listing the paths to the corresponding channels "
                                  "for each test-case. (see example files).")
    NAMES_FOR_PRED_PER_CASE = \
        config_data.add_elem("namesForPredictionsPerCase", elem_type='File', required=True,
                             description="Names of Predictions per Case",
                             info="The path to a file, which should list names to give to "
                                  "the results for each testing case. (see example file).")
    ROI_MASKS = \
        config_data.add_elem("roiMasks",  elem_type='File', description="RoI Masks",
                             info="The path to a file, which should list paths to the Region-Of-Interest "
                                  "masks for each testing case.\n"
                                  "If ROI masks are provided, inference will only be performed within them "
                                  "(faster). If not specified, inference will be performed on whole volume.")
    GT_LABELS = \
        config_data.add_elem("gtLabels",  elem_type='File', description="Ground Truth Labels",
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

    RUN_INP_CHECKS = \
        config_data.add_elem("run_input_checks", elem_type='Bool', description="Run Input Checks",
                             info="Checks for format correctness of loaded input images. Can slow down the process.",
                             default=True)

    PAD_INPUT = \
        config_data.add_elem("padInputImagesBool", elem_type='Bool', description="Pad Input Images",
                             info="Pad images to fully convolve.", default=True)

    NORM_VERB_LVL = \
        config_data.add_elem("norm_verbosity_lvl", widget_type='combobox', options=[0, 1, 2],
                             description='Verbosity Level',
                             info="Verbosity-level for logging info on intensity-normalization. "
                                  "0: Nothing (default), 1: Per-subject, 2: Per-channel")
    NORM_ZSCORE_PRMS = \
        config_data.add_elem("norm_zscore_prms", widget_type='multiple',
                             description="Z-Score Normalisation",
                             options={
                                 'apply_to_all_channels':
                                     ConfigElem('apply_to_all_channels', description="Apply to All Channels",
                                                elem_type='Bool',
                                                info="True/False. Whether to do z-score normalization to ALL channels.",
                                                default=False),
                                 'apply_per_channel':
                                     ConfigElem('apply_per_channel', description='Apply per Channel',
                                                info="None, or a List with one boolean per channel. "
                                                     "Whether to normalize specific channel.\n"
                                                     "NOTE: If apply_to_all_channels is True, "
                                                     "apply_per_channel MUST be None."),
                                 'cutoff_percents':
                                     ConfigElem('cutoff_percents', description="Cutoff Percentiles",
                                                info="Cutoff at percentiles [float_low, float_high], "
                                                     "values in [0.0 - 100.0])"),
                                 'cutoff_times_std':
                                     ConfigElem('cutoff_times_std', description="Cutoff Standard Deviation",
                                                info="Cutoff intensities below/above [float_below, float_above] "
                                                     "times std from the mean."),
                                 'cutoff_below_mean':
                                     ConfigElem('cutoff_below_mean', description="Cutoff Below Mean", elem_type='Bool',
                                                info="True/False. Cutoff intensities below image mean. "
                                                     "Useful to exclude air in brain MRI.")
                             })

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
