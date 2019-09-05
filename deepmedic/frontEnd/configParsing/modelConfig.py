# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

from deepmedic.frontEnd.configParsing.config import Config, ConfigData, ConfigElem


class ModelConfig(Config):
    # define sections
    config_data = ConfigData()

    config_data.set_curr_section('model', "Model Parameters")
    # Optional but highly suggested.
    MODEL_NAME = config_data.add_elem("modelName", elem_type='String',
                                      description='Model Name', default="cnnModel",
                                      info="[Optional but highly recommended] The name will be used in the filenames "
                                           "when saving the model.")
    # [REQUIRED] Output:
    FOLDER_OUTP = config_data.add_elem("folderForOutput", elem_type='String', required=True,
                                       description='Output Folder',
                                       info="The main folder that the output will be placed.")
    
    # ================ MODEL PARAMETERS =================
    NUM_CLASSES = config_data.add_elem("numberOfOutputClasses", required=True,
                                       description='Number of Classes',
                                       info="The number of classes in the task (including background).")
    NUM_INPUT_CHANS = config_data.add_elem("numberOfInputChannels", required=True,
                                           description='Number of Channels',
                                           info="The number of input channels (e.g. number of MRI modalities).")

    # ===Normal pathway===
    config_data.set_curr_section('normal', "Normal Pathway")

    N_FMS_NORM = config_data.add_elem("numberFMsPerLayerNormal", required=True,
                                      description='Num. of Feature Maps per Layer',
                                      info="This list should have as many entries as the number of layers in "
                                           "the normal pathway.\n"
                                           "Each entry is an integer that specifies the number of Feature Maps "
                                           "to use in each of the layers.")
    KERN_DIM_NORM = config_data.add_elem("kernelDimPerLayerNormal", required=True,
                                         description='Kernel Dimensions per Layer',
                                         info="This list should have as many entries as the number of layers in "
                                              "the normal pathway.\n"
                                              "Each entry should be a sublist with 3 entries. These should specify "
                                              "the dimensions of the kernel at the corresponding layer.")
    RESID_CONN_LAYERS_NORM = config_data.add_elem("layersWithResidualConnNormal", required=False,
                                                  description='Layers with Residual Connections',
                                                  info="List with number of layers, at the output of which to make a "
                                                       "residual connection with the input of the previous layer. "
                                                       "Ala Kaiming He et al, \"Deep Residual Learning for Image "
                                                       "Recognition\".\n"
                                                       "Note: Numbering starts from 1 for the first layer, which is "
                                                       "not an acceptable value (no previous layer).\n"
                                                       "Example: [4,6,8] will connect (add) to the output of Layer 4 "
                                                       "the input of Layer 3. Also, input to 5th will be added to "
                                                       "output of 6th, and input of 7th to output of 8th.",
                                                  default=[])
    LOWER_RANK_LAYERS_NORM = config_data.add_elem("lowerRankLayersNormal", required=False,
                                                  description='Lower Rank Layers',
                                                  info="Layers to make of lower rank. Ala Yani Ioannou et al, "
                                                       "\"Training CNNs with Low-Rank Filters For Efficient Image "
                                                       "Classification\".\n"
                                                       "Example: [3,5] will make the 3rd and 5th layers of lower rank.",
                                                  default=[])
    
    # ==Subsampled pathway==
    config_data.set_curr_section('sub', "Subsampled Pathway")

    USE_SUBSAMPLED = config_data.add_elem("useSubsampledPathway", elem_type='Bool', widget_type='checkbox',
                                          description='Use Subsampled Pathway',
                                          info="Specify whether to use a subsampled pathway. If False, all "
                                               "subsampled-related parameters will be read but disregarded in the "
                                               "model-construction.",
                                          default=False)
    # The below should be mirroring the pathway,
    # otherwise let them specify them but throw warning all around that receptive field should stay the same!
    N_FMS_SUBS = config_data.add_elem("numberFMsPerLayerSubsampled",
                                      description='Num. of Feature Maps per Layer',
                                      info="This list should have as many entries as the number of layers in "
                                           "the subsampled pathway.\n"
                                           "Each entry is an integer that specifies the number of Feature Maps "
                                           "to use in each of the layers.\n"
                                           "If ommitted and useSubsampledPathway is set to True, the subsampled pathway"
                                           " will be made similar to the normal pathway (suggested for easy use).\n"
                                           "[WARN] Subsampled pathway MUST have the same size of receptive field as "
                                           "the normal (limitation in the code). User could easily specify different "
                                           "number of FMs. But care must be given if number of layers is changed. "
                                           "In this case, kernel sizes should also be adjusted to achieve same size "
                                           "of Rec.Field.")
    KERN_DIM_SUBS = config_data.add_elem("kernelDimPerLayerSubsampled",
                                         description='Kernel Dimensions per Layer',
                                         info="This list should have as many entries as the number of layers in "
                                              "the subsampled pathway.\n"
                                              "Each entry should be a sublist with 3 entries. These should specify "
                                              "the dimensions of the kernel at the corresponding layer.\n"
                                              "If ommitted and useSubsampledPathway is set to True, the subsampled "
                                              "pathway will be made similar to the normal pathway "
                                              "(suggested for easy use).\n"
                                              "[WARN] Subsampled pathway MUST have the same size of receptive field as "
                                              "the normal (limitation in the code). The user could easily specify a "
                                              "different number of FMs, but care must be given if the number of layers "
                                              "is changed. In this case, kernel sizes should also be adjusted to "
                                              "achieve the same size of Rec. Field.")
    SUBS_FACTOR = config_data.add_elem("subsampleFactor",
                                       description='Subsampling Factor',
                                       info="How much to downsample the image that the subsampled-pathway processes.\n"
                                            "Requires either a) list of 3 integers, "
                                            "or b) list of lists of 3 integers.\n"
                                            "Example a) [3, 3, 3]   Creates one additional parallel pathway, where "
                                            "input is subsampled by 3 in the x,y,z axis (the 3 elements of the list).\n"
                                            "Example b) [[3, 3, 3], [5, 5, 5]]   Creates two additional parallel "
                                            "pathways. One with input subsampled by [3, 3, 3], and one subsampled by "
                                            "[5, 5, 5]. If not specified, each path mirrors the previous.",
                                       default=[[3, 3, 3]])
    RESID_CONN_LAYERS_SUBS = config_data.add_elem("layersWithResidualConnSubsampled",
                                                  description='Layers with Residual Connections',
                                                  info="List with number of layers, at the output of which to make a "
                                                       "residual connection with the input of the previous layer. "
                                                       "Ala Kaiming He et al, \"Deep Residual Learning for Image "
                                                       "Recognition\".\n"
                                                       "Note: Numbering starts from 1 for the first layer, which is "
                                                       "not an acceptable value (no previous layer).\n"
                                                       "Example: [4,6,8] will connect (add) to the output of Layer 4 "
                                                       "the input of Layer 3. Also, input to 5th will be added to "
                                                       "output of 6th, and input of 7th to output of 8th.\n"
                                                       "(default: mirrors normal pathway)")
    LOWER_RANK_LAYERS_SUBS = config_data.add_elem("lowerRankLayersSubsampled",
                                                  description='Lower Rank Layers',
                                                  info="Layers to make of lower rank. Ala Yani Ioannou et al, "
                                                       "\"Training CNNs with Low-Rank Filters For Efficient Image "
                                                       "Classification\".\n"
                                                       "Example: [3,5] will make the 3rd and 5th layers "
                                                       "of lower rank.\n"
                                                       "(default: mirrors the normal pathway)")
    
    # ==Extra hidden FC Layers. Final Classification layer is not included in here.
    config_data.set_curr_section('fc', "Fully Connected Layers")

    N_FMS_FC = config_data.add_elem("numberFMsPerLayerFC",
                                    description='Num. of Feature Maps per Layer',
                                    info="After the last layers of the normal and subsampled pathways are "
                                         "concatenated, additional Fully Connected hidden layers can be added before "
                                         "the final classification layer.\n"
                                         "Specify a list, with as many entries as the number of ADDITIONAL FC layers "
                                         "(other than the classification layer) to add. "
                                         "The entries specify the number of Feature Maps to use.",
                                    default=[])
    KERN_DIM_1ST_FC = config_data.add_elem("kernelDimFor1stFcLayer",
                                           description='Kernel Dimensions for 1st Layer',
                                           info="Specify dimensions of the kernel in the first FC layer. This kernel "
                                                "combines the features from multiple scales. Applies to the final "
                                                "Classification layer if no hidden FC layers in network.\n"
                                                "Note: convolution with this kernel retains the size of the FMs "
                                                "(input is padded).",
                                           default=[])
    RESID_CONN_LAYERS_FC = config_data.add_elem("layersWithResidualConnFC",
                                                description='Layers with Residual Connections',
                                                info="List with number of layers, at the output of which to make a "
                                                    "residual connection with the input of the previous layer. "
                                                    "Ala Kaiming He et al, \"Deep Residual Learning for Image "
                                                    "Recognition\".\n"
                                                    "Note: Numbering starts from 1 for the first layer, which is "
                                                    "not an acceptable value (no previous layer).\n"
                                                    "Example: [4,6,8] will connect (add) to the output of Layer 4 "
                                                    "the input of Layer 3. Also, input to 5th will be added to "
                                                    "output of 6th, and input of 7th to output of 8th.\n",
                                                default=[])
    
    # Size of Image Segments
    config_data.set_curr_section('seg', "Size of Image Segments")

    SEG_DIM_TRAIN = config_data.add_elem("segmentsDimTrain", required=True,
                                         description='Training Segments Dimensions',
                                         info="DeepMedic does not process patches of the image, but larger "
                                              "image-segments. Specify the size of the training segments here.\n"
                                              "Note: Size of training segments influence the captured distribution of "
                                              "samples from the different classes (see DeepMedic paper)")
    SEG_DIM_VAL = config_data.add_elem("segmentsDimVal",
                                       description='Validation Segments Dimensions',
                                       info="DeepMedic does not process patches of the image, but larger "
                                            "image-segments. Specify the size of segments to use during the "
                                            "validation-on-samples process that is performed throughout training "
                                            "if requested.\n"
                                            "(default: receptive field size, to validate on patches)")
    SEG_DIM_INFER = config_data.add_elem("segmentsDimInference",
                                         description='Inference Segments Dimensions',
                                         info="DeepMedic does not process patches of the image, but larger "
                                              "image-segments. Specify the size of the inference segments here.\n"
                                              "Note: Bigger image segments for Inference are safe to use and only "
                                              "speed up the process. Only limitation is the GPU memory.\n"
                                              "(default: size of training segments)")
    
    # Dropout Rates:
    config_data.set_curr_section('drop', "Dropout Rates")

    DROP_NORM = config_data.add_elem("dropoutRatesNormal", description='Normal Pathway Dropout Rates',
                                     info="Dropout Rates on the input connections of the various layers. Each list "
                                          "should have as many entries as the number of layers in the normal pathway.\n"
                                          "Example: 0 = no dropout. 1= 100% drop of the neurons. "
                                          "Empty list for no dropout.",
                                     default=[])
    DROP_SUBS = config_data.add_elem("dropoutRatesSubsampled", description='Subsampled Pathway Dropout Rates',
                                     info="Dropout Rates on the input connections of the various layers. Each list "
                                          "should have as many entries as the number of layers in the "
                                          "subsampled pathway.\n"
                                          "Example: 0 = no dropout. 1= 100% drop of the neurons. "
                                          "Empty list for no dropout.",
                                     default=[])
    DROP_FC = config_data.add_elem("dropoutRatesFc", description='FC Layers Dropout Rates',
                                   info="Dropout Rates on the input connections of the various layers. Each list "
                                        "should have as many entries as the number of layers in the "
                                        "FC layers.\n"
                                        "Example: 0 = no dropout. 1= 100% drop of the neurons. "
                                        "Empty list for no dropout.\n"
                                        "Note: The list for FC rates should have one additional entry in comparison "
                                        "to \"" + N_FMS_FC.name + "\", for the classification layer.\n"
                                        "(default: 50% dropout on every Fully Connected layer except for the "
                                        "first one after the concatenation)")

    # Other
    config_data.set_curr_section('other', "Other Options")
    
    # Initialization method of the kernel weights.
    CONV_W_INIT = config_data.add_elem("convWeightsInit", elem_type='Conv_w', widget_type='conv_w',
                                       description='Kernel Weight Initialisation Method',
                                       info='Initialization method for the convolution kernel weights;\n'
                                            'Options: ["normal", std] for sampling from N(0, std). '
                                            '["fanIn", scale] for scaling variance with (scale/fanIn). '
                                            'E.g. ["fanIn", 2] initializes a la "Delving Deep into Rectifiers".',
                                       default="fanIn",
                                       options={'fanIn': ConfigElem("init_scale",
                                                                    description='Initialisation Scale',
                                                                    info="scale for scaling variance with "
                                                                         "(scale/fanIn)",
                                                                    default=2),
                                                "normal": ConfigElem("init_std",
                                                                     description='Initialisation Standard Deviation',
                                                                     info="Standard deviation for sampling from "
                                                                          "N(0, std)")})
    # Activation Function for all convolutional layers:
    ACTIV_FUNC = config_data.add_elem("activationFunction", elem_type='String', widget_type='combobox',
                                      options=["linear", "relu", "prelu", "elu", "selu"],
                                      description='Activation Function',
                                      info="Activation Function for all convolutional layers.",
                                      default="prelu")
    
    # Batch Normalization
    BN_ROLL_AV_BATCHES = config_data.add_elem("rollAverageForBNOverThatManyBatches",
                                              description='Num. Batches for Batch Norm Rolling Average',
                                              info="Batch Normalization uses a rolling average of the "
                                                   "mus and std for inference.\n"
                                                   "Specify over how many batches (SGD iterations) this moving "
                                                   "average should be computed. Value <= 0 disables BN.",
                                              default=60)

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
    
    

