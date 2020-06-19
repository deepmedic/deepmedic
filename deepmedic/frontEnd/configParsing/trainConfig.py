# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os

from deepmedic.frontEnd.configParsing.utils import abs_from_rel_path
from deepmedic.frontEnd.configParsing.config import Config, ConfigData, ConfigElem
from deepmedic.frontEnd.configParsing.modelConfig import ModelConfig


class TrainConfig(Config):
    config_data = ConfigData()

    config_data.set_curr_section('session', "Session Parameters")
    
    # Optional but highly suggested.
    SESSION_NAME = \
        config_data.add_elem("sessionName", elem_type='String', description="Session Name",
                             info="[Optional but highly suggested] The name will be used for saving the "
                                  "models,logs and results.",
                             default="trainSession")

    # [REQUIRED]
    FOLDER_OUTP = \
        config_data.add_elem("folderForOutput", elem_type='Folder', required=True, description="Output Folder",
                             info="The main folder that the output will be placed.")
    SAVED_MODEL = \
        config_data.add_elem("cnnModelFilePath", elem_type='File', description="Saved Model Checkpoint",
                             info="Path to a saved model, to load parameters from at beginning of the "
                                  "session. If one is also specified in the command line/GUI, the latter "
                                  "will be used.")

    TENSORBOARD_LOG = \
        config_data.add_elem("tensorboardLog", elem_type='Bool', description="Log to Tensorboard", default=False)

    # =============TRAINING========================
    config_data.set_curr_section('train', "Training")

    DATAFRAME_TR = \
        config_data.add_elem("dataframe_train", elem_type='File', description="Input DataFrame",
                             info="a CSV file, the dataframe, that contains paths to all input files. One row per case."
                                  "Each row provides paths to channels (image modalities), roi-mask, "
                                  "ground-truth annotations etc for the case.\n"
                                  "See documentation for details. If not given, provide 'channels', 'roimasks', "
                                  "'gtLabels', 'namesForPredictionsPerCase'.")

    CHANNELS_TR = \
        config_data.add_elem("channelsTraining", description="Input Channels", required=True,
                             elem_type='Files',
                             info="A list that should contain as many entries as the channels of the input "
                                  "image (eg multi-modal MRI). The entries should be paths to files.\n"
                                  "Those files should be listing the paths to the corresponding channels "
                                  "for each training-case. (see example files).\n"
                                  'Example: ["./trainChannels_flair.cfg", "./trainChannels_t1c.cfg"]'
                             )
    GT_LBLS_TR = \
        config_data.add_elem("gtLabelsTraining", description="Ground Truth Labels", elem_type='File', required=True,
                             info='The path to a file which should list paths to the Ground Truth labels '
                                  'of each training case.')

    ROIS_TR = \
        config_data.add_elem("roiMasksTraining", description="RoI Masks", elem_type='File',
                             info="The path to a file, which should list paths to the Region-Of-Interest masks for "
                                  "each training case.\n"
                                  "If ROI masks are provided, the training samples will be extracted only within it. "
                                  "Otherwise from whole volume.")

    # ~~~~ Sampling (training) ~~~~~
    config_data.set_curr_section('sampling_train', "Sampling (Training)")

    TYPE_OF_SAMPLING_TR = \
        config_data.add_elem("typeOfSamplingForTraining", widget_type="combobox",
                             options=[0, 1, 2, 3], default=3, description="Type of Sampling",
                             info="Type of Sampling to use for training.\n"
                                  "[Possible Values] 0 = Foreground / Background, 1 = Uniform, "
                                  "2 = Whole Image (Not impl yet), 3 = Separately-Per-Class.\n"
                                  "Note: In case of (2) Full Image, ensure you provide \"" +
                                  ModelConfig.config_data.get_elem(
                                      ModelConfig.SEG_DIM_TRAIN).description +
                                  "\" in modelConfig.cfg at least as big as image dimensions "
                                  "(+ CNN's receptive field if padding is used)."
                             )
    PROP_OF_SAMPLES_PER_CAT_TR = \
        config_data.add_elem("proportionOfSamplesToExtractPerCategoryTraining",
                             description="Proportion of Samples to Extract",
                             info="List the proportion (0.0 to 1.0) of samples to extract from each "
                                  "category of samples.\n"
                                  "Note: Depending on the Type-of-Sampling chosen, list must be of the form:\n"
                                  ">> Fore/Background: [proportion-of-FOREground-samples, "
                                  "proportion-of-BACKground-samples], eg [0.3, 0.7]. "
                                  "IMPORTANT: FOREground first, background second!\n"
                                  ">> Uniform or Full-Image: Not Applicable and disregarded if given.\n"
                                  ">> Separate sampling of each class: [proportion-of-class-0(background), ..., "
                                  "proportion-of-class-N]\n"
                                  "Note: Values will be internally normalized (to add up to 1.0).\n"
                                  "(default: Foreground/Background or Separately-Each-Class : equal number of segments "
                                  "extracted for each of the categories. Uniform or Full-Image: N/A)"
                             )
    WEIGHT_MAPS_PER_CAT_FILEPATHS_TR = \
        config_data.add_elem("weightedMapsForSamplingEachCategoryTrain",
                             description="Weighted Maps for Sampling",
                             info="This variable allows providing weighted-maps to indicate where to extract more "
                                  "segments for each category of samples. Higher weight means more samples "
                                  "from that area.\n"
                                  "The value provided should be a List with paths to files. As many files as the "
                                  "categories of samples for the chosen Sampling-Type.\n"
                                  "Similarly to the files listing the Ground Truth, Channels, etc per subject, "
                                  "these files should list the paths to the weight-maps of each subject for the "
                                  "corresponding category.\n"
                                  "Note: Number of files required: Fore/Backgr:2, Uniform:1, Full-Image:N/A, Separate "
                                  "each class:NumOfOutputClasses (Incl Backgr).\n"
                                  "IMPORTANT: Sequence of weight-maps is important!\n"
                                  ">> If Fore/Background type of sampling, provide for the FOREground first!\n"
                                  ">> If Separately sampling each class, provide weightmap-files in the same sequence "
                                  "as the class-labels in your Ground Truth! "
                                  "Eg background-0 first, class-1 second, etc\n"
                                  "(default : If this variable is not provided, samples are extracted based on the "
                                  "Ground-Truth labels and the ROI.)"
                             )
    
    # ~~~~~ Training cycle ~~~~~~~~
    config_data.set_curr_section('train_cycle', "Training Cycle")

    NUM_EPOCHS = \
        config_data.add_elem("numberOfEpochs", description="Number of Epochs",
                             info="[Optional but highly suggested as this parameter is model dependent.]\n"
                                  "How many epochs to train for.",
                             default=35)
    NUM_SUBEP = \
        config_data.add_elem("numberOfSubepochs", description="Number of Subepochs",
                             info="[Optional but highly suggested as this parameter is model dependent.]\n"
                                  "How many subepochs comprise an epoch. "
                                  "Every subepoch gets Accuracy reported for.",
                             default=20)
    NUM_CASES_LOADED_PERSUB = \
        config_data.add_elem("numOfCasesLoadedPerSubepoch", description="Num. of Cases per Subepoch",
                             info="[Optional but highly suggested as this parameter is model dependent.]\n"
                                  "Every subepoch, load the images from this many cases and extract new "
                                  "training samples from.",
                             default=50)
    NUM_TR_SEGMS_LOADED_PERSUB = \
        config_data.add_elem("numberTrainingSegmentsLoadedOnGpuPerSubep",
                             description="Num. of Loaded Segments per Subepoch",
                             info="[Optional but highly suggested as this parameter is model dependent.]\n"
                                  "Every subepoch, extract in total this many segments and load them on the GPU. "
                                  "(Memory Limited)\n"
                                  "Note: This number in combination with the batchsize define the number of "
                                  "optimization steps per subepoch (= NumOfSegmentsOnGpu / BatchSize).",
                             default=1000)

    BATCHSIZE_TR = \
        config_data.add_elem("batchsize_train", description="Batch Size", info=" Batch size for training.")
    NUM_OF_PROC_SAMPL = \
        config_data.add_elem("num_processes_sampling", description="Num. of CPUs for Sampling",
                             info="Number of CPUs for sampling\n"
                                  "-1: No parallelism. 0: One parallel thread. 1,2,3...: Parallel processes spawned. ",
                             default=0)

    # ~~~~~ Preprocessing ~~~~~~~~
    config_data.set_curr_section('preproc', "Preprocessing")

    NORM = config_data.add_elem('norm', elem_type='Bool', description="Normalise Data", default=False)
    INT_NORM = config_data.add_elem('intensity_norm', elem_type='Bool', description="Intensity Normalisation",
                                    parent=NORM, default=False)
    CO_PERCENT = config_data.add_elem("cutoff_percent", description="Cutoff Percentile", parent=INT_NORM)
    CO_STD = config_data.add_elem("cutoff_std", description="Cutoff Standard Deviation", parent=INT_NORM)
    CO_MEAN = config_data.add_elem("cutoff_mean", description="Cutoff Whole Image Mean", parent=INT_NORM)
    
    # ~~~~~ Learning rate schedule ~~~~~
    config_data.set_curr_section('lr_schedule', "Learning Rate Schedule")

    LR_SCH_TYPE = \
        config_data.add_elem("typeOfLearningRateSchedule", description="Learning Rate Schedule Type",
                             elem_type='String', widget_type='combobox',
                             options=["stable", "predef", "poly", "auto"], default="poly",
                             info="The type of schedule to use for Learning Rate annealing. Schedule types:\n"
                                  "'stable' : stable LR.\n"
                                  "'predef' : lowering at predefined epochs.\n"
                                  "'poly' : lr=lr_base * (1-iter/max_iter) ^ 0.9 (from PSPNet)\n"
                                  "'auto' : Lower LR when validation accuracy plateaus.\n"
                                  "Note: LR schedule is important. We suggest running stable, "
                                  "observing when training error plateaus, and defined your predefined schedule.\n"
                                  "Otherwise, use poly with large-enough number of epochs.")
    # Stable + Auto + Predefined.
    DIV_LR_BY = \
        config_data.add_elem("whenDecreasingDivideLrBy",
                             description="LR Fraction When Decreasing",
                             info="[Auto & Predefined] By how much to divide LR when lowering.",
                             default=2)
    # Stable + Auto
    NUM_EPOCHS_WAIT = \
        config_data.add_elem("numEpochsToWaitBeforeLoweringLr", description="LR Decreasing Patience",
                             info="[Stable + Auto] How many epochs to initially wait before decreasing LR first time. "
                                  "For 'auto', this period specifies when val accuracy has plateaued. "
                                  "Irrelevant for 'predef'.")
    # Auto:
    AUTO_MIN_INCR_VAL_ACC = \
        config_data.add_elem("min_incr_of_val_acc_considered", advanced=True)
    # Predefined.
    PREDEF_SCH = \
        config_data.add_elem("predefinedSchedule", description="Predefined Schedule",
                             info="[Required for Predefined] At which epochs to lower LR.")
    # Exponential
    EXPON_SCH = \
        config_data.add_elem("paramsForExpSchedForLrAndMom", advanced=True)

    # ~~~~ Data Augmentation~~~
    config_data.set_curr_section('data_aug', "Data Augmentation")

    AUGM_IMG_PRMS_TR = \
        config_data.add_elem("augm_img_prms_tr", widget_type='multiple',
                             description="Image-level Augmentation",
                             default=None,
                             options={
                                 'affine': ConfigElem('affine', widget_type='multiple', description="Affine",
                                                      options={
                                                          'prob':
                                                              ConfigElem('prob', description="Probability",
                                                                         info="Chance [0.-1.] to augment an image "
                                                                              "(suggested: 0.5)",
                                                                         default=0),
                                                          'max_rot_xyz':
                                                              ConfigElem('max_rot_xyz',
                                                                         description='Maximum Rotation per Axis',
                                                                         info="Max degrees rotation per axis (x,y,z).\n"
                                                                              "Example: (45, 45, 45)"),
                                                          'max_scaling':
                                                              ConfigElem('max_scaling', description="Maximum Scaling",
                                                                         info="Maximum Scaling ([0-1])"),
                                                          'interp_order_imgs':
                                                              ConfigElem('interp_order_imgs', widget_type='combobox',
                                                                         description="Interpolation Order",
                                                                         options=[0, 1, 2],
                                                                         info="Interpolation order for images (0, 1 "
                                                                              "or 2), higher is better but slower "
                                                                              "(suggested: 1 or 2).")
                                                      })
                             })
    AUGM_SAMPLE_PRMS_TR = \
        config_data.add_elem("augm_sample_prms_tr", widget_type='multiple',
                             description="Segment-level Augmentation",
                             options={
                                 'hist_dist': ConfigElem('hist_dist', widget_type='multiple',
                                                         description='Histogram',
                                                         info="Shift and scale the intensity histogram.\n"
                                                              "I' = (I + shift) * scale.\n"
                                                              "Shift and scale values are sampled from "
                                                              "Gaussians N(mu,std).",
                                                         options={
                                                             'shift':
                                                                 ConfigElem('shift', widget_type='multiple',
                                                                            description='Shift',
                                                                            options={
                                                                                'mu': ConfigElem('mu',
                                                                                                 description='Mean'
                                                                                                 ),
                                                                                'std': ConfigElem('std',
                                                                                                  description=
                                                                                                  'Standard Deviation'
                                                                                                  )
                                                                                 }),
                                                             'scale':
                                                                 ConfigElem('scale', widget_type='multiple',
                                                                            description='Scale',
                                                                            options={
                                                                                'mu': ConfigElem('mu',
                                                                                                 description='Mean'
                                                                                                 ),
                                                                                'std': ConfigElem('std',
                                                                                                  description=
                                                                                                  'Standard Deviation'
                                                                                                  )
                                                                                 })
                                                         }),
                                 'reflect': ConfigElem('reflect', description='Reflection',
                                                       info='Augment by flipping samples. '
                                                            'Specify probabilities to flip X,Y,Z axis. '
                                                            'Set None for disabling.\n'
                                                            'Example: (0.5, 0, 0)'),
                                 'rotate90': ConfigElem('rotate90', description='90-degree Rotations',
                                                        widget_type='multiple',
                                                        info="Augment by rotating samples on xy,yz,xz planes by "
                                                             "0,90,180,270 degrees. (suggested: image-level 'affine'"
                                                             " seems better but slower)\n"
                                                             "Give probabilities of flipping a plane by 0,90,180,270 "
                                                             "degrees. Sum is internally normalised to 1.",
                                                        options={
                                                            'xy': ConfigElem('xy', description='xy Plane',
                                                                             widget_type='multiple',
                                                                             options={
                                                                                 '0': ConfigElem('0',
                                                                                                 description=
                                                                                                 "0 degrees"),
                                                                                 '90': ConfigElem('90',
                                                                                                  description=
                                                                                                  "90 degrees"),
                                                                                 '180': ConfigElem('180',
                                                                                                   description=
                                                                                                   "180 degrees"),
                                                                                 '270': ConfigElem('270',
                                                                                                   description=
                                                                                                   "270 degrees"),
                                                                             }),
                                                            'yz': ConfigElem('yz', description='yz Plane',
                                                                             widget_type='multiple',
                                                                             options={
                                                                                 '0': ConfigElem('0',
                                                                                                 description=
                                                                                                 "0 degrees"),
                                                                                 '90': ConfigElem('90',
                                                                                                  description=
                                                                                                  "90 degrees"),
                                                                                 '180': ConfigElem('180',
                                                                                                   description=
                                                                                                   "180 degrees"),
                                                                                 '270': ConfigElem('270',
                                                                                                   description=
                                                                                                   "270 degrees"),
                                                                             }),
                                                            'xz': ConfigElem('xz', description='xz Plane',
                                                                             widget_type='multiple',
                                                                             options={
                                                                                 '0': ConfigElem('0',
                                                                                                 description=
                                                                                                 "0 degrees"),
                                                                                 '90': ConfigElem('90',
                                                                                                  description=
                                                                                                  "90 degrees"),
                                                                                 '180': ConfigElem('180',
                                                                                                   description=
                                                                                                   "180 degrees"),
                                                                                 '270': ConfigElem('270',
                                                                                                   description=
                                                                                                   "270 degrees"),
                                                                             })
                                                        })
                             })
    
    # ============== VALIDATION ===================
    config_data.set_curr_section('val', "Validation")

    PERFORM_VAL_SAMPLES = \
        config_data.add_elem("performValidationOnSamplesThroughoutTraining",
                             elem_type='Bool', default=False,
                             description="Perform Validation on Samples",
                             info="Specify whether to perform validation on samples during training")
    PERFORM_VAL_INFERENCE = \
        config_data.add_elem("performFullInferenceOnValidationImagesEveryFewEpochs",
                             elem_type='Bool', default=False,
                             description="Perform Full Inference Validation",
                             info="Specify whether to perform full-inference validation every few epochs.")

    DATAFRAME_VAL = \
        config_data.add_elem("dataframe_val", elem_type='File', description="Input DataFrame",
                             info="a CSV file, the dataframe, that contains paths to all input files. One row per case."
                                  "Each row provides paths to channels (image modalities), roi-mask, "
                                  "ground-truth annotations etc for the case.\n"
                                  "See documentation for details. If not given, provide 'channels', 'roimasks', "
                                  "'gtLabels', 'namesForPredictionsPerCase'.")

    CHANNELS_VAL = \
        config_data.add_elem("channelsValidation", description="Validation Channels",
                             elem_type='Files',
                             info="A list that should contain as many entries as the channels of the input "
                                  "image (eg multi-modal MRI). The entries should be paths to files.\n"
                                  "Those files should be listing the paths to the corresponding channels "
                                  "for each validation-case. (see example files).\n"
                                  'Example: ["./validationChannels_flair.cfg", "./validationChannels_t1c.cfg"]'
                             )
    GT_LBLS_VAL = \
        config_data.add_elem("gtLabelsValidation", elem_type='File',
                             description="Validation Labels",
                             info='[Required for validation on samples, optional for full-inference] The path to a '
                                  'file which should list paths to the Ground Truth labels of each validation case.'
                             )
    ROIS_VAL = \
        config_data.add_elem("roiMasksValidation", elem_type='File', description="RoI Masks for Validation",
                             info="The path to a file, which should list paths to the Region-Of-Interest masks for "
                                  "each validation case.\n"
                                  "If ROI masks are provided, the validation samples will be extracted only within it. "
                                  "Otherwise from whole volume."
                             )
    NUM_VAL_SEGMS_LOADED_PERSUB = \
        config_data.add_elem("numberValidationSegmentsLoadedOnGpuPerSubep", default=3000,
                             description="Num. of Loaded Segments per Subepoch",
                             parent=PERFORM_VAL_SAMPLES,
                             info="Every subepoch, extract in total this many segments and load them on the GPU. "
                                  "(Memory Limited)\n"
                                  "Only influences how accurately the validation samples will represent whole data. ")
    BATCHSIZE_VAL_SAMPL = \
        config_data.add_elem("batchsize_val_samples", default=50,
                             description="Batch Size",
                             parent=PERFORM_VAL_SAMPLES,
                             info="Batch size for validation on sampled image segments.")

    # ~~~~~~~~ Sampling (validation) ~~~~~~~~~~~~
    config_data.set_curr_section('sampling_val', "Sampling (Validation)")

    TYPE_OF_SAMPLING_VAL = \
        config_data.add_elem("typeOfSamplingForVal", widget_type="combobox",
                             options=[0, 1, 2, 3], default=1, description="Type of Sampling",
                             parent=PERFORM_VAL_SAMPLES,
                             info="Type of Sampling to use for validation.\n"
                                  "[Possible Values] 0 = Foreground / Background, 1 = Uniform, "
                                  "2 = Whole Image (Not impl yet), 3 = Separately-Per-Class."
                             )
    PROP_OF_SAMPLES_PER_CAT_VAL = \
        config_data.add_elem("proportionOfSamplesToExtractPerCategoryVal",
                             description="Proportion of Samples to Extract",
                             parent=PERFORM_VAL_SAMPLES,
                             info="List the proportion (0.0 to 1.0) of samples to extract from each "
                                  "category of samples.\n"
                                  "Note: Depending on the Type-of-Sampling chosen, list must be of the form:\n"
                                  ">> Fore/Background: [proportion-of-FOREground-samples, "
                                  "proportion-of-BACKground-samples], eg [0.3, 0.7]. "
                                  "IMPORTANT: FOREground first, background second!\n"
                                  ">> Uniform or Full-Image: Not Applicable and disregarded if given.\n"
                                  ">> Separate sampling of each class: [proportion-of-class-0(background), ..., "
                                  "proportion-of-class-N]\n"
                                  "Note: Values will be internally normalized (to add up to 1.0).\n"
                                  "(default: Foreground/Background or Separately-Each-Class : equal number of "
                                  "segments extracted for each of the categories. Uniform or Full-Image: N/A)")
    WEIGHT_MAPS_PER_CAT_FILEPATHS_VAL = \
        config_data.add_elem("weightedMapsForSamplingEachCategoryVal",
                             description="Weighted Maps for Sampling",
                             elem_type='Files',
                             parent=PERFORM_VAL_SAMPLES,
                             info="This variable allows providing weighted-maps to indicate where to extract more "
                                  "segments for each category of samples. Higher weight means more samples "
                                  "from that area.\n"
                                  "The value provided should be a List with paths to files. As many files as the "
                                  "categories of samples for the chosen Sampling-Type.\n"
                                  "Similarly to the files listing the Ground Truth, Channels, etc per subject, "
                                  "these files should list the paths to the weight-maps of each subject for the "
                                  "corresponding category.\n"
                                  "Note: Number of files required: Fore/Backgr:2, Uniform:1, Full-Image:N/A, Separate "
                                  "each class:NumOfOutputClasses (Incl Backgr).\n"
                                  "IMPORTANT: Sequence of weight-maps is important!\n"
                                  ">> If Fore/Background type of sampling, provide for the FOREground first!\n"
                                  ">> If Separately sampling each class, provide weightmap-files in the same sequence "
                                  "as the class-labels in your Ground Truth! "
                                  "Eg background-0 first, class-1 second, etc\n"
                                  "(default: If this variable is not provided, samples are extracted based on the "
                                  "Ground-Truth labels and the ROI.)")
    
    # ~~~~~~~~~ Validation by fully inferring whole validation cases ~~~~~~~~
    config_data.set_curr_section('val_whole', "Validation on Whole Volumes")

    NUM_EPOCHS_BETWEEN_VAL_INF = \
        config_data.add_elem("numberOfEpochsBetweenFullInferenceOnValImages",
                             description="Inference Validation",
                             parent=PERFORM_VAL_INFERENCE,
                             info="How often (epochs) to perform validation by fully inferring validation volumes. "
                                  "Time consuming.",
                             default=1)
    BATCHSIZE_VAL_WHOLE = \
        config_data.add_elem("batchsize_val_whole",
                             description="Batch Size",
                             parent=PERFORM_VAL_INFERENCE,
                             info="Batch size for validation on whole volumes.",
                             default=10)
    FNAMES_PREDS_VAL = \
        config_data.add_elem("namesForPredictionsPerCaseVal", elem_type='File',
                             description="",
                             parent=PERFORM_VAL_INFERENCE,
                             info="[Required if requested to save results] The path to a file, which should list names "
                                  "for each validation case, to name the results after.")
    SAVE_SEGM_VAL = \
        config_data.add_elem("saveSegmentationVal", elem_type='Bool', default=True,
                             description="Save Validation Segmentation",
                             info="Specify whether to save the segmentation for each class.\n"
                                  "(default: True to all)",
                             parent=PERFORM_VAL_INFERENCE)
    SAVE_PROBMAPS_PER_CLASS_VAL = \
        config_data.add_elem("saveProbMapsForEachClassVal",
                             description="Save per Class Probability Maps",
                             info="Specify whether to save the probability maps for each class.",
                             parent=PERFORM_VAL_INFERENCE)
    SUFFIX_SEGM_PROB_VAL = \
        config_data.add_elem("suffixForSegmAndProbsDictVal",
                             advanced=True,
                             parent=PERFORM_VAL_INFERENCE)

    config_data.set_curr_section('fms', "Feature Maps", advanced=True)

    SAVE_INDIV_FMS_VAL = \
        config_data.add_elem("saveIndividualFmsVal",
                             elem_type='bool',
                             parent=PERFORM_VAL_INFERENCE)
    INDICES_OF_FMS_TO_SAVE_NORMAL_VAL = \
        config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathwayVal",
                             parent=PERFORM_VAL_INFERENCE)
    INDICES_OF_FMS_TO_SAVE_SUBSAMPLED_VAL = \
        config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathwayVal",
                             parent=PERFORM_VAL_INFERENCE)
    INDICES_OF_FMS_TO_SAVE_FC_VAL = \
        config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathwayVal",
                             parent=PERFORM_VAL_INFERENCE)

    # ====OPTIMIZATION=====
    config_data.set_curr_section('optimisation', "Optimisation")

    LRATE = \
        config_data.add_elem("learningRate", description="Initial Learning Rate", default=0.001)
    OPTIMIZER = \
        config_data.add_elem("sgd0orAdam1orRms2", description="Optimiser", widget_type="combobox",
                             options=[0, 1, 2],
                             info="Optimizer to use. 0 for classic SGD, 1 for Adam, 2 for RmsProp.",
                             default=2)
    MOM_TYPE = \
        config_data.add_elem("classicMom0OrNesterov1", description="Momentum Type", widget_type="combobox",
                             options=[0, 1],
                             info="Type of momentum to use. 0 for standard momentum, 1 for Nesterov.",
                             default=1)
    MOM = \
        config_data.add_elem("momentumValue", description="Momentum Value", default=0.6)
    MOM_NORM_NONNORM = \
        config_data.add_elem("momNonNorm0orNormalized1", description="Normalised Momentum", widget_type="combobox",
                             options=[0, 1],
                             info="Non-Normalized (0) or Normalized momentum (1).\n"
                                  "Bear in mind that Normalized mom may result in smaller gradients and might need "
                                  "relatively higher Learning Rate.",
                             default=1)
    # Adam
    B1_ADAM = \
        config_data.add_elem("b1Adam", advanced=True)
    B2_ADAM = \
        config_data.add_elem("b2Adam", advanced=True)
    EPS_ADAM = \
        config_data.add_elem("epsilonAdam", advanced=True)
    # RMS
    RHO_RMS = \
        config_data.add_elem("rhoRms", description="RmsProp - rho",
                             default=0.9,
                             info="Rho parameter for RmsProp")
    EPS_RMS = \
        config_data.add_elem("epsilonRms", description="RmsProp - e",
                             default=10**(-4),
                             info="e parameter for RmsProp")
    # Losses
    LOSSES_WEIGHTS = \
        config_data.add_elem("losses_and_weights", description="Losses and Weights",
                             widget_type='multiple',
                             info="Losses and their weights for the total cost.\n"
                                  "Note: Give no weight for a cost to not be computed at all (faster)",
                             options={"xentr": ConfigElem("xentr", description="Cross-entropy",
                                                          default=1),
                                      "iou": ConfigElem("iou",
                                                        description="Intersection-over-Union (IoU)"),
                                      "dsc": ConfigElem("dsc", description="DICE Score (DSC)")})
    W_C_IN_COST = \
        config_data.add_elem("reweight_classes_in_cost", advanced=True)
    # Regularization L1 and L2.
    L1_REG = \
        config_data.add_elem("L1_reg", description="L1 Regularisation", default=0.000001)
    L2_REG = \
        config_data.add_elem("L2_reg", description="L1 Regularisation", default=0.0001)
    
    # ~~~  Freeze Layers ~~~
    config_data.set_curr_section('freeze_layers', "Freeze Layers")

    LAYERS_TO_FREEZE_NORM = \
        config_data.add_elem("layersToFreezeNormal", default=[],
                             description="Layers to Freeze (Normal)",
                             info="Specify layers the weights of which you wish to be kept fixed during training "
                                  "(e.g. to use weights from pre-training). First layer is 1.\n"
                                  "One list for each of the normal, subsampled, and fully-connected (as 1x1 convs) "
                                  "pathways. For instance, provide [1,2,3] to keep first 3 layers fixed. [] "
                                  "or remove entry to train all layers.")
    LAYERS_TO_FREEZE_SUBS = \
        config_data.add_elem("layersToFreezeSubsampled",
                             description="Layers to Freeze (Subsampled)",
                             info="Specify layers the weights of which you wish to be kept fixed during training "
                                  "(e.g. to use weights from pre-training). First layer is 1.\n"
                                  "One list for each of the normal, subsampled, and fully-connected (as 1x1 convs) "
                                  "pathways. For instance, provide [1,2,3] to keep first 3 layers fixed. [] "
                                  "or remove entry to train all layers.\n"
                                  "(default: if entry is not specified, we mirror the option used "
                                  "for the Normal pathway.)")
    LAYERS_TO_FREEZE_FC = \
        config_data.add_elem("layersToFreezeFC", default=[],
                             description="Layers to Freeze (Fully Connected)",
                             info="Specify layers the weights of which you wish to be kept fixed during training "
                                  "(e.g. to use weights from pre-training). First layer is 1.\n"
                                  "One list for each of the normal, subsampled, and fully-connected (as 1x1 convs) "
                                  "pathways. For instance, provide [1,2,3] to keep first 3 layers fixed. [] "
                                  "or remove entry to train all layers.")
    
    # ========= GENERICS =========
    config_data.set_curr_section('generic', "Generic")

    RUN_INP_CHECKS = \
        config_data.add_elem("run_input_checks", elem_type='Bool', default=True,
                             description="Run Input Checks",
                             info="Checks for format correctness of loaded input images. Can slow down the process.")

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

    # ======== DEPRECATED, backwards compatibility =======
    REFL_AUGM_PER_AXIS = \
        config_data.add_elem("reflectImagesPerAxis", advanced=True)
    PERF_INT_AUGM_BOOL = \
        config_data.add_elem("performIntAugm", advanced=True)
    INT_AUGM_SHIF_MUSTD = \
        config_data.add_elem("sampleIntAugmShiftWithMuAndStd", advanced=True)
    INT_AUGM_MULT_MUSTD = \
        config_data.add_elem("sampleIntAugmMultiWithMuAndStd", advanced=True)
    OLD_AUGM_SAMPLE_PRMS_TR = \
        config_data.add_elem("augm_params_tr", advanced=True)
    
    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)

    # If certain config args are given in command line, completely override the corresponding ones in the config files.
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        if args.saved_model is not None:
            abs_path_model_cmd_line = abs_from_rel_path(args.saved_model, os.getcwd())
            if self.get(self.SAVED_MODEL) is not None:
                log.print3("WARN: A model to load was specified both in the command line and in the train-config file!"
                           "\n\t The input by the command line will be used: " + str(abs_path_model_cmd_line))
            
            self._configStruct[self.SAVED_MODEL] = abs_path_model_cmd_line
