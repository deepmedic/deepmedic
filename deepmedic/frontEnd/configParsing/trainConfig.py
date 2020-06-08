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


class TrainConfig(Config):
    
    # Optional but highly suggested.
    SESSION_NAME = "sessionName"
    # [REQUIRED]
    FOLDER_OUTP = "folderForOutput"
    SAVED_MODEL = "cnnModelFilePath"
    TENSORBOARD_LOG = "tensorboard_log"

    # =============TRAINING========================
    DATAFRAME_TR = "dataframe_train"
    CHANNELS_TR = "channelsTraining"
    GT_LBLS_TR = "gtLabelsTraining"
    ROIS_TR = "roiMasksTraining"
    
    # ~~~~ Sampling (training) ~~~~~
    TYPE_OF_SAMPLING_TR = "typeOfSamplingForTraining"
    PROP_OF_SAMPLES_PER_CAT_TR = "proportionOfSamplesToExtractPerCategoryTraining"
    WEIGHT_MAPS_PER_CAT_FILEPATHS_TR = "weightedMapsForSamplingEachCategoryTrain"
    
    # ~~~~~ Training cycle ~~~~~~~~
    NUM_EPOCHS = "numberOfEpochs"
    NUM_SUBEP = "numberOfSubepochs"
    NUM_CASES_LOADED_PERSUB = "numOfCasesLoadedPerSubepoch"
    NUM_TR_SEGMS_LOADED_PERSUB = "numberTrainingSegmentsLoadedOnGpuPerSubep"
    BATCHSIZE_TR = "batchsize_train"
    NUM_OF_PROC_SAMPL = "num_processes_sampling"
    
    # ~~~~~ Learning rate schedule ~~~~~
    LR_SCH_TYPE = "typeOfLearningRateSchedule"
    # Stable + Auto + Predefined.
    DIV_LR_BY = "whenDecreasingDivideLrBy"
    # Stable + Auto
    NUM_EPOCHS_WAIT = "numEpochsToWaitBeforeLoweringLr"
    # Auto:
    AUTO_MIN_INCR_VAL_ACC = "min_incr_of_val_acc_considered"
    # Predefined.
    PREDEF_SCH = "predefinedSchedule"
    # Exponential
    EXPON_SCH = "paramsForExpSchedForLrAndMom"
    # ~~~~ Data Augmentation~~~
    AUGM_IMG_PRMS_TR = "augm_img_prms_tr"
    AUGM_SAMPLE_PRMS_TR = "augm_sample_prms_tr"
    
    # ============== VALIDATION ===================
    PERFORM_VAL_SAMPLES = "performValidationOnSamplesThroughoutTraining"
    PERFORM_VAL_INFERENCE = "performFullInferenceOnValidationImagesEveryFewEpochs"
    DATAFRAME_VAL = "dataframe_val"
    CHANNELS_VAL = "channelsValidation"
    GT_LBLS_VAL = "gtLabelsValidation"
    ROIS_VAL = "roiMasksValidation"
    NUM_VAL_SEGMS_LOADED_PERSUB = "numberValidationSegmentsLoadedOnGpuPerSubep"  # For val on samples.
    BATCHSIZE_VAL_SAMPL = "batchsize_val_samples"
    
    # ~~~~~~~~ Sampling (validation) ~~~~~~~~~~~~
    TYPE_OF_SAMPLING_VAL = "typeOfSamplingForVal"
    PROP_OF_SAMPLES_PER_CAT_VAL = "proportionOfSamplesToExtractPerCategoryVal"
    WEIGHT_MAPS_PER_CAT_FILEPATHS_VAL = "weightedMapsForSamplingEachCategoryVal"
    
    # ~~~~~~~~~ Validation by fully inferring whole validation cases ~~~~~~~~
    NUM_EPOCHS_BETWEEN_VAL_INF = "numberOfEpochsBetweenFullInferenceOnValImages"
    BATCHSIZE_VAL_WHOLE = "batchsize_val_whole"
    FNAMES_PREDS_VAL = "namesForPredictionsPerCaseVal"
    SAVE_SEGM_VAL = "saveSegmentationVal"
    SAVE_PROBMAPS_PER_CLASS_VAL = "saveProbMapsForEachClassVal"
    SUFFIX_SEGM_PROB_VAL = "suffixForSegmAndProbsDictVal"
    SAVE_INDIV_FMS_VAL = "saveIndividualFmsVal"
    INDICES_OF_FMS_TO_SAVE_NORMAL_VAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathwayVal"
    INDICES_OF_FMS_TO_SAVE_SUBSAMPLED_VAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathwayVal"
    INDICES_OF_FMS_TO_SAVE_FC_VAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathwayVal"
    
    # ====OPTIMIZATION=====
    LRATE = "learningRate"
    OPTIMIZER = "sgd0orAdam1orRms2"
    MOM_TYPE = "classicMom0OrNesterov1"
    MOM = "momentumValue"
    MOM_NORM_NONNORM = "momNonNorm0orNormalized1"
    # Adam
    B1_ADAM = "b1Adam"
    B2_ADAM = "b2Adam"
    EPS_ADAM = "epsilonAdam"
    # RMS
    RHO_RMS = "rhoRms"
    EPS_RMS = "epsilonRms"
    # Losses
    LOSSES_WEIGHTS = "losses_and_weights"
    W_C_IN_COST = "reweight_classes_in_cost"
    # Regularization L1 and L2.
    L1_REG = "L1_reg"
    L2_REG = "L2_reg"
    
    # ~~~  Freeze Layers ~~~
    LAYERS_TO_FREEZE_NORM = "layersToFreezeNormal"
    LAYERS_TO_FREEZE_SUBS = "layersToFreezeSubsampled"
    LAYERS_TO_FREEZE_FC = "layersToFreezeFC"
    
    # ========= GENERICS =========
    # ~~~~ Data compabitiliby checks ~~~
    RUN_INP_CHECKS = "run_input_checks"
    # ~~~~~ Preprocessing ~~~~~~~~
    PAD_INPUT = "padInputImagesBool"
    NORM_VERB_LVL = "norm_verbosity_lvl"
    NORM_ZSCORE_PRMS = "norm_zscore_prms"
    
    # ======== DEPRECATED, backwards compatibility =======
    REFL_AUGM_PER_AXIS = "reflectImagesPerAxis"
    PERF_INT_AUGM_BOOL = "performIntAugm"
    INT_AUGM_SHIF_MUSTD = "sampleIntAugmShiftWithMuAndStd"
    INT_AUGM_MULT_MUSTD = "sampleIntAugmMultiWithMuAndStd"
    OLD_AUGM_SAMPLE_PRMS_TR = "augm_params_tr"
    
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
