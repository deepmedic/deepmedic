# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

from deepmedic.frontEnd.configParsing.utils import *
from deepmedic.frontEnd.configParsing.config import Config, ConfigData


class TrainConfig(Config):
    config_data = ConfigData()

    config_data.set_curr_section('session', "Session Parameters")
    
    # Optional but highly suggested.
    SESSION_NAME = config_data.add_elem("sessionName", elem_type='String')
    # [REQUIRED]
    FOLDER_OUTP = config_data.add_elem("folderForOutput", elem_type='String', required=True)
    SAVED_MODEL = config_data.add_elem("cnnModelFilePath", elem_type='String', required=True)
    TENSORBOARD_LOG = config_data.add_elem("tensorboardLogtensorboardLog", required=True)

    # =============TRAINING========================
    config_data.set_curr_section('input', "Input")

    CHANNELS_TR = config_data.add_elem("channelsTraining", required=True)
    GT_LABELS_TR = config_data.add_elem("gtLabelsTraining", elem_type='String', required=True)
    
    # ~~~~ Sampling (training) ~~~~~
    config_data.set_curr_section('sampling_train', "Sampling (Training)")

    ROI_MASKS_TR = config_data.add_elem("roiMasksTraining", elem_type='String')

    TYPE_OF_SAMPLING_TR = config_data.add_elem("typeOfSamplingForTraining")
    PROP_OF_SAMPLES_PER_CAT_TR = config_data.add_elem("proportionOfSamplesToExtractPerCategoryTraining")
    WEIGHT_MAPS_PER_CAT_FILEPATHS_TR = config_data.add_elem("weightedMapsForSamplingEachCategoryTrain")
    
    # ~~~~~ Training cycle ~~~~~~~~
    config_data.set_curr_section('train_cycle', "Training Cycle")

    NUM_EPOCHS = config_data.add_elem("numberOfEpochs")
    NUM_SUBEP = config_data.add_elem("numberOfSubepochs")
    NUM_CASES_LOADED_PERSUB = config_data.add_elem("numOfCasesLoadedPerSubepoch")
    NUM_TR_SEGMS_LOADED_PERSUB = config_data.add_elem("numberTrainingSegmentsLoadedOnGpuPerSubep")
    BATCHSIZE_TR = config_data.add_elem("batchsize_train")
    NUM_OF_PROC_SAMPL = config_data.add_elem("num_processes_sampling")

    # ~~~~~ Preprocessing ~~~~~~~~
    config_data.set_curr_section('preproc', "Preprocessing")

    NORM = config_data.add_elem('norm')
    INT_NORM = config_data.add_elem('intensity_norm')
    CO_PERCENT = config_data.add_elem("cutoff_percent")
    CO_STD = config_data.add_elem("cutoff_std")
    CO_MEAN = config_data.add_elem("cutoff_mean")
    
    # ~~~~~ Learning rate schedule ~~~~~
    config_data.set_curr_section('lr_schedule', "Learning Rate Schedule")

    LR_SCH_TYPE = config_data.add_elem("typeOfLearningRateSchedule")
    # Stable + Auto + Predefined.
    DIV_LR_BY = config_data.add_elem("whenDecreasingDivideLrBy")
    # Stable + Auto
    NUM_EPOCHS_WAIT = config_data.add_elem("numEpochsToWaitBeforeLoweringLr")
    # Auto:
    AUTO_MIN_INCR_VAL_ACC = config_data.add_elem("min_incr_of_val_acc_considered")
    # Predefined.
    PREDEF_SCH = config_data.add_elem("predefinedSchedule")
    # Exponential
    EXPON_SCH = config_data.add_elem("paramsForExpSchedForLrAndMom")

    # ~~~~ Data Augmentation~~~
    config_data.set_curr_section('data_aug', "Data Augmentation")

    AUGM_IMG_PRMS_TR = config_data.add_elem("augm_img_prms_tr")
    AUGM_SAMPLE_PRMS_TR = config_data.add_elem("augm_sample_prms_tr")
    
    # ============== VALIDATION ===================
    config_data.set_curr_section('val', "Validation")

    PERFORM_VAL_SAMPLES = config_data.add_elem("performValidationOnSamplesThroughoutTraining",
                                               elem_type='Bool', default=True)
    PERFORM_VAL_INFERENCE = config_data.add_elem("performFullInferenceOnValidationImagesEveryFewEpochs",
                                                 elem_type='Bool', default=True)
    CHANNELS_VAL = config_data.add_elem("channelsValidation")
    GT_LABELS_VAL = config_data.add_elem("gtLabelsValidation")
    ROI_MASKS_VAL = config_data.add_elem("roiMasksValidation")
    NUM_VAL_SEGMS_LOADED_PERSUB = config_data.add_elem("numberValidationSegmentsLoadedOnGpuPerSubep")  # For val on samples.
    BATCHSIZE_VAL_SAMPL = config_data.add_elem("batchsize_val_samples", required=True)
    
    # ~~~~~~~~ Sampling (validation) ~~~~~~~~~~~~
    config_data.set_curr_section('sampling_val', "Sampling (Validation)")

    TYPE_OF_SAMPLING_VAL = config_data.add_elem("typeOfSamplingForVal")
    PROP_OF_SAMPLES_PER_CAT_VAL = config_data.add_elem("proportionOfSamplesToExtractPerCategoryVal")
    WEIGHT_MAPS_PER_CAT_FILEPATHS_VAL = config_data.add_elem("weightedMapsForSamplingEachCategoryVal")
    
    # ~~~~~~~~~ Validation by fully inferring whole validation cases ~~~~~~~~
    config_data.set_curr_section('val_whole', "Validation on Whole Volumes")

    NUM_EPOCHS_BETWEEN_VAL_INF = config_data.add_elem("numberOfEpochsBetweenFullInferenceOnValImages")
    BATCHSIZE_VAL_WHOLE = config_data.add_elem("batchsize_val_whole")
    NAMES_FOR_PRED_PER_CASE_VAL = config_data.add_elem("namesForPredictionsPerCaseVal")
    SAVE_SEGM_VAL = config_data.add_elem("saveSegmentationVal", elem_type='Bool', default=True)
    SAVE_PROBMAPS_PER_CLASS_VAL = config_data.add_elem("saveProbMapsForEachClassVal")
    SUFFIX_SEGM_PROB_VAL = config_data.add_elem("suffixForSegmAndProbsDictVal")
    SAVE_INDIV_FMS_VAL = config_data.add_elem("saveIndividualFmsVal")
    SAVE_4DIM_FMS_VAL = config_data.add_elem("saveAllFmsIn4DimImageVal")
    INDICES_OF_FMS_TO_SAVE_NORMAL_VAL = config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathwayVal")
    INDICES_OF_FMS_TO_SAVE_SUBSAMPLED_VAL = config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathwayVal")
    INDICES_OF_FMS_TO_SAVE_FC_VAL = config_data.add_elem("minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathwayVal")
    
    # ====OPTIMIZATION=====
    config_data.set_curr_section('optimisation', "Optimisation")

    LRATE = config_data.add_elem("learningRate")
    OPTIMIZER = config_data.add_elem("sgd0orAdam1orRms2")
    MOM_TYPE = config_data.add_elem("classicMom0OrNesterov1")
    MOM = config_data.add_elem("momentumValue")
    MOM_NORM_NONNORM = config_data.add_elem("momNonNorm0orNormalized1")
    # Adam
    B1_ADAM = config_data.add_elem("b1Adam")
    B2_ADAM = config_data.add_elem("b2Adam")
    EPS_ADAM = config_data.add_elem("epsilonAdam")
    # RMS
    RHO_RMS = config_data.add_elem("rhoRms")
    EPS_RMS = config_data.add_elem("epsilonRms")
    # Losses
    LOSSES_WEIGHTS = config_data.add_elem("losses_and_weights")
    W_C_IN_COST = config_data.add_elem("reweight_classes_in_cost")
    # Regularization L1 and L2.
    L1_REG = config_data.add_elem("L1_reg")
    L2_REG = config_data.add_elem("L2_reg")
    
    # ~~~  Freeze Layers ~~~
    config_data.set_curr_section('freeze_layers', "Freeze Layers")

    LAYERS_TO_FREEZE_NORM = config_data.add_elem("layersToFreezeNormal")
    LAYERS_TO_FREEZE_SUBS = config_data.add_elem("layersToFreezeSubsampled")
    LAYERS_TO_FREEZE_FC = config_data.add_elem("layersToFreezeFC")
    
    # ========= GENERICS =========
    config_data.set_curr_section('generic', "Generic")

    PAD_INPUT = config_data.add_elem("padInputImagesBool", elem_type='Bool', default=True)
    RUN_INP_CHECKS = config_data.add_elem("run_input_checks", elem_type='Bool', default=True)
    
    # ======== DEPRECATED, backwards compatibility =======
    REFL_AUGM_PER_AXIS = config_data.add_elem("reflectImagesPerAxis", advanced=True)
    PERF_INT_AUGM_BOOL = config_data.add_elem("performIntAugm", advanced=True)
    INT_AUGM_SHIF_MUSTD = config_data.add_elem("sampleIntAugmShiftWithMuAndStd", advanced=True)
    INT_AUGM_MULT_MUSTD = config_data.add_elem("sampleIntAugmMultiWithMuAndStd", advanced=True)
    OLD_AUGM_SAMPLE_PRMS_TR = config_data.add_elem("augm_params_tr", advanced=True)
    
    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)

    # If certain config args are given in command line, completely override the corresponding ones in the config files.
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        if args.saved_model is not None:
            abs_path_model_cmd_line = getAbsPathEvenIfRelativeIsGiven(args.saved_model, os.getcwd())
            if self.get(self.SAVED_MODEL) is not None:
                log.print3("WARN: A model to load was specified both in the command line and in the train-config file!"
                           "\n\t The input by the command line will be used: " + str(abs_path_model_cmd_line))
            
            self._configStruct[self.SAVED_MODEL] = abs_path_model_cmd_line
