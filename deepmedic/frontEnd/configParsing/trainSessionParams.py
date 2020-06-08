# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import os
import pandas as pd

from deepmedic.frontEnd.configParsing.utils import abs_from_rel_path, parse_filelist, check_and_adjust_path_to_ckpt, \
    get_paths_from_df, parse_fpaths_of_channs_from_filelists
from deepmedic.dataManagement import samplingType
from deepmedic.dataManagement.augmentImage import AugmenterAffineParams


def get_default(value, default, required=False):
    if value is not None:
        return value
    else:
        if default is not None or not required:
            return default
        else:
            raise Exception('Required Element')


def get_config_value(cfg, elem):
    return get_default(cfg[elem.name], elem.default, elem.required)


class TrainSessionParameters(object):

    # To be called from outside too.
    @staticmethod
    def get_session_name(sessionName):
        return sessionName if sessionName is not None else "trainSession"


    # REQUIRED:
    @staticmethod
    def errorRequireChannelsTraining():
        print(
            "ERROR: Parameter \"channelsTraining\" needed but not provided in config file. "
            "This parameter should provide paths to files, as many as the channels (modalities) of the task. "
            "Each of the files should contain a list of paths, one for each case to train on. "
            "These paths in a file should point to the .nii(.gz) files that are the corresponding channel for a "
            "patient. Please provide it in the format: channelsTraining = [\"path-to-file-for-channel1\", ..., "
            "\"path-to-file-for-channelN\"]. The paths should be given in quotes, separated by commas "
            "(list of strings, python-style). Exiting.")
        exit(1)

    errReqChansTr = errorRequireChannelsTraining

    @staticmethod
    def errorRequireGtLabelsTraining():
        print(
            "ERROR: Parameter \"gtLabelsTraining\" needed but not provided in config file. "
            "This parameter should provide the path to a file. That file should contain a list of paths, "
            "one for each case to train on. These paths should point to the .nii(.gz) files that contain the c"
            "orresponding Ground-Truth labels for a case. Please provide it in the format: "
            "gtLabelsTraining = \"path-to-file\". The path should be given in quotes (a string, python-style). "
            "Exiting.")
        exit(1)

    errReqGtTr = errorRequireGtLabelsTraining

    @staticmethod
    def errorRequireBatchsizeTrain():
        print("ERROR: Please provide size of batch size in train-config. See parameter \'batchsize\'. Exiting.")
        exit(1)

    errReqBatchSizeTr = errorRequireBatchsizeTrain

    @staticmethod
    def errorRequirePredefinedLrSched():
        print(
            "ERROR: Parameter \"typeOfLearningRateSchedule\" was set to \"predefined\", but no predefined schedule "
            "was given. Please specify at which epochs to lower the Learning Rate, by providing the corresponding "
            "parameter in the format: predefinedSchedule = [epoch-for-1st-decrease, ..., epoch-for-last-decrease], "
            "where the epochs are specified by an integer > 0. Exiting.")
        exit(1)

    errReqPredLrSch = errorRequirePredefinedLrSched

    @staticmethod
    def errorAutoRequiresValSamples():
        print(
            "ERROR: Parameter \"typeOfLearningRateSchedule\" was set to \"auto\". This requires performing validation "
            "on samples throughout training, because this schedule lowers the Learning Rate when validation-accuracy "
            "plateaus. However the parameter \"performValidationOnSamplesThroughoutTraining\" was set to False in the "
            "configuration file, or was ommitted, which triggers the default value, False! Please set the parameter "
            "performValidationOnSamplesThroughoutTraining = True. You will then need to provide the path to the "
            "channels of the validation cases in the format: channelsValidation = "
            "[\"path-to-file-that-lists-paths-to-channel-1-for-every-case\", ..., "
            "\"path-to-file-that-lists-paths-to-channel-N-for-every-case\"] (python style list-of-strings)."
            "\t Also, you will need to provide the Ground-Truth for the validation cases, in the format:  "
            "gtLabelsValidation = \"path-to-file\", where the file lists the paths to the GT labels of each validation "
            "case. Exiting!")
        exit(1)

    @staticmethod
    def errorRequireChannelsVal():
        print(
            "ERROR: Parameter \"channelsValidation\" was not provided, although it is required to perform validation, "
            "although validation was requested (parameters \"performValidationOnSamplesThroughoutTraining\" or "
            "\"performFullInferenceOnValidationImagesEveryFewEpochs\" was set to True). You will need to provide a "
            "list with path to files that list where the channels for each validation case can be found. "
            "The corresponding parameter must be provided in the format: channelsValidation = "
            "[\"path-to-file-that-lists-paths-to-channel-1-for-every-case\", ..., "
            "\"path-to-file-that-lists-paths-to-channel-N-for-every-case\"] (python style list-of-strings). Exiting.")
        exit(1)

    errReqChannsVal = errorRequireChannelsVal

    @staticmethod
    def errorReqGtLabelsVal():
        print(
            "ERROR: Parameter \"gtLabelsValidation\" was not provided, although it is required to perform validation "
            "on training-samples, which was requested (parameter \"performValidationOnSamplesThroughoutTraining\" "
            "was set to True). It is also useful so that the DSC score is reported if full-inference on the "
            "validation samples is performed (when parameter \"performFullInferenceOnValidationImagesEveryFewEpochs\" "
            "is set to True)! You will need to provide the path to a file that lists where the GT labels for each "
            "validation case can be found. The corresponding parameter must be provided in the format: "
            "gtLabelsValidation = \"path-to-file-that-lists-GT-labels-for-every-case\" (python style string). Exiting.")
        exit(1)

    @staticmethod
    def errorIntNormZScoreTwoAppliesGiven():
        print("ERROR: In training-config, for the variable (dictionary) norm_zscore_prms,"
              "\n\tif ['apply_to_all_channels': True] then it must ['apply_per_channel': None]"
              "\n\tOtherwise, requires ['apply_to_all_channels': False] if ['apply_per_channel': [..list..] ]"
              "\n\tExiting!")
        exit(1)

    # VALIDATION
    @staticmethod
    def errorReqn_epochsBetweenFullValInfGreaterThan0():
        print(
            "ERROR: It was requested to perform full-inference on validation images by setting parameter "
            "\"performFullInferenceOnValidationImagesEveryFewEpochs\" to True. For this, it is required to specify "
            "the number of epochs between two full-inference procedures. This number was given equal to 0. "
            "Please specify a number greater than 0, in the format: n_epochsBetweenFullInferenceOnValImages = 1 "
            "(Any integer. Default is 1). Exiting!")
        exit(1)

    @staticmethod
    def errorRequireNamesOfPredictionsVal():
        print(
            "ERROR: It was requested to perform full-inference on validation images by setting parameter "
            "\"performFullInferenceOnValidationImagesEveryFewEpochs\" to True and then save some of the results "
            "(segmentation maps, probability maps or feature maps), either manually or by default. For this, it is "
            "required to specify the path to a file, which should contain names to give to the results. "
            "Please specify the path to such a file in the format: namesForPredictionsPerCaseVal = "
            "\"./validation/validationNamesOfPredictionsSimple.cfg\" (python-style string), "
            "or in the column 'pred' of the input csv, if that's the chosen option. Exiting!")
        exit(1)

    @staticmethod
    def errorRequireOptimizer012():
        print("ERROR: The parameter \"sgd0orAdam1orRms2\" must be given 0,1 or 2. Omit for default. Exiting!")
        exit(1)

    @staticmethod
    def errorRequireMomentumClass0Nestov1():
        print("ERROR: The parameter \"classicMom0OrNesterov1\" must be given 0 or 1. Omit for default. Exiting!")
        exit(1)

    @staticmethod
    def errorRequireMomValueBetween01():
        print("ERROR: The parameter \"momentumValue\" must be given between 0.0 and 1.0 Omit for default. Exiting!")
        exit(1)

    @staticmethod
    def errorRequireMomNonNorm0Norm1():
        print("ERROR: The parameter \"momNonNorm0orNormalized1\" must be given 0 or 1. Omit for default. Exiting!")
        exit(1)

    def __init__(self,
                 log,
                 main_outp_folder,
                 out_folder_models,
                 out_folder_preds,
                 out_folder_fms,
                 n_classes,
                 model_name,
                 cfg):

        # Importants for running session.
        # From Session:
        self.log = log
        self.main_outp_folder = main_outp_folder  # absolute

        # From Config:
        self._session_name = self.get_session_name(cfg[cfg.SESSION_NAME])

        abs_path_cfg = cfg.get_abs_path_to_cfg()
        path_model = abs_from_rel_path(cfg[cfg.SAVED_MODEL], abs_path_cfg) if cfg[cfg.SAVED_MODEL] is not None else None
        self.model_ckpt_path = check_and_adjust_path_to_ckpt(self.log, path_model) if path_model is not None else None

        self.tensorboardLog = cfg[cfg.TENSORBOARD_LOG] if cfg[cfg.TENSORBOARD_LOG] is not None else False

        # ====================TRAINING==========================
        self.filepath_to_save_models = out_folder_models + "/" + model_name + "." + self._session_name

        if cfg[cfg.DATAFRAME_TR] is not None:
            self.csv_fname_train = abs_from_rel_path(cfg[cfg.DATAFRAME_TR], abs_path_cfg)
            try:
                self.dataframe_tr = pd.read_csv(self.csv_fname_train, skipinitialspace=True)
            except OSError:  # FileNotFoundError exception only in Py3, which is child of OSError.
                raise OSError("File given for dataframe (train) does not exist: " + self.csv_fname_train)
            (self.channels_fpaths_tr,
             self.gt_fpaths_tr,
             self.roi_fpaths_tr,
             _) = get_paths_from_df(self.log, self.dataframe_tr, os.path.dirname(self.csv_fname_train), req_gt=True)
        else:
            self.csv_fname_train = None
            self.dataframe_tr = None
            if cfg[cfg.CHANNELS_TR] is None:
                self.errReqChansTr()
            if cfg[cfg.GT_LBLS_TR] is None:
                self.errReqGtTr()

            self.channels_fpaths_tr = parse_fpaths_of_channs_from_filelists(cfg[cfg.CHANNELS_TR], abs_path_cfg)
            self.gt_fpaths_tr = parse_filelist(abs_from_rel_path(cfg[cfg.GT_LBLS_TR], abs_path_cfg), make_abs=True)
            self.roi_fpaths_tr = parse_filelist(abs_from_rel_path(cfg[cfg.ROIS_TR], abs_path_cfg), make_abs=True) \
                if cfg[cfg.ROIS_TR] is not None else None

        # [Optionals]
        # ~~~~~~~~~Sampling~~~~~~~
        sampling_type_flag_tr = cfg[cfg.TYPE_OF_SAMPLING_TR] if cfg[cfg.TYPE_OF_SAMPLING_TR] is not None else 3
        self.sampling_type_inst_tr = samplingType.SamplingType(self.log, sampling_type_flag_tr, n_classes)
        if sampling_type_flag_tr in [0, 3] and cfg[cfg.PROP_OF_SAMPLES_PER_CAT_TR] is not None:
            self.sampling_type_inst_tr.set_perc_of_samples_per_cat(cfg[cfg.PROP_OF_SAMPLES_PER_CAT_TR])
        else:
            n_sampl_cats_tr = self.sampling_type_inst_tr.get_n_sampling_cats()
            self.sampling_type_inst_tr.set_perc_of_samples_per_cat(
                [1.0 / n_sampl_cats_tr] * n_sampl_cats_tr)

        self.paths_to_wmaps_per_sampl_cat_per_subj_train = None
        if cfg[cfg.WEIGHT_MAPS_PER_CAT_FILEPATHS_TR] is not None:
            # [[case1-weightMap1, ..., caseN-weightMap1], [case1-weightMap2,...,caseN-weightMap2]]
            self.paths_to_wmaps_per_sampl_cat_per_subj_train = [
                parse_filelist(abs_from_rel_path(weightMapConfPath, abs_path_cfg), make_abs=True)
                for weightMapConfPath in cfg[cfg.WEIGHT_MAPS_PER_CAT_FILEPATHS_TR]
            ]

        # ~~~~~~~~ Training Cycle ~~~~~~~~~~~
        self.n_epochs = cfg[cfg.NUM_EPOCHS] if cfg[cfg.NUM_EPOCHS] is not None else 35
        self.n_subepochs = cfg[cfg.NUM_SUBEP] if cfg[cfg.NUM_SUBEP] is not None else 20
        self.max_n_cases_per_subep_train = \
            cfg[cfg.NUM_CASES_LOADED_PERSUB] if cfg[cfg.NUM_CASES_LOADED_PERSUB] is not None else 50
        self.n_samples_per_subep_train = \
            cfg[cfg.NUM_TR_SEGMS_LOADED_PERSUB] if cfg[cfg.NUM_TR_SEGMS_LOADED_PERSUB] is not None else 1000
        self.batchsize_train = cfg[cfg.BATCHSIZE_TR] if cfg[cfg.BATCHSIZE_TR] is not None else self.errReqBatchSizeTr()
        self.num_parallel_proc_sampling = cfg[cfg.NUM_OF_PROC_SAMPL] if cfg[cfg.NUM_OF_PROC_SAMPL] is not None else 0

        # ~~~~~~~ Learning Rate Schedule ~~~~~~~~

        assert cfg[cfg.LR_SCH_TYPE] in ['stable', 'predef', 'poly', 'auto', 'expon']
        self.lr_sched_params = {
            'type': cfg[cfg.LR_SCH_TYPE] if cfg[cfg.LR_SCH_TYPE] is not None else 'poly',
            'predef': {'epochs': cfg[cfg.PREDEF_SCH],
                       'div_lr_by': cfg[cfg.DIV_LR_BY] if cfg[cfg.DIV_LR_BY] is not None else 2.0
                       },
            'auto': {'min_incr_of_val_acc_considered':
                         cfg[cfg.AUTO_MIN_INCR_VAL_ACC] if cfg[cfg.AUTO_MIN_INCR_VAL_ACC] is not None else 0.0,
                     'epochs_wait_before_decr': cfg[cfg.NUM_EPOCHS_WAIT] if cfg[cfg.NUM_EPOCHS_WAIT] is not None else 5,
                     'div_lr_by': cfg[cfg.DIV_LR_BY] if cfg[cfg.DIV_LR_BY] is not None else 2.0
                     },
            'poly': {'epochs_wait_before_decr':
                         cfg[cfg.NUM_EPOCHS_WAIT] if cfg[cfg.NUM_EPOCHS_WAIT] is not None else self.n_epochs / 3,
                     'final_ep_for_sch': self.n_epochs
                     },
            'expon': {'epochs_wait_before_decr':
                          cfg[cfg.NUM_EPOCHS_WAIT] if cfg[cfg.NUM_EPOCHS_WAIT] is not None else self.n_epochs / 3,
                      'final_ep_for_sch': self.n_epochs,
                      'lr_to_reach_at_last_ep':
                          cfg[cfg.EXPON_SCH][0] if cfg[cfg.EXPON_SCH] is not None else 1.0 / (2 ** 8),
                      'mom_to_reach_at_last_ep': cfg[cfg.EXPON_SCH][1] if cfg[cfg.EXPON_SCH] is not None else 0.9
                      }
        }
        # Predefined.
        if self.lr_sched_params['type'] == 'predef' and self.lr_sched_params['predef']['epochs'] is None:
            self.errReqPredLrSch()

        # ~~~~~~~~~~~~~~ Augmentation~~~~~~~~~~~~~~
        # Image level
        self.augm_img_prms_tr = {'affine': None}  # If var is None, no augm at all.
        if cfg[cfg.AUGM_IMG_PRMS_TR] is not None:
            self.augm_img_prms_tr['affine'] = AugmenterAffineParams(cfg[cfg.AUGM_IMG_PRMS_TR]['affine'])

        # Patch/Segment level
        self.augm_sample_prms_tr = {'hist_dist': None, 'reflect': None, 'rotate90': None}
        if cfg[cfg.AUGM_SAMPLE_PRMS_TR] is not None:
            for key in cfg[cfg.AUGM_SAMPLE_PRMS_TR]:
                # For exact form of parameters, see ./deepmedic/dataManagement/augmentation.py
                self.augm_sample_prms_tr[key] = cfg[cfg.AUGM_SAMPLE_PRMS_TR][key]

        # ===================VALIDATION========================
        self.val_on_samples_during_train = \
            cfg[cfg.PERFORM_VAL_SAMPLES] if cfg[cfg.PERFORM_VAL_SAMPLES] is not None else False
        if self.lr_sched_params['type'] == 'auto' and not self.val_on_samples_during_train:
            self.errorAutoRequiresValSamples()
        self.val_on_whole_volumes = \
            cfg[cfg.PERFORM_VAL_INFERENCE] if cfg[cfg.PERFORM_VAL_INFERENCE] is not None else False

        # Input:
        if (not self.val_on_samples_during_train) and (not self.val_on_whole_volumes):
            self.csv_fname_val = None
            self.dataframe_val = None
            self.channels_fpaths_val = []
            self.gt_fpaths_val = None
            self.roi_fpaths_val = None
            self.out_preds_fnames_val = None
        elif cfg[cfg.DATAFRAME_VAL] is not None:  # doing one of validations, and given dataframe.
            self.csv_fname_val = abs_from_rel_path(cfg[cfg.DATAFRAME_VAL], abs_path_cfg)
            try:
                self.dataframe_val = pd.read_csv(self.csv_fname_val, skipinitialspace=True)
            except OSError:  # FileNotFoundError exception only in Py3, which is child of OSError.
                raise OSError("File given for dataframe (validation) does not exist: " + self.csv_fname_val)
            (self.channels_fpaths_val,
             self.gt_fpaths_val,
             self.roi_fpaths_val,
             self.out_preds_fnames_val) = get_paths_from_df(self.log,
                                                            self.dataframe_val,
                                                            os.path.dirname(self.csv_fname_val),
                                                            req_gt=True)
        else:  # Doing one of validations, and not given dataframe.
            self.csv_fname_val = None
            self.dataframe_val = None
            if cfg[cfg.CHANNELS_VAL]:
                self.channels_fpaths_val = parse_fpaths_of_channs_from_filelists(cfg[cfg.CHANNELS_VAL], abs_path_cfg)
            else:
                self.errReqChannsVal()

            self.gt_fpaths_val = parse_filelist(abs_from_rel_path(cfg[cfg.GT_LBLS_VAL], abs_path_cfg), make_abs=True) \
                if cfg[cfg.GT_LBLS_VAL] is not None else self.errorReqGtLabelsVal()
            self.roi_fpaths_val = parse_filelist(abs_from_rel_path(cfg[cfg.ROIS_VAL], abs_path_cfg), make_abs=True) \
                if cfg[cfg.ROIS_VAL] is not None else None
            self.out_preds_fnames_val = parse_filelist(abs_from_rel_path(cfg[cfg.FNAMES_PREDS_VAL], abs_path_cfg)) \
                if cfg[cfg.FNAMES_PREDS_VAL] else None

        # ~~~~~Validation on Samples~~~~~~~~
        self.n_samples_per_subep_val = \
            cfg[cfg.NUM_VAL_SEGMS_LOADED_PERSUB] if cfg[cfg.NUM_VAL_SEGMS_LOADED_PERSUB] is not None else 3000
        self.batchsize_val_samples = cfg[cfg.BATCHSIZE_VAL_SAMPL] if cfg[cfg.BATCHSIZE_VAL_SAMPL] is not None else 50

        # ~~~~~~~~~ Sampling (Validation) ~~~~~~~~~~~
        sampling_type_flag_val = cfg[cfg.TYPE_OF_SAMPLING_VAL] if cfg[cfg.TYPE_OF_SAMPLING_VAL] is not None else 1
        self.sampling_type_inst_val = samplingType.SamplingType(self.log, sampling_type_flag_val, n_classes)
        if sampling_type_flag_val in [0, 3] and cfg[cfg.PROP_OF_SAMPLES_PER_CAT_VAL] is not None:
            self.sampling_type_inst_val.set_perc_of_samples_per_cat(cfg[cfg.PROP_OF_SAMPLES_PER_CAT_VAL])
        else:
            n_sampl_cats_val = self.sampling_type_inst_val.get_n_sampling_cats()
            self.sampling_type_inst_val.set_perc_of_samples_per_cat([1.0 / n_sampl_cats_val] * n_sampl_cats_val)

        self.paths_to_wmaps_per_sampl_cat_per_subj_val = None
        if cfg[cfg.WEIGHT_MAPS_PER_CAT_FILEPATHS_VAL] is not None:
            # [[case1-weightMap1, ..., caseN-weightMap1], [case1-weightMap2,...,caseN-weightMap2]]
            self.paths_to_wmaps_per_sampl_cat_per_subj_val = [
                parse_filelist(abs_from_rel_path(wmap_conf_path, abs_path_cfg), make_abs=True)
                for wmap_conf_path in cfg[cfg.WEIGHT_MAPS_PER_CAT_FILEPATHS_VAL]
            ]

        # ~~~~~~Full inference on validation image~~~~~~
        self.num_epochs_between_val_on_whole_volumes = \
            cfg[cfg.NUM_EPOCHS_BETWEEN_VAL_INF] if cfg[cfg.NUM_EPOCHS_BETWEEN_VAL_INF] is not None else 1
        if self.num_epochs_between_val_on_whole_volumes == 0 and self.val_on_whole_volumes:
            self.errorReqn_epochsBetweenFullValInfGreaterThan0()

        self.batchsize_val_whole = cfg[cfg.BATCHSIZE_VAL_WHOLE] if cfg[cfg.BATCHSIZE_VAL_WHOLE] is not None else 10

        # predictions
        self.save_segms_val = cfg[cfg.SAVE_SEGM_VAL] if cfg[cfg.SAVE_SEGM_VAL] is not None else True
        self.save_probs_per_cl_val = cfg[cfg.SAVE_PROBMAPS_PER_CLASS_VAL] \
            if (cfg[cfg.SAVE_PROBMAPS_PER_CLASS_VAL] is not None and cfg[cfg.SAVE_PROBMAPS_PER_CLASS_VAL] != []) \
            else [True] * n_classes
        self.suffixes_for_outp_val = cfg[cfg.SUFFIX_SEGM_PROB_VAL] if cfg[cfg.SUFFIX_SEGM_PROB_VAL] is not None \
            else {"segm": "Segm", "prob": "ProbMapClass"}
        # features:
        self.save_fms_flag_val = cfg[cfg.SAVE_INDIV_FMS_VAL] if cfg[cfg.SAVE_INDIV_FMS_VAL] is not None else False
        if self.save_fms_flag_val is True:
            inds_fms = [cfg[cfg.INDICES_OF_FMS_TO_SAVE_NORMAL_VAL]] + \
                       [cfg[cfg.INDICES_OF_FMS_TO_SAVE_SUBSAMPLED_VAL]] + \
                       [cfg[cfg.INDICES_OF_FMS_TO_SAVE_FC_VAL]]
            # By default, save none.
            self.inds_fms_per_pathtype_per_layer_to_save = [item if item is not None else [] for item in inds_fms]
        else:
            self.inds_fms_per_pathtype_per_layer_to_save = None

        # Output:
        # Filled by call to self.makeFilepathsForPredictionsAndFeatures()
        self.out_preds_fpaths_val = None
        self.out_fms_fpaths_val = None
        # Checks on output params:
        if not self.out_preds_fnames_val and self.val_on_whole_volumes \
                and (self.save_segms_val or True in self.save_probs_per_cl_val
                     or self.save_fms_flag_val):
            self.errorRequireNamesOfPredictionsVal()

        # ===================== PRE-PROCESSING ======================
        # === Data compatibility checks ===
        self.run_input_checks = cfg[cfg.RUN_INP_CHECKS] if cfg[cfg.RUN_INP_CHECKS] is not None else True
        # == Padding ==
        self.pad_input = cfg[cfg.PAD_INPUT] if cfg[cfg.PAD_INPUT] is not None else True
        # == Normalization ==
        norm_zscore_prms = {'apply_to_all_channels': False, # True/False
                            'apply_per_channel': None, # Must be None if above True. Else, List Bool per channel
                            'cutoff_percents': None, # None or [low, high], each from 0.0 to 100. Eg [5.,95.]
                            'cutoff_times_std': None, # None or [low, high], each positive Float. Eg [3.,3.]
                            'cutoff_below_mean': False}
        if cfg[cfg.NORM_ZSCORE_PRMS] is not None:
            for key in cfg[cfg.NORM_ZSCORE_PRMS]:
                norm_zscore_prms[key] = cfg[cfg.NORM_ZSCORE_PRMS][key]
        if norm_zscore_prms['apply_to_all_channels'] and norm_zscore_prms['apply_per_channel'] is not None:
            self.errorIntNormZScoreTwoAppliesGiven()
        if norm_zscore_prms['apply_per_channel'] is not None:
            assert len(norm_zscore_prms['apply_per_channel']) == len(cfg[cfg.CHANNELS_TR]) # num channels
        # Aggregate params from all types of normalization:
        # norm_prms = None : No int normalization will be performed.
        # norm_prms['verbose_lvl']: 0: No logging, 1: Type of cutoffs and timing 2: Stats.
        self.norm_prms = {'verbose_lvl': cfg[cfg.NORM_VERB_LVL] if cfg[cfg.NORM_VERB_LVL] is not None else 0,
                          'zscore': norm_zscore_prms}
        
        # ============= OTHERS ==========
        # Others useful internally or for reporting:
        self.n_cases_tr = len(self.channels_fpaths_tr)
        self.n_cases_val = len(self.channels_fpaths_val)
        
        # ========= HIDDENS =============
        # no config allowed for these at the moment:

        # Re-weight samples in the cost function *on a per-class basis*: Type of re-weighting and training schedule.
        # E.g. to exclude a class, or counter class imbalance.
        # "type": string/None, "prms": any/None, "schedule": [ min_epoch, max_epoch ]
        # Type, prms combinations: "freq", None || "per_c", [0., 2., 1., ...] (as many as classes)
        # "schedule": Constant before epoch [0], linear change towards equal weight (=1) until epoch [1],
        # constant equal weights (=1) afterwards.
        self.reweight_classes_in_cost = cfg[cfg.W_C_IN_COST] if cfg[cfg.W_C_IN_COST] is not None \
            else {"type": None, "prms": None, "schedule": [0, self.n_epochs]}
        if self.reweight_classes_in_cost["type"] == "per_c":
            assert len(self.reweight_classes_in_cost["prms"]) == n_classes

        self._make_fpaths_for_preds_and_fms(out_folder_preds, out_folder_fms)

        # ====Optimization=====
        self.learningRate = cfg[cfg.LRATE] if cfg[cfg.LRATE] is not None else 0.001
        self.optimizerSgd0Adam1Rms2 = cfg[cfg.OPTIMIZER] if cfg[cfg.OPTIMIZER] is not None else 2
        if self.optimizerSgd0Adam1Rms2 == 0:
            self.b1Adam = "placeholder"
            self.b2Adam = "placeholder"
            self.eAdam = "placeholder"
            self.rhoRms = "placeholder"
            self.eRms = "placeholder"
        elif self.optimizerSgd0Adam1Rms2 == 1:
            self.b1Adam = cfg[cfg.B1_ADAM] if cfg[cfg.B1_ADAM] is not None else 0.9  # default in paper and seems good
            self.b2Adam = cfg[cfg.B2_ADAM] if cfg[cfg.B2_ADAM] is not None else 0.999  # default in paper and seems good
            self.eAdam = cfg[cfg.EPS_ADAM] if cfg[cfg.EPS_ADAM] is not None else 10 ** (-8)
            self.rhoRms = "placeholder"
            self.eRms = "placeholder"
        elif self.optimizerSgd0Adam1Rms2 == 2:
            self.b1Adam = "placeholder"
            self.b2Adam = "placeholder"
            self.eAdam = "placeholder"
            self.rhoRms = cfg[cfg.RHO_RMS] if cfg[cfg.RHO_RMS] is not None else 0.9  # default in paper and seems good
            # 1e-6 was the default in the paper, but blew up the gradients in first try. Never tried 1e-5 yet.
            self.eRms = cfg[cfg.EPS_RMS] if cfg[cfg.EPS_RMS] is not None else 10 ** (-4)
        else:
            self.errorRequireOptimizer012()

        self.classicMom0Nesterov1 = cfg[cfg.MOM_TYPE] if cfg[cfg.MOM_TYPE] is not None else 1
        if self.classicMom0Nesterov1 not in [0, 1]:
            self.errorRequireMomentumClass0Nestov1()
        self.momNonNormalized0Normalized1 = cfg[cfg.MOM_NORM_NONNORM] if cfg[cfg.MOM_NORM_NONNORM] is not None else 1
        if self.momNonNormalized0Normalized1 not in [0, 1]:
            self.errorRequireMomNonNorm0Norm1()
        self.momentumValue = cfg[cfg.MOM] if cfg[cfg.MOM] is not None else 0.6
        if self.momentumValue < 0. or self.momentumValue > 1:
            self.errorRequireMomValueBetween01()

        # ==Regularization==
        self.L1_reg_weight = cfg[cfg.L1_REG] if cfg[cfg.L1_REG] is not None else 0.000001
        self.L2_reg_weight = cfg[cfg.L2_REG] if cfg[cfg.L2_REG] is not None else 0.0001

        # ============= HIDDENS ==============
        # Indices of layers that should not be trained (kept fixed).
        layers_to_freeze_per_pathtype = [cfg[cfg.LAYERS_TO_FREEZE_NORM],
                                         cfg[cfg.LAYERS_TO_FREEZE_SUBS],
                                         cfg[cfg.LAYERS_TO_FREEZE_FC]]
        inds_layers_freeze_norm = \
            [l - 1 for l in layers_to_freeze_per_pathtype[0]] if layers_to_freeze_per_pathtype[0] is not None else []
        inds_layers_freeze_subs = \
            [l - 1 for l in layers_to_freeze_per_pathtype[1]] \
            if layers_to_freeze_per_pathtype[1] is not None \
            else inds_layers_freeze_norm
        inds_layers_freeze_fc = \
            [l - 1 for l in layers_to_freeze_per_pathtype[2]] if layers_to_freeze_per_pathtype[2] is not None else []
        # Three sublists, one per pathway type: Normal, Subsampled, FC. eg: [[0,1,2],[0,1,2],[]
        self.inds_layers_per_pathtype_freeze = [inds_layers_freeze_norm, inds_layers_freeze_subs, inds_layers_freeze_fc]

        self.losses_and_weights = cfg[cfg.LOSSES_WEIGHTS] if cfg[cfg.LOSSES_WEIGHTS] is not None else {"xentr": 1.0,
                                                                                                       "iou": None,
                                                                                                       "dsc": None}
        assert True in [self.losses_and_weights[k] is not None for k in ["xentr", "iou", "dsc"]]

        self._backwards_compat_with_deprecated_cfg(cfg)

        """
        #NOTES: variables that have to do with number of pathways: 
                self.inds_layers_per_pathtype_freeze (="all" always currently. Hardcoded)
                inds_fms_per_pathtype_per_layer_to_save (Repeat subsampled!)
        """

    def _backwards_compat_with_deprecated_cfg(self, cfg):
        # Augmentation
        if cfg[cfg.REFL_AUGM_PER_AXIS] is not None:
            self.augm_sample_prms_tr['reflect'] = [0.5 if bool else 0. for bool in cfg[cfg.REFL_AUGM_PER_AXIS]]
        if cfg[cfg.PERF_INT_AUGM_BOOL] is True:
            self.augm_sample_prms_tr['hist_dist'] = {
                'shift': {'mu': cfg[cfg.INT_AUGM_SHIF_MUSTD][0], 'std': cfg[cfg.INT_AUGM_SHIF_MUSTD][1]},
                'scale': {'mu': cfg[cfg.INT_AUGM_MULT_MUSTD][0], 'std': cfg[cfg.INT_AUGM_MULT_MUSTD][1]}}
        if cfg[cfg.OLD_AUGM_SAMPLE_PRMS_TR] is not None:
            self.log.print3(
                "ERROR: In training's config, variable \'augm_params_tr\' is deprecated. "
                "Replace it with \'augm_sample_prms_tr\'.")

    def _make_fpaths_for_preds_and_fms(self, out_folder_preds, out_folder_fms):
        # TODO: Merge with same in testSessionParams
        self.out_preds_fpaths_val = []
        self.out_fms_fpaths_val = []
        if self.out_preds_fnames_val is not None:  # standard behavior
            for case_i in range(self.n_cases_val):
                fpaths_for_case_pred = out_folder_preds + "/" + self.out_preds_fnames_val[case_i]
                self.out_preds_fpaths_val.append(fpaths_for_case_pred)
                fpaths_for_case_fms = out_folder_fms + "/" + self.out_preds_fnames_val[case_i]
                self.out_fms_fpaths_val.append(fpaths_for_case_fms)
        else:  # Names for predictions not given. Special handling...
            if self.n_cases_val > 1:  # Many cases, create corresponding namings for files.
                for case_i in range(self.n_cases_val):
                    self.out_preds_fpaths_val.append(out_folder_preds + "/pred_case" + str(case_i) + ".nii.gz")
                    self.out_fms_fpaths_val.append(out_folder_preds + "/pred_case" + str(case_i) + ".nii.gz")
            else:  # Only one case. Just give the output prediction folder, the io.py will save output accordingly.
                self.out_preds_fpaths_val.append(out_folder_preds)
                self.out_fms_fpaths_val.append(out_folder_preds)

    def get_path_to_load_model_from(self):
        return self.model_ckpt_path

    def get_tensorboard_bool(self):
        return self.tensorboardLog

    def print_params(self):
        logPrint = self.log.print3
        logPrint("")
        logPrint("=============================================================")
        logPrint("========= PARAMETERS FOR THIS TRAINING SESSION ==============")
        logPrint("=============================================================")
        logPrint("Session's name = " + str(self._session_name))
        logPrint("Model will be loaded from save = " + str(self.model_ckpt_path))
        logPrint("~~Output~~")
        logPrint("Main output folder = " + str(self.main_outp_folder))
        logPrint("Log performance metrics for tensorboard = " + str(self.tensorboardLog))
        logPrint("Path and filename to save trained models = " + str(self.filepath_to_save_models))

        logPrint("~~~~~~~~~~~~~~~~~~Generic Information~~~~~~~~~~~~~~~~")
        logPrint("Number of Cases for Training = " + str(self.n_cases_tr))
        logPrint("Number of Cases for Validation = " + str(self.n_cases_val))

        logPrint("~~~~~~~~~~~~~~~~~~Training parameters~~~~~~~~~~~~~~~~")
        logPrint("Dataframe (csv) filename = " + str(self.csv_fname_train))
        logPrint("Filepaths to Channels of the Training Cases = " + str(self.channels_fpaths_tr))
        logPrint("Filepaths to Ground-Truth labels of the Training Cases = " + str(self.gt_fpaths_tr))
        logPrint("Filepaths to ROI Masks of the Training Cases = " + str(self.roi_fpaths_tr))

        logPrint("~~ Sampling (train) ~~")
        logPrint("Type of Sampling = " + str(self.sampling_type_inst_tr.get_type_as_str()) +
                 " (" + str(self.sampling_type_inst_tr.get_type_as_int()) + ")")
        logPrint("Sampling Categories = " + str(self.sampling_type_inst_tr.get_sampling_cats_as_str()))
        logPrint("Percent of Samples to extract per Sampling Category = " +
                 str(self.sampling_type_inst_tr.get_perc_to_sample_per_cat()))
        logPrint("Paths to weight-Maps for sampling of each category = " +
                 str(self.paths_to_wmaps_per_sampl_cat_per_subj_train))

        logPrint("~~Training Cycle~~")
        logPrint("Number of Epochs = " + str(self.n_epochs))
        logPrint("Number of Subepochs per epoch = " + str(self.n_subepochs))
        logPrint("Number of cases to load per Subepoch (for extracting the samples for this subepoch) = " +
                 str(self.max_n_cases_per_subep_train))
        logPrint("Number of Segments loaded per subepoch for Training = " + str(self.n_samples_per_subep_train) +
                 ". NOTE: This number of segments divided by the batch-size defines the number of "
                 "optimization-iterations that will be performed every subepoch!")
        logPrint("Batch size (train) = " + str(self.batchsize_train))
        logPrint("Number of parallel processes for sampling = " + str(self.num_parallel_proc_sampling))

        logPrint("~~Learning Rate Schedule~~")
        logPrint("Type of schedule = " + str(self.lr_sched_params['type']))
        logPrint("[Predef] Predefined schedule of epochs when the LR will be lowered = " +
                 str(self.lr_sched_params['predef']['epochs']))
        logPrint("[Predef] When decreasing Learning Rate, divide LR by = " +
                 str(self.lr_sched_params['predef']['div_lr_by']))
        logPrint("[Poly] Initial epochs to wait before lowering LR = " +
                 str(self.lr_sched_params['poly']['epochs_wait_before_decr']))
        logPrint("[Poly] Final epoch for the schedule = " + str(self.lr_sched_params['poly']['final_ep_for_sch']))
        logPrint("[Auto] Initial epochs to wait before lowering LR = " +
                 str(self.lr_sched_params['auto']['epochs_wait_before_decr']))
        logPrint("[Auto] When decreasing Learning Rate, divide LR by = " +
                 str(self.lr_sched_params['auto']['div_lr_by']))
        logPrint("[Auto] Minimum increase in validation accuracy (0. to 1.) that resets the waiting counter = " +
                 str(self.lr_sched_params['auto']['min_incr_of_val_acc_considered']))
        logPrint("[Expon] (Deprecated) parameters = " + str(self.lr_sched_params['expon']))

        logPrint("~~Data Augmentation During Training~~")
        logPrint("Image level augmentation:")
        logPrint("Parameters for image-level augmentation: " + str(self.augm_img_prms_tr))
        if self.augm_img_prms_tr is not None:
            logPrint("\t affine: " + str(self.augm_img_prms_tr['affine']))
        logPrint("Patch level augmentation:")
        logPrint("Mu and std for shift and scale of histograms = " + str(self.augm_sample_prms_tr['hist_dist']))
        logPrint("Probabilities of reflecting each axis = " + str(self.augm_sample_prms_tr['reflect']))
        logPrint("Probabilities of rotating planes 0/90/180/270 degrees = " + str(self.augm_sample_prms_tr['rotate90']))

        logPrint("~~~~~~~~~~~~~~~~~~Validation parameters~~~~~~~~~~~~~~~~")
        logPrint("Perform Validation on Samples throughout training? = " + str(self.val_on_samples_during_train))
        logPrint("Perform Full Inference on validation cases every few epochs? = " + str(self.val_on_whole_volumes))
        logPrint("Dataframe (csv) filename = " + str(self.csv_fname_val))
        logPrint("Filepaths to Channels of Validation Cases = " + str(self.channels_fpaths_val))
        logPrint("Filepaths to Ground-Truth labels of the Validation Cases = " + str(self.gt_fpaths_val))
        logPrint("Filepaths to ROI masks for Validation Cases = " + str(self.roi_fpaths_val))

        logPrint("~~~~~~~Validation on Samples throughout Training~~~~~~~")
        logPrint("Number of Segments loaded per subepoch for Validation = " + str(self.n_samples_per_subep_val))
        logPrint("Batch size (val on samples) = " + str(self.batchsize_val_samples))

        logPrint("~~ Sampling (val) ~~")
        logPrint("Type of Sampling = " + str(self.sampling_type_inst_val.get_type_as_str()) + " (" +
                 str(self.sampling_type_inst_val.get_type_as_int()) + ")")
        logPrint("Sampling Categories = " + str(self.sampling_type_inst_val.get_sampling_cats_as_str()))
        logPrint("Percent of Samples to extract per Sampling Category = " +
                 str(self.sampling_type_inst_val.get_perc_to_sample_per_cat()))
        logPrint("Paths to weight-maps for sampling of each category = " +
                 str(self.paths_to_wmaps_per_sampl_cat_per_subj_val))

        logPrint("~~~~~Validation with Full Inference on Validation Cases~~~~~")
        logPrint("Perform Full-Inference on Val. cases every that many epochs = " +
                 str(self.num_epochs_between_val_on_whole_volumes))
        logPrint("Batch size (val on whole volumes) = " + str(self.batchsize_val_whole))
        logPrint("~~Predictions (segmentations and prob maps on val. cases)~~")
        logPrint("Save Segmentations = " + str(self.save_segms_val))
        logPrint("Save Probability Maps for each class = " + str(self.save_probs_per_cl_val))
        logPrint("Filepaths to save results per case = " + str(self.out_preds_fpaths_val))
        logPrint("Suffixes with which to save segmentations and probability maps = " +
                 str(self.suffixes_for_outp_val))
        logPrint("~~Feature Maps~~")
        logPrint("Save Feature Maps = " + str(self.save_fms_flag_val))
        logPrint("Min/Max Indices of FMs to visualise per pathway-type and per layer = " +
                 str(self.inds_fms_per_pathtype_per_layer_to_save))
        logPrint("Filepaths to save FMs per case = " + str(self.out_fms_fpaths_val))

        logPrint("~~Optimization~~")
        logPrint("Initial Learning rate = " + str(self.learningRate))
        logPrint("Optimizer to use: SGD(0), Adam(1), RmsProp(2) = " + str(self.optimizerSgd0Adam1Rms2))
        logPrint(
            "Parameters for Adam: b1= " + str(self.b1Adam) + ", b2=" + str(self.b2Adam) + ", e= " + str(self.eAdam))
        logPrint("Parameters for RmsProp: rho= " + str(self.rhoRms) + ", e= " + str(self.eRms))
        logPrint("Momentum Type: Classic (0) or Nesterov (1) = " + str(self.classicMom0Nesterov1))
        logPrint("Momentum Non-Normalized (0) or Normalized (1) = " + str(self.momNonNormalized0Normalized1))
        logPrint("Momentum Value = " + str(self.momentumValue))
        logPrint("~~Costs~~")
        logPrint("Loss functions and their weights = " + str(self.losses_and_weights))
        logPrint("Reweight samples in cost on a per-class basis = " + str(self.reweight_classes_in_cost))
        logPrint("L1 Regularization term = " + str(self.L1_reg_weight))
        logPrint("L2 Regularization term = " + str(self.L2_reg_weight))
        logPrint("~~Freeze Weights of Certain Layers~~")
        logPrint("Indices of layers from each type of pathway that will be kept fixed (first layer is 0):")
        logPrint("Normal pathway's layers to freeze = " + str(self.inds_layers_per_pathtype_freeze[0]))
        logPrint("Subsampled pathway's layers to freeze = " + str(self.inds_layers_per_pathtype_freeze[1]))
        logPrint("FC pathway's layers to freeze = " + str(self.inds_layers_per_pathtype_freeze[2]))

        logPrint("~~~~~~~~~~~~~~~~~~ PRE-PROCESSING ~~~~~~~~~~~~~~~~")
        logPrint("~~Data Compabitibility Checks~~")
        logPrint("Check whether input data has correct format (can slow down process) = " + str(self.run_input_checks))
        logPrint("~~Padding~~")
        logPrint("Pad Input Images = " + str(self.pad_input))
        logPrint("~~Intensity Normalization~~")
        logPrint("Verbosity level = " + str(self.norm_prms['verbose_lvl']))
        logPrint("Z-Score parameters = " + str(self.norm_prms['zscore']))

        logPrint("========== Done with printing session's parameters ==========")
        logPrint("=============================================================\n")

    def get_args_for_train_routine(self):

        args = [self.log,
                self.filepath_to_save_models,

                self.val_on_samples_during_train,
                {"segm": self.save_segms_val, "prob": self.save_probs_per_cl_val},

                self.out_preds_fpaths_val,
                self.suffixes_for_outp_val,

                self.channels_fpaths_tr,
                self.channels_fpaths_val,

                self.gt_fpaths_tr,
                self.gt_fpaths_val,

                self.paths_to_wmaps_per_sampl_cat_per_subj_train,
                self.paths_to_wmaps_per_sampl_cat_per_subj_val,

                self.roi_fpaths_tr,
                self.roi_fpaths_val,

                self.n_epochs,
                self.n_subepochs,
                self.max_n_cases_per_subep_train,
                self.n_samples_per_subep_train,
                self.n_samples_per_subep_val,
                self.num_parallel_proc_sampling,

                # -------Sampling Type---------
                self.sampling_type_inst_tr,
                self.sampling_type_inst_val,
                self.batchsize_train,
                self.batchsize_val_samples,
                self.batchsize_val_whole,
                
                # -------Data Augmentation-------
                self.augm_img_prms_tr,
                self.augm_sample_prms_tr,

                # --- Validation on whole volumes ---
                self.val_on_whole_volumes,
                self.num_epochs_between_val_on_whole_volumes,

                # --------For FM visualisation---------
                self.save_fms_flag_val,
                self.inds_fms_per_pathtype_per_layer_to_save,
                self.out_fms_fpaths_val,

                # --- Data compatibility checks ---
                self.run_input_checks,
                
                # -------- Pre-Processing ------
                self.pad_input,
                self.norm_prms
                ]
        return args

    def get_args_for_trainer(self):
        args = [self.log,
                self.inds_layers_per_pathtype_freeze,
                self.losses_and_weights,
                # Regularisation
                self.L1_reg_weight,
                self.L2_reg_weight,
                # Cost Schedules
                # Weighting Classes differently in the CNN's cost function during training:
                self.reweight_classes_in_cost
                ]
        return args

    def get_args_for_optimizer(self):
        args = [self.log,
                self.optimizerSgd0Adam1Rms2,

                self.lr_sched_params,

                self.learningRate,
                self.momentumValue,
                self.classicMom0Nesterov1,
                self.momNonNormalized0Normalized1,
                self.b1Adam,
                self.b2Adam,
                self.eAdam,
                self.rhoRms,
                self.eRms
                ]
        return args
