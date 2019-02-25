# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

from deepmedic.frontEnd.configParsing.utils import getAbsPathEvenIfRelativeIsGiven, parseAbsFileLinesInList, parseFileLinesInList, check_and_adjust_path_to_ckpt

class TestSessionParameters(object) :
    #To be called from outside too.
    @staticmethod
    def getSessionName(sessionName) :
        return sessionName if sessionName is not None else "testSession"
    
    def __init__(self,
                log,
                mainOutputAbsFolder,
                folderForPredictions,
                folderForFeatures,
                num_classes,
                cfg):
        #Importants for running session.
        # From Session:
        self.log = log
        self.mainOutputAbsFolder = mainOutputAbsFolder
        
        # From test config:
        self.sessionName = self.getSessionName( cfg[cfg.SESSION_NAME] )
        
        abs_path_to_cfg = cfg.get_abs_path_to_cfg()
        abs_path_to_saved = getAbsPathEvenIfRelativeIsGiven( cfg[cfg.SAVED_MODEL], abs_path_to_cfg ) if cfg[cfg.SAVED_MODEL] is not None else None # Where to load the model from.
        self.savedModelFilepath = check_and_adjust_path_to_ckpt( self.log, abs_path_to_saved) if abs_path_to_saved is not None else None
        
        #Input:
        #[[case1-ch1, ..., caseN-ch1], [case1-ch2,...,caseN-ch2]]
        listOfAListPerChannelWithFilepathsOfAllCases = [parseAbsFileLinesInList(getAbsPathEvenIfRelativeIsGiven(channelConfPath, abs_path_to_cfg)) for channelConfPath in cfg[cfg.CHANNELS]]
        self.channelsFilepaths = [ list(item) for item in zip(*tuple(listOfAListPerChannelWithFilepathsOfAllCases)) ] # [[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
        self.providedGt = True if cfg[cfg.GT_LABELS] is not None else False
        self.gtLabelsFilepaths = parseAbsFileLinesInList( getAbsPathEvenIfRelativeIsGiven(cfg[cfg.GT_LABELS], abs_path_to_cfg) ) if cfg[cfg.GT_LABELS] is not None else None
        self.providedRoiMasks = True if cfg[cfg.ROI_MASKS] is not None else False
        self.roiMasksFilepaths = parseAbsFileLinesInList( getAbsPathEvenIfRelativeIsGiven(cfg[cfg.ROI_MASKS], abs_path_to_cfg) ) if self.providedRoiMasks else None
        
        #Output:
        self.namesToSavePredictionsAndFeatures = parseFileLinesInList( getAbsPathEvenIfRelativeIsGiven(cfg[cfg.NAMES_FOR_PRED_PER_CASE], abs_path_to_cfg) ) if cfg[cfg.NAMES_FOR_PRED_PER_CASE] is not None else None #CAREFUL: different parser! #Optional. Not required if not saving results.
        #predictions
        self.saveSegmentation = cfg[cfg.SAVE_SEGM] if cfg[cfg.SAVE_SEGM] is not None else True
        self.saveProbMapsBoolPerClass = cfg[cfg.SAVE_PROBMAPS_PER_CLASS] if (cfg[cfg.SAVE_PROBMAPS_PER_CLASS] is not None and cfg[cfg.SAVE_PROBMAPS_PER_CLASS] != []) else [True]*num_classes
        self.filepathsToSavePredictionsForEachPatient = None #Filled by call to self.makeFilepathsForPredictionsAndFeatures()
        self.suffixForSegmAndProbsDict = cfg[cfg.SUFFIX_SEGM_PROB] if cfg[cfg.SUFFIX_SEGM_PROB] is not None else {"segm": "Segm", "prob": "ProbMapClass"}
        #features:
        self.saveIndividualFmImages = cfg[cfg.SAVE_INDIV_FMS] if cfg[cfg.SAVE_INDIV_FMS] is not None else False
        self.saveMultidimensionalImageWithAllFms = cfg[cfg.SAVE_4DIM_FMS] if cfg[cfg.SAVE_4DIM_FMS] is not None else False
        if self.saveIndividualFmImages == True or self.saveMultidimensionalImageWithAllFms == True:
            indices_fms_per_pathtype_per_layer_to_save = [cfg[cfg.INDICES_OF_FMS_TO_SAVE_NORMAL]] +\
                                                         [cfg[cfg.INDICES_OF_FMS_TO_SAVE_SUBSAMPLED]] +\
                                                         [cfg[cfg.INDICES_OF_FMS_TO_SAVE_FC]]
            self.indices_fms_per_pathtype_per_layer_to_save = [item if item is not None else [] for item in indices_fms_per_pathtype_per_layer_to_save]
        else:
            self.indices_fms_per_pathtype_per_layer_to_save = None
        self.filepathsToSaveFeaturesForEachPatient = None #Filled by call to self.makeFilepathsForPredictionsAndFeatures()
        
        #Preprocessing
        self.padInputImagesBool = cfg[cfg.PAD_INPUT] if cfg[cfg.PAD_INPUT] is not None else True
        
        #Others useful internally or for reporting:
        self.numberOfCases = len(self.channelsFilepaths)
        
        #HIDDENS, no config allowed for these at the moment:
        self.useSameSubChannelsAsSingleScale = True
        self.subsampledChannelsFilepaths = "placeholder" #List of Lists with filepaths per patient. Only used when above is False.
        
        self._makeFilepathsForPredictionsAndFeatures( folderForPredictions, folderForFeatures )
        
    def _makeFilepathsForPredictionsAndFeatures(self,
                                                absPathToFolderForPredictionsFromSession,
                                                absPathToFolderForFeaturesFromSession
                                                ) :
        self.filepathsToSavePredictionsForEachPatient = []
        self.filepathsToSaveFeaturesForEachPatient = []
        if self.namesToSavePredictionsAndFeatures is not None : # standard behavior
            for case_i in range(self.numberOfCases) :
                filepathForCasePrediction = absPathToFolderForPredictionsFromSession + "/" + self.namesToSavePredictionsAndFeatures[case_i]
                self.filepathsToSavePredictionsForEachPatient.append( filepathForCasePrediction )
                filepathForCaseFeatures = absPathToFolderForFeaturesFromSession + "/" + self.namesToSavePredictionsAndFeatures[case_i]
                self.filepathsToSaveFeaturesForEachPatient.append( filepathForCaseFeatures )
        else : # Names for predictions not given. Special handling...
            if self.numberOfCases > 1 : # Many cases, create corresponding namings for files.
                for case_i in range(self.numberOfCases) :
                    self.filepathsToSavePredictionsForEachPatient.append( absPathToFolderForPredictionsFromSession + "/pred_case" + str(case_i) + ".nii.gz" )
                    self.filepathsToSaveFeaturesForEachPatient.append( absPathToFolderForPredictionsFromSession + "/pred_case" + str(case_i) + ".nii.gz" )
            else : # Only one case. Just give the output prediction folder, the io.py will save output accordingly.
                self.filepathsToSavePredictionsForEachPatient.append( absPathToFolderForPredictionsFromSession )
                self.filepathsToSaveFeaturesForEachPatient.append( absPathToFolderForPredictionsFromSession )
    
    
    def get_path_to_load_model_from(self):
        return self.savedModelFilepath
    
    
    def print_params(self) :
        logPrint = self.log.print3
        logPrint("")
        logPrint("=============================================================")
        logPrint("=========== PARAMETERS OF THIS TESTING SESSION ==============")
        logPrint("=============================================================")
        logPrint("sessionName = " + str(self.sessionName))
        logPrint("Model will be loaded from save = " + str(self.savedModelFilepath))
        logPrint("~~~~~~~~~~~~~~~~~~~~INPUT~~~~~~~~~~~~~~~~")
        logPrint("Number of cases to perform inference on = " + str(self.numberOfCases))
        logPrint("Paths to the channels of each case = " + str(self.channelsFilepaths))
        logPrint("User provided Ground Truth labels for DSC calculation = " + str(self.providedGt))
        if not self.providedGt :
            logPrint(">>> WARN: The DSC accuracy will NOT be evaluated and reported!")
        logPrint("Paths to the provided GT labels per case = " + str(self.gtLabelsFilepaths))
        logPrint("User provided Region-Of-Interest Masks for faster inference = " + str(self.providedRoiMasks))
        logPrint("Filepaths of the ROI Masks provided per case = " + str(self.roiMasksFilepaths))
        if not self.providedRoiMasks :
            logPrint(">>> WARN: Inference will be performed on whole scan. Consider providing a ROI image for faster results, if possible!")
            
        logPrint("~~~~~~~~~~~~~~~~~~~OUTPUT~~~~~~~~~~~~~~~")
        logPrint("Path to the main output-folder = " + str(self.mainOutputAbsFolder))
        logPrint("Provided names to use to save results for each case = " + str(self.namesToSavePredictionsAndFeatures))
        
        logPrint("~~~~~~~Ouput-parameters for Predictions (segmentation and probability maps)~~~~")
        logPrint("Save the predicted segmentation = " + str(self.saveSegmentation))
        logPrint("Save the probability maps = " + str(self.saveProbMapsBoolPerClass))
        logPrint("Paths where to save predictions per case = " + str(self.filepathsToSavePredictionsForEachPatient))
        logPrint("Suffixes with which to save segmentations and probability maps = " + str(self.suffixForSegmAndProbsDict))
        if not (self.saveSegmentation or self.saveProbMapsBoolPerClass) :
            logPrint(">>> WARN: Segmentation and Probability Maps won't be saved. I guess you only wanted the feature maps?")
            
        logPrint("~~~~~~~Ouput-parameters for Feature Maps (FMs)~~~~~~")
        logPrint("Save FMs in individual images = " + str(self.saveIndividualFmImages))
        logPrint("Save all requested FMs in one 4D image = " + str(self.saveMultidimensionalImageWithAllFms))
        if self.saveMultidimensionalImageWithAllFms :
            logPrint(">>> WARN : The 4D image can be hundreds of MBytes if the CNN is big and many FMs are chosen to be saved. Configure wisely.")
        logPrint("Indices of min/max FMs to save, per type of pathway (normal/subsampled/FC) and per layer = " + str(self.indices_fms_per_pathtype_per_layer_to_save))
        logPrint("Save Feature Maps at = " + str(self.filepathsToSaveFeaturesForEachPatient))
        
        logPrint("~~~~~~~ Parameters for Preprocessing ~~~~~~")
        logPrint("Pad Input Images = " + str(self.padInputImagesBool))
        if not self.padInputImagesBool :
            logPrint(">>> WARN: Inference near the borders of the image might be incomplete if not padded! Although some speed is gained if not padded. Task-specific, your choice.")
        logPrint("========== Done with printing session's parameters ==========")
        logPrint("=============================================================\n")
        
    def get_args_for_testing(self) :
        
        validation0orTesting1 = 1
        
        args = [self.log,
                validation0orTesting1,
                {"segm": self.saveSegmentation, "prob": self.saveProbMapsBoolPerClass},
                
                self.channelsFilepaths,
                
                self.providedGt,
                self.gtLabelsFilepaths,
                
                self.providedRoiMasks,
                self.roiMasksFilepaths,
                
                self.filepathsToSavePredictionsForEachPatient,
                self.suffixForSegmAndProbsDict,
                
                #----Preprocessing------
                self.padInputImagesBool,
                
                #for extension
                self.useSameSubChannelsAsSingleScale,
                self.subsampledChannelsFilepaths,
                
                #--------For FM visualisation---------
                self.saveIndividualFmImages,
                self.saveMultidimensionalImageWithAllFms,
                self.indices_fms_per_pathtype_per_layer_to_save,
                self.filepathsToSaveFeaturesForEachPatient
                ]
        
        return args





