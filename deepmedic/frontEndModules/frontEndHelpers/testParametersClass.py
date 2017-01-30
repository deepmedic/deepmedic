# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

class TestSessionParameters(object) :
    #To be called from outside too.
    @staticmethod
    def getDefaultSessionName() :
        return "testSession"
    
    def __init__(self,
                 
                sessionName,
                sessionLogger,
                mainOutputAbsFolder,
                cnn3dInstance,
                cnnModelFilepath,
                
                #Input:
                listWithAListPerCaseWithFilepathPerChannel,
                gtLabelsFilepaths,
                roiMasksFilepaths,
                
                #Output
                namesToSavePredictionsAndFeatures,
                #predictions
                saveSegmentation,
                saveProbMapsBoolPerClass,
                folderForPredictions,
                
                #features:
                saveIndividualFmImages,
                saveMultidimensionalImageWithAllFms,
                indicesOfFmsToVisualisePerPathwayAndLayer,
                folderForFeatures,
                
                padInputImagesBool,
                
                ):
        #Importants for running session.
        self.sessionName = sessionName if sessionName else self.getDefaultSessionName()
        self.sessionLogger = sessionLogger
        self.mainOutputAbsFolder = mainOutputAbsFolder
        
        self.cnn3dInstance = cnn3dInstance #Must be filled from outside after initialization
        self.cnnModelFilepath = cnnModelFilepath
        
        #Input:
        self.channelsFilepaths = listWithAListPerCaseWithFilepathPerChannel #[[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
        self.providedGt = True if gtLabelsFilepaths <> None else False
        self.gtLabelsFilepaths = gtLabelsFilepaths if gtLabelsFilepaths <> None else []
        self.providedRoiMasks = True if roiMasksFilepaths <> None else False
        self.roiMasksFilepaths = roiMasksFilepaths if roiMasksFilepaths <> None else []
        
        #Output:
        self.namesToSavePredictionsAndFeatures = namesToSavePredictionsAndFeatures #Required. Given by the config file, and is then used to fill filepathsToSavePredictionsForEachPatient and filepathsToSaveFeaturesForEachPatient.
        #predictions
        self.saveSegmentation = saveSegmentation if saveSegmentation <> None else True
        self.saveProbMapsBoolPerClass = saveProbMapsBoolPerClass if (saveProbMapsBoolPerClass<>None and saveProbMapsBoolPerClass<>[]) else [True]*cnn3dInstance.numberOfOutputClasses
        self.filepathsToSavePredictionsForEachPatient = None #Filled by call to self.makeFilepathsForPredictionsAndFeatures()
        #features:
        self.saveIndividualFmImages = saveIndividualFmImages if saveIndividualFmImages <> None else False
        self.saveMultidimensionalImageWithAllFms = saveMultidimensionalImageWithAllFms if saveMultidimensionalImageWithAllFms <> None else False
        self.indicesOfFmsToVisualisePerPathwayAndLayer = [item if item <> None else [] for item in indicesOfFmsToVisualisePerPathwayAndLayer]
        self.indicesOfFmsToVisualisePerPathwayAndLayer.append([]) #for the Zoomed-in pathway. HIDDEN
        self.filepathsToSaveFeaturesForEachPatient = None #Filled by call to self.makeFilepathsForPredictionsAndFeatures()
        
        #Preprocessing
        self.padInputImagesBool = padInputImagesBool if padInputImagesBool <> None else True
        
        #Others useful internally or for reporting:
        self.numberOfCases = len(self.channelsFilepaths)
        self.numberOfClasses = cnn3dInstance.numberOfOutputClasses
        self.numberOfChannels = cnn3dInstance.numberOfImageChannelsPath1
        
        #HIDDENS, no config allowed for these at the moment:
        self.useSameSubChannelsAsSingleScale = True
        self.subsampledChannelsFilepaths = "placeholder" #List of Lists with filepaths per patient. Only used when above is False.
        self.smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage = [None, None]
        
        self._makeFilepathsForPredictionsAndFeatures( folderForPredictions, folderForFeatures )
        
    def printParametersOfThisSession(self) :
        logPrint = self.sessionLogger.print3
        logPrint("=============================================================")
        logPrint("================= PARAMETERS OF THIS SESSION ================")
        logPrint("=============================================================")
        logPrint("sessionName = " + str(self.sessionName))
        logPrint("~~~~~~~~~~~~~~~~~~~~INPUT~~~~~~~~~~~~~~~~")
        logPrint("Number of Classes (from CNN Model) = " + str(self.numberOfClasses))
        logPrint("Number of cases to perform inference on = " + str(self.numberOfCases))
        logPrint("number of channels (modalities) = " + str(self.numberOfChannels))
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
        if not (self.saveSegmentation or self.saveProbMapsBoolPerClass) :
            logPrint(">>> WARN: Segmentation and Probability Maps won't be saved. I guess you only wanted the feature maps?")
            
        logPrint("~~~~~~~Ouput-parameters for Feature Maps (FMs)~~~~~~")
        logPrint("Save FMs in individual images = " + str(self.saveIndividualFmImages))
        logPrint("Save all requested FMs in one 4D image = " + str(self.saveMultidimensionalImageWithAllFms))
        if self.saveMultidimensionalImageWithAllFms :
            logPrint(">>> WARN : The 4D image can be hundreds of MBytes if the CNN is big and many FMs are chosen to be saved. Configure wisely.")
        logPrint("Indices of min/max FMs to save, per type of pathway (normal/subsampled/FC) and per layer = " + str(self.indicesOfFmsToVisualisePerPathwayAndLayer[:-1]))
        logPrint("Save Feature Maps at = " + str(self.filepathsToSaveFeaturesForEachPatient))
        
        logPrint("~~~~~~~ Parameters for Preprocessing ~~~~~~")
        logPrint("Pad Input Images = " + str(self.padInputImagesBool))
        if not self.padInputImagesBool :
            logPrint(">>> WARN: Inference near the borders of the image might be incomplete if not padded! Although some speed is gained if not padded. Task-specific, your choice.")
        logPrint("========== Done with printing session's parameters ==========")
        logPrint("=============================================================")
        
    def getTupleForCnnTesting(self) :
        borrowFlag = True
        
        validation0orTesting1 = 1
        
        testTuple = (self.sessionLogger,
            validation0orTesting1,
            [self.saveSegmentation, self.saveProbMapsBoolPerClass],
            self.cnn3dInstance,
            
            self.channelsFilepaths,
            
            self.providedGt,
            self.gtLabelsFilepaths,
            
            self.providedRoiMasks,
            self.roiMasksFilepaths,
            
            borrowFlag,
            self.filepathsToSavePredictionsForEachPatient,
            
            #----Preprocessing------
            self.padInputImagesBool,
            self.smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
            
            #for extension
            self.useSameSubChannelsAsSingleScale,
            self.subsampledChannelsFilepaths,
            
            #--------For FM visualisation---------
            self.saveIndividualFmImages,
            self.saveMultidimensionalImageWithAllFms,
            self.indicesOfFmsToVisualisePerPathwayAndLayer,
            self.filepathsToSaveFeaturesForEachPatient
            )
        
        return testTuple
    
    def _makeFilepathsForPredictionsAndFeatures(self,
                                                absPathToFolderForPredictionsFromSession,
                                                absPathToFolderForFeaturesFromSession
                                                ) :
        self.filepathsToSavePredictionsForEachPatient = []
        self.filepathsToSaveFeaturesForEachPatient = []
        
        if self.namesToSavePredictionsAndFeatures <> None :
            for case_i in xrange(self.numberOfCases) :
                filepathForCasePrediction = absPathToFolderForPredictionsFromSession + "/" + self.namesToSavePredictionsAndFeatures[case_i]
                self.filepathsToSavePredictionsForEachPatient.append( filepathForCasePrediction )
                filepathForCaseFeatures = absPathToFolderForFeaturesFromSession + "/" + self.namesToSavePredictionsAndFeatures[case_i]
                self.filepathsToSaveFeaturesForEachPatient.append( filepathForCaseFeatures )
