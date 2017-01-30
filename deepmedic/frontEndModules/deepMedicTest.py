# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import os

from deepmedic import myLoggerModule

from deepmedic.trainValidateTestVisualiseParallel import performInferenceForTestingOnWholeVolumes

from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import getAbsPathEvenIfRelativeIsGiven
from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import checkIfAllElementsOfAListAreFilesAndExitIfNot
from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import parseAbsFileLinesInList
from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import parseFileLinesInList
from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import checkListContainsCorrectNumberOfCasesOtherwiseExitWithError
from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import checkThatAllEntriesOfAListFollowNameConventions

from deepmedic.frontEndModules.frontEndHelpers.testParametersClass import TestSessionParameters
from deepmedic.frontEndModules.frontEndHelpers.preparationForSessionHelpers import makeFoldersNeededForTestingSession

from deepmedic.genericHelpers import load_object_from_gzip_file

class TestConfig(object):
    configStruct = {} #In here will be placed all read arguments.
    def get(self,string1) :
        return self.configStruct[string1] if string1 in self.configStruct else None
    
    #Optional but highly suggested.
    SESSION_NAME = "sessionName"
    #[REQUIRED]
    FOLDER_FOR_OUTPUT = "folderForOutput" #MUST BE GIVEN
    CNN_MODEL_FILEPATH = "cnnModelFilePath" #MUST BE GIVEN
    CHANNELS = "channels" #MUST BE GIVEN
    
    NAMES_FOR_PRED_PER_CASE = "namesForPredictionsPerCase"
    
    
    #[OPTIONALS]
    PAD_INPUT = "padInputImagesBool"
    
    ROI_MASKS = "roiMasks"
    
    GT_LABELS = "gtLabels"
    
    SAVE_SEGM = "saveSegmentation" # Default True
    SAVE_PROBMAPS_PER_CLASS = "saveProbMapsForEachClass" # Default True
    
    #optionals, cause default is False.
    SAVE_INDIV_FMS = "saveIndividualFms"
    SAVE_4DIM_FMS = "saveAllFmsIn4DimImage"
    
    INDICES_OF_FMS_TO_SAVE_NORMAL = "minMaxIndicesOfFmsToSaveFromEachLayerOfNormalPathway"
    INDICES_OF_FMS_TO_SAVE_SUBSAMPLED = "minMaxIndicesOfFmsToSaveFromEachLayerOfSubsampledPathway"
    INDICES_OF_FMS_TO_SAVE_FC = "minMaxIndicesOfFmsToSaveFromEachLayerOfFullyConnectedPathway"
    
    def checkIfConfigIsCorrectForParticularCnnModel(self, cnnInstance) :
        print "Checking if configuration is correct in relation to the loaded model (correct number of input channels, number of classes, etc) ..."
        #Check whether the given channels are as many as the channels when the model was built.
        if len(self.configStruct[self.CHANNELS]) <> cnnInstance.numberOfImageChannelsPath1 :
            print "ERROR:\tConfiguration parameter \"", self.configStruct[self.CHANNELS], "\" should have the same number of elements as the number of channels specified when constructing the cnnModel!\n\tCnnModel was constructed to take as input #", cnnInstance.numberOfImageChannelsPath1, " while the list given in the config-file contained #", len(self.configStruct[self.CHANNELS]), " elements."
            print "ERROR:\tPlease provide a list of files that contain the paths to each case's image-channels.\n\tThis parameter should be given in the format : ", testConst.CHANNELS, " = [\"path-to-file-with-paths-for-channel1-of-each-case\", ... , \"path-to-file-with-paths-for-channelN-of-each-case\"] in the configuration file.\n\tExiting!"; exit(1)
        #NOTE: Currently not checking the subsampled path, cause user is only allowed to use the same channels as the normal pathway. But should be possible in the future.
        #cnnInstance.numberOfImageChannelsPath2
        usingSubsampledWaypath = len(cnnInstance.cnnLayersSubsampled)>0
        
        #Check whether the boolean list that saves whether to save the prob-maps has same number of elements as the classes the model has.
        if self.configStruct[self.SAVE_PROBMAPS_PER_CLASS] and len(self.configStruct[self.SAVE_PROBMAPS_PER_CLASS]) <> cnnInstance.numberOfOutputClasses :
            print "ERROR:\tConfiguration parameter \"", self.configStruct[self.SAVE_PROBMAPS_PER_CLASS], "\" should have the same number of elements as the number of classes in the task! CnnModel was constructed to predict #", cnnInstance.numberOfOutputClasses, " while the list for the parameter contained #", len(self.configStruct[self.SAVE_PROBMAPS_PER_CLASS]), " elements."
            print "ERROR:\tConfiguration parameter \"", self.configStruct[self.SAVE_PROBMAPS_PER_CLASS], "\" should be a list of booleans, one for each class of the task (including the background as class-0). True to save the predicted probability map for the corresponding class, False otherwise.\n\tAs an example, it should be given in the form: ", self.configStruct[self.SAVE_PROBMAPS_PER_CLASS], " = [False, True, False] (python style boolean list).\n\tThis would result in not saving the prob-maps for the class-0 (background), save for class-1, and not save for class-2. Please correct it or ommit it completely for default.\nExiting!"; exit(1)
            
            
        #Check that the lists that say which featureMaps to save in each layer have the correct amount of entries, same as each pathway's layers.
        savingFms = self.configStruct[self.SAVE_INDIV_FMS] or self.configStruct[self.SAVE_4DIM_FMS]
        numNormLayers = len(cnnInstance.cnnLayers); numSubsLayers = len(cnnInstance.cnnLayersSubsampled); numFcLayers = len(cnnInstance.fcLayers)
        numLayerEntriesGivenNorm =  None if not self.configStruct[self.INDICES_OF_FMS_TO_SAVE_NORMAL] else len(self.configStruct[self.INDICES_OF_FMS_TO_SAVE_NORMAL])
        numLayerEntriesGivenSubs =  None if not self.configStruct[self.INDICES_OF_FMS_TO_SAVE_SUBSAMPLED] else len(self.configStruct[self.INDICES_OF_FMS_TO_SAVE_SUBSAMPLED])
        numLayerEntriesGivenFc =  None if not self.configStruct[self.INDICES_OF_FMS_TO_SAVE_FC] else len(self.configStruct[self.INDICES_OF_FMS_TO_SAVE_FC])
        
        wrongFmsEntryForNormal = True if (savingFms and numLayerEntriesGivenNorm <> None and numLayerEntriesGivenNorm <> numNormLayers) else False
        wrongFmsEntryForSubsampled = True if (savingFms and numLayerEntriesGivenSubs <> None and numLayerEntriesGivenSubs <> numSubsLayers) else False
        wrongFmsEntryForFc = True if (savingFms and numLayerEntriesGivenFc <> None and numLayerEntriesGivenFc <> numFcLayers) else False
        wrongEntryGiven = wrongFmsEntryForNormal or wrongFmsEntryForSubsampled or wrongFmsEntryForFc
        if wrongEntryGiven :
            print "ERROR:\tIn order to save the feature maps during inference, the min and max indeces of the feature maps wanted need to be specified. A pair of (min,max) must be specified for each layer, for each pathway of the CNN (normal/subsampled(if used), fully-connected). The given lists for each of the pathway-types need to have the same number of elements as the number of layers in that pathway. They were found inconsistent in comparison to the number of layers in the loaded CNN model:"
        if wrongFmsEntryForNormal :
            print "\tNormal pathway: In config file: ", self.configStruct[self.INDICES_OF_FMS_TO_SAVE_NORMAL], " given #", numLayerEntriesGivenNorm, " elements, while the CNN-model has #", numNormLayers, " layers in this pathway."
        if wrongFmsEntryForSubsampled :
            print "\tSubsampled pathway: In config file: ", self.configStruct[self.INDICES_OF_FMS_TO_SAVE_SUBSAMPLED], " given #", numLayerEntriesGivenSubs, " elements, while the CNN-model has #", numSubsLayers, " layers in this pathway."
        if wrongFmsEntryForFc :
            print "\tFully-Connected pathway: In config file: ", self.configStruct[self.INDICES_OF_FMS_TO_SAVE_FC], " given #", numLayerEntriesGivenFc, " elements, while the CNN-model has #", numFcLayers, " layers in this pathway."
        if wrongEntryGiven :
            print "\tExiting!"; exit(1)
            
        print "The given configuration looks correct in comparison to the CNN-model's parameters. Very nice! We are almost ready for inference!"
        
#Checks whether the main test-config has the REQUIRED parameters.
def checkIfMainTestConfigIsCorrect(testConfig, testConfigFilepath, absPathToSavedModelFromCmdLine) :
    configStruct = testConfig.configStruct
    print "Checking whether the given test-configuration file is correct..."
    if not configStruct[testConfig.FOLDER_FOR_OUTPUT] or not isinstance(configStruct[testConfig.FOLDER_FOR_OUTPUT], str) :
        print "ERROR:\tPlease provide the path to an output folder for this session, by providing an entry: ", testConfig.FOLDER_FOR_OUTPUT, " = \"path-to-folder\" in the configuration file (python-like-string).\n\tExiting!"
        exit(1)
    if not (configStruct[testConfig.CNN_MODEL_FILEPATH] and isinstance(configStruct[testConfig.CNN_MODEL_FILEPATH], str) ) and not absPathToSavedModelFromCmdLine :
        print "ERROR:\tPlease provide the path to an existing and trained cnn model folder for this session, by providing an entry: ", testConfig.CNN_MODEL_FILEPATH, " = \"path-to-saved-model\" in the configuration file, or using the corresponding option of the main deepmedic script.\n\tExiting!"
        exit(1)
    if not configStruct[testConfig.CHANNELS] or not isinstance(configStruct[testConfig.CHANNELS], list) or not all(isinstance(item, str) for item in configStruct[testConfig.CHANNELS]) :
        print "ERROR:\tPlease provide a list of files that contain the paths to each case's image-channels.\n\tThis parameter should be given in the format : ", testConfig.CHANNELS, " = [\"path-to-file-with-paths-for-channel1-of-each-case\", ... , \"path-to-file-with-paths-for-channelN-of-each-case\"] in the configuration file (list of strings, python style).\n\tExiting!"
        exit(1)
    if not configStruct[testConfig.NAMES_FOR_PRED_PER_CASE] or not isinstance(configStruct[testConfig.NAMES_FOR_PRED_PER_CASE], str) :
        print "ERROR:\tPlease provide the path to a file, which contains a name for each case, in order to name the inference's output correspondingly.\n\tThis parameter should be given in the format : ", testConfig.NAMES_FOR_PRED_PER_CASE, " = \"path-to-file-with-paths-for-channel1-for-each-patient\" in the configuration file.\n\tExiting!"
        exit(1)
        
    #absPathToSavedModelFromCmdLine should be absolute path already!
    absPathToCnnModelGiven = absPathToSavedModelFromCmdLine if absPathToSavedModelFromCmdLine else getAbsPathEvenIfRelativeIsGiven(configStruct[testConfig.CNN_MODEL_FILEPATH], testConfigFilepath)
    if not os.path.isfile(absPathToCnnModelGiven) :
        print "ERROR: Specified path to a cnn-model (", absPathToCnnModelGiven, ") does not point to an existing file! Exiting!"
        exit(1)
        
    print "Test-configuration file seems correctly completed at first check. I hope this was not too complicated!"
    
    
#Checks whether testing-config's listing-files are correct (channels, masks etc)
def checkIfFilesThatListFilesPerCaseAreCorrect(testConfig, testConfigFilepath) :
    configStruct = testConfig.configStruct
    print "Checking whether the given files that list channels, masks, etc per case are correctly filled..."
    numberOfChannels = len(configStruct[testConfig.CHANNELS])
    numberOfCases = -1
    print "Number of given files that list the channels per case were: ", numberOfChannels
    
    listOfListingFilesProvided = configStruct[testConfig.CHANNELS]
    listOfListingFilesProvided = listOfListingFilesProvided if not configStruct[testConfig.ROI_MASKS] else listOfListingFilesProvided + [configStruct[testConfig.ROI_MASKS]]
    listOfListingFilesProvided = listOfListingFilesProvided if not configStruct[testConfig.GT_LABELS] else listOfListingFilesProvided + [configStruct[testConfig.GT_LABELS]]
    for pathToListingFile_i in xrange(len(listOfListingFilesProvided)) :
        pathToCurrentFileListingPaths = listOfListingFilesProvided[pathToListingFile_i]
        absolutePathToCurrentFileListingPaths = getAbsPathEvenIfRelativeIsGiven(pathToCurrentFileListingPaths, testConfigFilepath)
        if not os.path.isfile(absolutePathToCurrentFileListingPaths) :
            print "ERROR: path provided does not correspond to a file:", absolutePathToCurrentFileListingPaths
            print "Exiting!"
            exit(1)
        listOfFilepathsForEachCaseInCurrentListingFile = parseAbsFileLinesInList(absolutePathToCurrentFileListingPaths)
        
        if numberOfCases == -1 :
            numberOfCases = len(listOfFilepathsForEachCaseInCurrentListingFile)
        else :
            checkListContainsCorrectNumberOfCasesOtherwiseExitWithError(numberOfCases, absolutePathToCurrentFileListingPaths, listOfFilepathsForEachCaseInCurrentListingFile)
            
        checkIfAllElementsOfAListAreFilesAndExitIfNot(absolutePathToCurrentFileListingPaths, listOfFilepathsForEachCaseInCurrentListingFile)
        
    #Check the list of names to give to predictions, cause it is a different case...
    listingFileWithNamesForPredictionsPerCase = configStruct[testConfig.NAMES_FOR_PRED_PER_CASE]
    absolutePathToListingFileWithPredictionNamesPerCase = getAbsPathEvenIfRelativeIsGiven(listingFileWithNamesForPredictionsPerCase, testConfigFilepath)
    if not os.path.isfile(absolutePathToListingFileWithPredictionNamesPerCase) :
        print "ERROR: path provided does not correspond to a file:", absolutePathToListingFileWithPredictionNamesPerCase
        print "Exiting!"; exit(1)
    listOfPredictionNamesForEachCaseInListingFile = parseFileLinesInList(absolutePathToListingFileWithPredictionNamesPerCase) #CAREFUL: Here we use a different parsing function!
    checkThatAllEntriesOfAListFollowNameConventions(listOfPredictionNamesForEachCaseInListingFile)
    print "Files that list the channels for each case seem fine. Thanks."
    
def checkIfOptionalParametersAreGivenCorrectly(testConfig, testConfigFilepath) :
    print "Checking optional parameters..."
    
    configStruct = testConfig.configStruct
    
    if configStruct[testConfig.SESSION_NAME] and not isinstance(configStruct[testConfig.SESSION_NAME], str) :
        print "ERROR: Optional parameter ", configStruct[testConfig.SESSION_NAME], " should be a string, given in the format: ", configStruct[testConfig.SESSION_NAME]," = \"yourTestSessionName\".\nExiting!"; exit(1)
        
    listOfBooleanParametersStrings = [testConfig.PAD_INPUT, testConfig.SAVE_SEGM, testConfig.SAVE_INDIV_FMS, testConfig.SAVE_4DIM_FMS, ]
    for booleanParameterString in listOfBooleanParametersStrings :
        if (configStruct[booleanParameterString]) and not (configStruct[booleanParameterString] in [True,False]) :
            print "ERROR: Optional parameter ", configStruct[booleanParameterString], " should be given either True or False in the format: ", configStruct[booleanParameterString]," = True/False (No quotes, first letter capital. Python-Boolean style!).\nExiting!"; exit(1)
            
    #ROI and GT masks already checked.
    
    # SAVE_PROBMAPS_PER_CLASS cannot be fully checked. Need to load the model first and then make sure it has as many entries as it should!
    variableSaveProbMaps = configStruct[testConfig.SAVE_PROBMAPS_PER_CLASS]
    if (variableSaveProbMaps) :
        valid1 = True
        if isinstance(variableSaveProbMaps, list) and len(variableSaveProbMaps) > 0:
            for entry in variableSaveProbMaps :
                if not entry in [True, False] :
                    valid1 = False
        else :
            valid1 = False
        if not valid1 :
            print "ERROR: Configuration parameter \"", testConfig.SAVE_PROBMAPS_PER_CLASS, "\" should be a list of booleans, one for each class of the task (including the background). True to save the predicted probability map for the corresponding class, False otherwise. As an example, it should be given in the form: ", testConfig.SAVE_PROBMAPS_PER_CLASS, " = [False, True, False] (python style boolean list). This would result in not saving the prob-maps for the class-0 (background), save for class-1, and not save for class-2. Please correct it or ommit it completely for default.\nExiting!"
            exit(1)
            
    #INDICES_OF_FMS_TO_SAVE
    listOfTheIndicesOfFmsPerPathwayType = [testConfig.INDICES_OF_FMS_TO_SAVE_NORMAL, testConfig.INDICES_OF_FMS_TO_SAVE_SUBSAMPLED, testConfig.INDICES_OF_FMS_TO_SAVE_FC]
    for indicesOfFmsPerPathwayTypeString in listOfTheIndicesOfFmsPerPathwayType :
        if (configStruct[indicesOfFmsPerPathwayTypeString]) :
            variableIndicesOfFmsPerPathwayType = configStruct[indicesOfFmsPerPathwayTypeString]
            valid1 = True
            if not isinstance(variableIndicesOfFmsPerPathwayType, list) :
                valid1 = False
            elif len(variableIndicesOfFmsPerPathwayType) > 0: #Remember, it can be [] to save no feature maps from this pathway!
                for entryForALayer in variableIndicesOfFmsPerPathwayType :
                    if isinstance(entryForALayer, list) :
                        if len(entryForALayer) > 0 : #This can be [], to not save any FM of this layer.
                            for indexOfFm in entryForALayer :
                                if not (isinstance(indexOfFm, int) and indexOfFm >= 0 ):
                                    valid1 = False
                    else :
                        valid1 = False
            if not valid1 :
                print "ERROR: Configuration parameter \"", indicesOfFmsPerPathwayTypeString ,"\" should be given in the form: ", indicesOfFmsPerPathwayTypeString, " = [[minFmOfLayer0, maxFmOfLayer0], , ..., [minFmOfLayerN, maxFmOfLayerN]] (python style list of lists of two integers). min/maxFmOfLayerN are integers (equal/greater than 0), that are the minimum and maximum indices of the Feature Maps that I want to visualise from this particular pathway type. An entry can be given [] if I don't want to visualise any FMs from a pathway or a certain layer. Please correct it or ommit it completely for default.\nExiting!"; exit(1)
                
    print "Optional parameters seem alright at first check, although we ll need to double-check, after the cnn-model is loaded..."
    
#Both the arguments are absolute paths. The "absPathToSavedModelFromCmdLine" can be None if it was not provided in cmd line.
def deepMedicTestMain(testConfigFilepath, absPathToSavedModelFromCmdLine) :
    print "Given Test-Configuration File: ", testConfigFilepath
    #Parse the config file in this naive fashion...
    testConfig = TestConfig()
    #configStruct = testConfig.configStruct
    execfile(testConfigFilepath, testConfig.configStruct)
    configGet = testConfig.get #Main interface
    
    #Do checks.
    #checkIfMainTestConfigIsCorrect(testConfig, testConfigFilepath, absPathToSavedModelFromCmdLine) #Checks REQUIRED fields are complete.
    #checkIfFilesThatListFilesPerCaseAreCorrect(testConfig, testConfigFilepath) #Checks listing-files (whatever given).
    #checkIfOptionalParametersAreGivenCorrectly(testConfig, testConfigFilepath)
    
    #At this point it was checked that all parameters (that could be checked) and filepaths are correct, pointing to files/dirs and all files/dirs exist.
    
    #Create Folders and Logger
    mainOutputAbsFolder = getAbsPathEvenIfRelativeIsGiven(configGet(testConfig.FOLDER_FOR_OUTPUT), testConfigFilepath)
    sessionName = configGet(testConfig.SESSION_NAME) if configGet(testConfig.SESSION_NAME) else TestSessionParameters.getDefaultSessionName()
    [folderForLogs,
    folderForPredictions,
    folderForFeatures] = makeFoldersNeededForTestingSession(mainOutputAbsFolder, sessionName)
    loggerFileName = folderForLogs + "/" + sessionName + ".txt"
    sessionLogger = myLoggerModule.MyLogger(loggerFileName)
    
    sessionLogger.print3("CONFIG: The configuration file for the testing session was loaded from: " + str(testConfigFilepath))
    
    #Load the CNN Model!
    sessionLogger.print3("=========== Loading the CNN model for testing... ===============")
    #If CNN-Model was specified in command line, completely override the one in the config file.
    filepathToCnnModelToLoad = None
    if absPathToSavedModelFromCmdLine and configGet(testConfig.CNN_MODEL_FILEPATH) :
        sessionLogger.print3("WARN: A CNN-Model to use was specified both in the command line input and in the test-config-file! The input by the command line will be used: " + str(absPathToSavedModelFromCmdLine) )
        filepathToCnnModelToLoad = absPathToSavedModelFromCmdLine
    elif absPathToSavedModelFromCmdLine :
        filepathToCnnModelToLoad = absPathToSavedModelFromCmdLine
    else :
        filepathToCnnModelToLoad = getAbsPathEvenIfRelativeIsGiven(configGet(testConfig.CNN_MODEL_FILEPATH), testConfigFilepath)
    sessionLogger.print3("...Loading the network can take a few minutes if the model is big...")
    cnn3dInstance = load_object_from_gzip_file(filepathToCnnModelToLoad)
    sessionLogger.print3("The CNN model was loaded successfully from: " + str(filepathToCnnModelToLoad))
    #Do final checks of the parameters. Check the ones that need check in comparison to the model's parameters! Such as: SAVE_PROBMAPS_PER_CLASS, INDICES_OF_FMS_TO_SAVE, Number of Channels!
    #testConfig.checkIfConfigIsCorrectForParticularCnnModel(cnn3dInstance)
    
    #Fill in the session's parameters.
    #[[case1-ch1, ..., caseN-ch1], [case1-ch2,...,caseN-ch2]]
    listOfAListPerChannelWithFilepathsOfAllCases = [parseAbsFileLinesInList(getAbsPathEvenIfRelativeIsGiven(channelConfPath, testConfigFilepath)) for channelConfPath in configGet(testConfig.CHANNELS)]
    #[[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
    listWithAListPerCaseWithFilepathPerChannel = [ list(item) for item in zip(*tuple(listOfAListPerChannelWithFilepathsOfAllCases)) ]
    gtLabelsFilepaths = parseAbsFileLinesInList( getAbsPathEvenIfRelativeIsGiven(configGet(testConfig.GT_LABELS), testConfigFilepath) ) if configGet(testConfig.GT_LABELS) else None
    roiMasksFilepaths = parseAbsFileLinesInList( getAbsPathEvenIfRelativeIsGiven(configGet(testConfig.ROI_MASKS), testConfigFilepath) ) if configGet(testConfig.ROI_MASKS) else None
    namesToSavePredsAndFeats = parseFileLinesInList( getAbsPathEvenIfRelativeIsGiven(configGet(testConfig.NAMES_FOR_PRED_PER_CASE), testConfigFilepath) ) if configGet(testConfig.NAMES_FOR_PRED_PER_CASE) else None #CAREFUL: Here we use a different parsing function!
    
    testSessionParameters = TestSessionParameters(
                    sessionName = sessionName,
                    sessionLogger = sessionLogger,
                    mainOutputAbsFolder = mainOutputAbsFolder,
                    cnn3dInstance = cnn3dInstance,
                    cnnModelFilepath = filepathToCnnModelToLoad,
                    
                    #Input:
                    listWithAListPerCaseWithFilepathPerChannel = listWithAListPerCaseWithFilepathPerChannel,
                    gtLabelsFilepaths = gtLabelsFilepaths,
                    roiMasksFilepaths = roiMasksFilepaths,
                    
                    #Output                                
                    namesToSavePredictionsAndFeatures = namesToSavePredsAndFeats,
                    #predictions
                    saveSegmentation = configGet(testConfig.SAVE_SEGM),
                    saveProbMapsBoolPerClass = configGet(testConfig.SAVE_PROBMAPS_PER_CLASS),
                    folderForPredictions = folderForPredictions,
                    
                    #features:
                    saveIndividualFmImages = configGet(testConfig.SAVE_INDIV_FMS),
                    saveMultidimensionalImageWithAllFms = configGet(testConfig.SAVE_4DIM_FMS),
                    indicesOfFmsToVisualisePerPathwayAndLayer = [configGet(testConfig.INDICES_OF_FMS_TO_SAVE_NORMAL),
                                                                configGet(testConfig.INDICES_OF_FMS_TO_SAVE_SUBSAMPLED),
                                                                configGet(testConfig.INDICES_OF_FMS_TO_SAVE_FC)
                                                                ],
                    folderForFeatures = folderForFeatures,
                    
                    padInputImagesBool = configGet(testConfig.PAD_INPUT),
                    )
    
    testSessionParameters.sessionLogger.print3("===========       NEW TESTING SESSION         ===============")
    testSessionParameters.printParametersOfThisSession()
    
    testSessionParameters.sessionLogger.print3("======================================================")
    testSessionParameters.sessionLogger.print3("=========== Testing with the CNN model ===============")
    testSessionParameters.sessionLogger.print3("======================================================")
    performInferenceForTestingOnWholeVolumes(*testSessionParameters.getTupleForCnnTesting())
    testSessionParameters.sessionLogger.print3("======================================================")
    testSessionParameters.sessionLogger.print3("=========== Testing session finished =================")
    testSessionParameters.sessionLogger.print3("======================================================")
    
    