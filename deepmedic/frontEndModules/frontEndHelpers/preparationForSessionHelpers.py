# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import os
def makeFoldersToSaveFilesForThisRun():
    #Create folders for saving the prediction images:
    parentFolderForSavingPredictedImages = folderToSavePredictionImages[:folderToSavePredictionImages.find("/")] #Usually should be ./predictions"
    if not os.path.exists(parentFolderForSavingPredictedImages) :
        os.mkdir(parentFolderForSavingPredictedImages)
    if not os.path.exists(folderToSavePredictionImages) : #The inner folder, for this particular run.
        os.mkdir(folderToSavePredictionImages)
    fullFolderPathToSavePredictedImagesFromDiceEvaluationOnValidationCasesDuringTraining = folderToSavePredictionImages + "/validationDuringTraining/"
    if not os.path.exists(fullFolderPathToSavePredictedImagesFromDiceEvaluationOnValidationCasesDuringTraining) :
        os.mkdir(fullFolderPathToSavePredictedImagesFromDiceEvaluationOnValidationCasesDuringTraining)
    fullFolderPathToSavePredictedImagesFromDiceEvaluationOnTestingCasesDuringTraining = folderToSavePredictionImages + "/testingDuringTraining/"
    if not os.path.exists(fullFolderPathToSavePredictedImagesFromDiceEvaluationOnTestingCasesDuringTraining) :
        os.mkdir(fullFolderPathToSavePredictedImagesFromDiceEvaluationOnTestingCasesDuringTraining)
    fullFolderPathToSavePredictedImagesDuringTesting = folderToSavePredictionImages + "/testing/"
    if not os.path.exists(fullFolderPathToSavePredictedImagesDuringTesting) :
        os.mkdir(fullFolderPathToSavePredictedImagesDuringTesting)
        
    folderWhereToPlaceVisualisationResults = folderToSavePredictionImages + "/visualisations/"
    if not os.path.exists(folderWhereToPlaceVisualisationResults) :
        os.mkdir(folderWhereToPlaceVisualisationResults)
        
    return [fullFolderPathToSavePredictedImagesFromDiceEvaluationOnValidationCasesDuringTraining,
            fullFolderPathToSavePredictedImagesFromDiceEvaluationOnTestingCasesDuringTraining,
            fullFolderPathToSavePredictedImagesDuringTesting,
            folderWhereToPlaceVisualisationResults]
    
def createMainOutputFolder(absMainOutputFolder) :
    if not os.path.exists(absMainOutputFolder) :
        os.mkdir(absMainOutputFolder)
        print "\t>>Created main output folder: ", absMainOutputFolder
def createLogsFolder(folderForLogs) :
    if not os.path.exists(folderForLogs) :
        os.mkdir(folderForLogs)
        print "\t>>Created folder for logs: ", folderForLogs
def createFolderForPredictions(folderForPredictions) :
    if not os.path.exists(folderForPredictions) :
        os.mkdir(folderForPredictions)
        print "\t>>Created folder for predictions: ", folderForPredictions
def createFolderForSessionResults(folderForSessionResults) :
    if not os.path.exists(folderForSessionResults) :
        os.mkdir(folderForSessionResults)
        print "\t>>Created folder for session: ", folderForSessionResults
def createFolderForSegmAndProbMaps(folderForSegmAndProbMaps) :
    if not os.path.exists(folderForSegmAndProbMaps) :
        os.mkdir(folderForSegmAndProbMaps)
        print "\t>>Created folder for segmentations and probability maps: ", folderForSegmAndProbMaps
def createFolderForFeatures(folderForFeatures) :
    if not os.path.exists(folderForFeatures) :
        os.mkdir(folderForFeatures)
        print "\t>>Created folder for features: ", folderForFeatures
def makeFoldersNeededForTestingSession(absMainOutputFolder, sessionName):
    #Create folders for saving the prediction images:
    print ">>Creating necessary folders for testing-session..."
    createMainOutputFolder(absMainOutputFolder)
    
    folderForLogs = absMainOutputFolder + "/logs/"
    createLogsFolder(folderForLogs)
    
    folderForPredictions = absMainOutputFolder + "/predictions"
    createFolderForPredictions(folderForPredictions)
    
    folderForSessionResults = folderForPredictions + "/" + sessionName
    createFolderForSessionResults(folderForSessionResults)
    
    folderForSegmAndProbMaps = folderForSessionResults + "/predictions/"
    createFolderForSegmAndProbMaps(folderForSegmAndProbMaps)
    
    folderForFeatures = folderForSessionResults + "/features/"
    createFolderForFeatures(folderForFeatures)
    
    return [folderForLogs,
            folderForSegmAndProbMaps,
            folderForFeatures]

def createFolderForCnnModels(folderForCnnModels) :
    if not os.path.exists(folderForCnnModels) :
        os.mkdir(folderForCnnModels)
        print "\t>>Created folder to save cnn-models as they get trained: ", folderForCnnModels

def createFolderForSessionCnnModels(folderForSessionCnnModels) :
    if not os.path.exists(folderForSessionCnnModels) :
        os.mkdir(folderForSessionCnnModels)
        print "\t>>Created folder to save session's cnn-models as they get trained: ", folderForSessionCnnModels

def makeFoldersNeededForTrainingSession(absMainOutputFolder, sessionName):
    #Create folders for saving the prediction images:
    print ">>Creating necessary folders for testing-session..."
    createMainOutputFolder(absMainOutputFolder)
    
    folderForCnnModels = absMainOutputFolder + "/cnnModels/"
    createFolderForCnnModels(folderForCnnModels)
    
    folderForSessionCnnModels = folderForCnnModels + "/" + sessionName + "/"
    createFolderForSessionCnnModels(folderForSessionCnnModels)
    
    folderForLogs = absMainOutputFolder + "/logs/"
    createLogsFolder(folderForLogs)
    
    folderForPredictions = absMainOutputFolder + "/predictions"
    createFolderForPredictions(folderForPredictions)
    
    folderForSessionResults = folderForPredictions + "/" + sessionName
    createFolderForSessionResults(folderForSessionResults)
    
    folderForSegmAndProbMaps = folderForSessionResults + "/predictions/"
    createFolderForSegmAndProbMaps(folderForSegmAndProbMaps)
    
    folderForFeatures = folderForSessionResults + "/features/"
    createFolderForFeatures(folderForFeatures)
    
    return [folderForSessionCnnModels,
            folderForLogs,
            folderForSegmAndProbMaps,
            folderForFeatures]
    

def makeFoldersNeededForCreateModelSession(absMainOutputFolder, modelName):
    #Create folders for saving the prediction images:
    print ">>Creating necessary folders for create-model-session..."
    createMainOutputFolder(absMainOutputFolder)
    
    folderForCnnModels = absMainOutputFolder + "/cnnModels/"
    createFolderForCnnModels(folderForCnnModels)
    
    folderForLogs = absMainOutputFolder + "/logs/"
    createLogsFolder(folderForLogs)
    
    return [folderForCnnModels,
            folderForLogs]

