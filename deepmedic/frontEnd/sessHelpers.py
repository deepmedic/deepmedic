# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os


def createMainOutputFolder(absMainOutputFolder) :
    if not os.path.exists(absMainOutputFolder) :
        os.mkdir(absMainOutputFolder)
        print("\t>>Created main output folder: ", absMainOutputFolder)
def createLogsFolder(folderForLogs) :
    if not os.path.exists(folderForLogs) :
        os.mkdir(folderForLogs)
        print("\t>>Created folder for logs: ", folderForLogs)
def createFolderForPredictions(folderForPredictions) :
    if not os.path.exists(folderForPredictions) :
        os.mkdir(folderForPredictions)
        print("\t>>Created folder for predictions: ", folderForPredictions)
def createFolderForSessionResults(folderForSessionResults) :
    if not os.path.exists(folderForSessionResults) :
        os.mkdir(folderForSessionResults)
        print("\t>>Created folder for session: ", folderForSessionResults)
def createFolderForSegmAndProbMaps(folderForSegmAndProbMaps) :
    if not os.path.exists(folderForSegmAndProbMaps) :
        os.mkdir(folderForSegmAndProbMaps)
        print("\t>>Created folder for segmentations and probability maps: ", folderForSegmAndProbMaps)
def createFolderForFeatures(folderForFeatures) :
    if not os.path.exists(folderForFeatures) :
        os.mkdir(folderForFeatures)
        print("\t>>Created folder for features: ", folderForFeatures)

def makeFoldersNeededForTestingSession(absMainOutputFolder, sessionName):
    #Create folders for saving the prediction images:
    print("Creating necessary folders for testing session...")
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
        print("\t>>Created folder to save cnn-models as they get trained: ", folderForCnnModels)

def createFolderForSessionCnnModels(folderForSessionCnnModels) :
    if not os.path.exists(folderForSessionCnnModels) :
        os.mkdir(folderForSessionCnnModels)
        print("\t>>Created folder to save session's cnn-models as they get trained: ", folderForSessionCnnModels)

def makeFoldersNeededForTrainingSession(absMainOutputFolder, sessionName):
    #Create folders for saving the prediction images:
    print("Creating necessary folders for training session...")
    createMainOutputFolder(absMainOutputFolder)
    
    folderForLogs = absMainOutputFolder + "/logs/"
    createLogsFolder(folderForLogs)
        
    folderForCnnModels = absMainOutputFolder + "/saved_models/"
    createFolderForCnnModels(folderForCnnModels)
    
    folderForSessionCnnModels = folderForCnnModels + "/" + sessionName + "/"
    createFolderForSessionCnnModels(folderForSessionCnnModels)
    
    folderForPredictions = absMainOutputFolder + "/predictions"
    createFolderForPredictions(folderForPredictions)
    
    folderForSessionResults = folderForPredictions + "/" + sessionName
    createFolderForSessionResults(folderForSessionResults)
    
    folderForSegmAndProbMaps = folderForSessionResults + "/predictions/"
    createFolderForSegmAndProbMaps(folderForSegmAndProbMaps)
    
    folderForFeatures = folderForSessionResults + "/features/"
    createFolderForFeatures(folderForFeatures)
    
    return [folderForLogs,
            folderForSessionCnnModels,
            folderForSegmAndProbMaps,
            folderForFeatures]
    

def makeFoldersNeededForCreateModelSession(absMainOutputFolder, modelName):
    #Create folders for saving the prediction images:
    print("Creating necessary folders for create-new-model session...")
    createMainOutputFolder(absMainOutputFolder)
    
    folderForLogs = absMainOutputFolder + "/logs/"
    createLogsFolder(folderForLogs)
    
    folderForCnnModels = absMainOutputFolder + "/saved_models/"
    createFolderForCnnModels(folderForCnnModels)
    
    folderForSessionCnnModels = folderForCnnModels + "/" + modelName + "/"
    createFolderForSessionCnnModels(folderForSessionCnnModels)
    
    return [folderForLogs,
            folderForSessionCnnModels]


def handle_exception_tf_restore(log, exc):
    import sys, traceback
    log.print3("")
    log.print3("ERROR: DeepMedic caught exception when trying to load parameters from the given path of a previously saved model.\n"+\
               "Two reasons are very likely:\n"+\
               "a) Most probably you passed the wrong path. You need to provide the path to the Tensorflow checkpoint, as expected by Tensorflow.\n"+\
               "\t In the traceback further below, Tensorflow may report this error of type [NotFoundError].\n"+\
               "\t DeepMedic uses tensorflow checkpoints to save the models. For this, it stores different types of files for every saved timepoint.\n"+\
               "\t Those files will be by default in ./examples/output/saved_models, and of the form:\n"+\
               "\t filename.datetime.model.ckpt.data-0000-of-0001 \n"+\
               "\t filename.datetime.model.ckpt.index \n"+\
               "\t filename.datetime.model.ckpt.meta (Maybe this is missing. That's ok.) \n"+\
               "\t To load this checkpoint, you have to provide the path, OMMITING the part after the [.ckpt]. I.e., your command should look like:\n"+\
               "\t python ./deepMedicRun.py -model path/to/model/config -train path/to/train/config -load filename.datetime.model.ckpt \n"+\
               "b) You have created a network of different architecture than the one that is being loaded and Tensorflow fails to match their variables.\n"+\
               "\t If this is the case, Tensorflow may report it below as error of type [DataLossError]. \n"+\
               "\t If you did not mean to change architectures, ensure that you point to the same modelConfig.cfg as used when the saved model was made.\n"+\
               "\t If you meant to change architectures, then you will have to create your own script to load the parameters from the saved checkpoint," +\
               " where the script must describe which variables of the new model match the ones from the saved model.\n"+\
               "c) The above are \"most likely\" reasons, but others are possible."+\
               " Please read the following Tensorflow stacktrace and error report carefully, and debug accordingly...\n")
    log.print3( traceback.format_exc() )
    sys.exit(1)
    
