# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import os

from deepmedic import myLoggerModule

from deepmedic.cnn3d import Cnn3d

from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import getAbsPathEvenIfRelativeIsGiven
from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import checkIfAllElementsOfAListAreFilesAndExitIfNot
from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import checkListContainsCorrectNumberOfCasesOtherwiseExitWithError
from deepmedic.frontEndModules.frontEndHelpers.parsingFilesHelpers import checkThatAllEntriesOfAListFollowNameConventions

from deepmedic.frontEndModules.frontEndHelpers.createModelParametersClass import CreateModelSessionParameters
from deepmedic.frontEndModules.frontEndHelpers.preparationForSessionHelpers import makeFoldersNeededForCreateModelSession

from deepmedic.cnnHelpers import dump_cnn_to_gzip_file_dotSave
from deepmedic.genericHelpers import datetimeNowAsStr


class ModelConfig(object):
    configStruct = {} #In here will be placed all read arguments.
    def get(self,string1) :
        return self.configStruct[string1] if string1 in self.configStruct else None
    
    #Optional but highly suggested.
    MODEL_NAME = "modelName"
    #[REQUIRED] Output:
    FOLDER_FOR_OUTPUT = "folderForOutput" #MUST BE GIVEN
    
    #================ MODEL PARAMETERS =================
    NUMB_CLASSES = "numberOfOutputClasses"
    NUMB_INPUT_CHANNELS_NORMAL = "numberOfInputChannels"
    
    #===Normal pathway===
    N_FMS_NORM = "numberFMsPerLayerNormal"
    KERN_DIM_NORM = "kernelDimPerLayerNormal"
    RESID_CONN_LAYERS_NORM = "layersWithResidualConnNormal"
    LOWER_RANK_LAYERS_NORM = "lowerRankLayersNormal"
    
    #==Subsampled pathway==
    USE_SUBSAMPLED = "useSubsampledPathway"
    #The below should be mirroring the pathway, otherwise let them specify them but throw warning all around that receptive field should stay the same!
    N_FMS_SUBS = "numberFMsPerLayerSubsampled"
    KERN_DIM_SUBS = "kernelDimPerLayerSubsampled"
    SUBS_FACTOR = "subsampleFactor"
    RESID_CONN_LAYERS_SUBS = "layersWithResidualConnSubsampled"
    LOWER_RANK_LAYERS_SUBS = "lowerRankLayersSubsampled"
    
    #==Extra hidden FC Layers. Final Classification layer is not included in here.
    N_FMS_FC = "numberFMsPerLayerFC"
    KERN_DIM_1ST_FC = "kernelDimFor1stFcLayer"
    RESID_CONN_LAYERS_FC = "layersWithResidualConnFC"
    
    #Size of Image Segments
    SEG_DIM_TRAIN = "segmentsDimTrain"
    SEG_DIM_VAL = "segmentsDimVal"
    SEG_DIM_INFERENCE = "segmentsDimInference"
    
    #==Batch Sizes===
    #Required.
    BATCH_SIZE_TR = "batchSizeTrain"
    BATCH_SIZE_VAL = "batchSizeVal"
    BATCH_SIZE_INFER = "batchSizeInfer"
    
    #Dropout Rates:
    DROP_R_NORM = "dropoutRatesNormal"
    DROP_R_SUBS = "dropoutRatesSubsampled"
    DROP_R_FC = "dropoutRatesFc"
    
    #Regularization L1 and L2.
    L1_REG = "L1_reg"
    L2_REG = "L2_reg"
    
    #Initialization method of the kernel weights. Classic is what I was using for my first year. "Delving Deep" for journal.
    INITIAL_METHOD = "initializeClassic0orDelving1"
    #Activation Function for all convolutional layers:
    ACTIV_FUNCTION = "relu0orPrelu1"
    
    #Batch Normalization
    BN_ROLL_AV_BATCHES = "rollAverageForBNOverThatManyBatches"
    
    
    #====OPTIMIZATION=====
    LRATE = "learningRate"
    OPTIMIZER = "sgd0orAdam1orRms2"
    MOM_TYPE = "classicMom0OrNesterov1"
    MOM = "momentumValue"
    MOM_NORM_NONNORM = "momNonNorm0orNormalized1"
    #Adam
    B1_ADAM = "b1Adam"
    B2_ADAM = "b2Adam"
    EPS_ADAM = "epsilonAdam"
    #RMS
    RHO_RMS = "rhoRms"
    EPS_RMS = "epsilonRms"
    
#The argument should be absolute path to the config file for the model to create.
def deepMedicNewModelMain(modelConfigFilepath) :
    print "Given Model-Configuration File: ", modelConfigFilepath
    #Parse the config file in this naive fashion...
    modelConfig = ModelConfig()
    execfile(modelConfigFilepath, modelConfig.configStruct)
    configGet = modelConfig.get #Main interface
    
    """
    #Do checks.
    checkIfMainTestConfigIsCorrect(testConfig, testConfigFilepath, absPathToSavedModelFromCmdLine) #Checks REQUIRED fields are complete.
    checkIfFilesThatListFilesPerCaseAreCorrect(testConfig, testConfigFilepath) #Checks listing-files (whatever given).
    checkIfOptionalParametersAreGivenCorrectly(testConfig, testConfigFilepath)
    
    #At this point it was checked that all parameters (that could be checked) and filepaths are correct, pointing to files/dirs and all files/dirs exist.
    """
    
    #Create Folders and Logger
    mainOutputAbsFolder = getAbsPathEvenIfRelativeIsGiven(configGet(modelConfig.FOLDER_FOR_OUTPUT), modelConfigFilepath)
    modelName = configGet(modelConfig.MODEL_NAME) if configGet(modelConfig.MODEL_NAME) else CreateModelSessionParameters.getDefaultModelName()
    [folderForCnnModels,
    folderForLogs] = makeFoldersNeededForCreateModelSession(mainOutputAbsFolder, modelName)
    loggerFileName = folderForLogs + "/" + modelName + ".txt"
    sessionLogger = myLoggerModule.MyLogger(loggerFileName)
    
    sessionLogger.print3("CONFIG: The configuration file for the model-creation session was loaded from: " + str(modelConfigFilepath))
    
    #Fill in the session's parameters.
    createModelSessionParameters = CreateModelSessionParameters(
                    cnnModelName=modelName,
                    sessionLogger=sessionLogger,
                    mainOutputAbsFolder=mainOutputAbsFolder,
                    folderForSessionCnnModels=folderForCnnModels,
                    #===MODEL PARAMETERS===
                    numberClasses=configGet(modelConfig.NUMB_CLASSES),
                    numberOfInputChannelsNormal=configGet(modelConfig.NUMB_INPUT_CHANNELS_NORMAL),
                    #===Normal pathway===
                    numFMsNormal=configGet(modelConfig.N_FMS_NORM),
                    kernDimNormal=configGet(modelConfig.KERN_DIM_NORM),
                    residConnAtLayersNormal=configGet(ModelConfig.RESID_CONN_LAYERS_NORM),
                    lowerRankLayersNormal=configGet(ModelConfig.LOWER_RANK_LAYERS_NORM),
                    #==Subsampled pathway==
                    useSubsampledBool=configGet(modelConfig.USE_SUBSAMPLED),
                    numFMsSubsampled=configGet(modelConfig.N_FMS_SUBS),
                    kernDimSubsampled=configGet(modelConfig.KERN_DIM_SUBS),
                    subsampleFactor=configGet(modelConfig.SUBS_FACTOR),
                    residConnAtLayersSubsampled=configGet(ModelConfig.RESID_CONN_LAYERS_SUBS),
                    lowerRankLayersSubsampled=configGet(ModelConfig.LOWER_RANK_LAYERS_SUBS),
                    #==FC Layers====
                    numFMsFc=configGet(modelConfig.N_FMS_FC),
                    kernelDimensionsFirstFcLayer=configGet(modelConfig.KERN_DIM_1ST_FC),
                    residConnAtLayersFc=configGet(ModelConfig.RESID_CONN_LAYERS_FC),
                    #==Size of Image Segments ==
                    segmDimTrain=configGet(modelConfig.SEG_DIM_TRAIN),
                    segmDimVal=configGet(modelConfig.SEG_DIM_VAL),
                    segmDimInfer=configGet(modelConfig.SEG_DIM_INFERENCE),
                    #== Batch Sizes ==
                    batchSizeTrain=configGet(modelConfig.BATCH_SIZE_TR),
                    batchSizeVal=configGet(modelConfig.BATCH_SIZE_VAL),
                    batchSizeInfer=configGet(modelConfig.BATCH_SIZE_INFER),
                    #===Other Architectural Parameters ===
                    activationFunction=configGet(modelConfig.ACTIV_FUNCTION),
                    #==Dropout Rates==
                    dropNormal=configGet(modelConfig.DROP_R_NORM),
                    dropSubsampled=configGet(modelConfig.DROP_R_SUBS),
                    dropFc=configGet(modelConfig.DROP_R_FC),
                    #==Regularization==
                    l1Reg=configGet(modelConfig.L1_REG),
                    l2Reg=configGet(modelConfig.L2_REG),
                    #== Weight Initialization==
                    initialMethod=configGet(modelConfig.INITIAL_METHOD),
                    #== Batch Normalization ==
                    bnRollingAverOverThatManyBatches=configGet(modelConfig.BN_ROLL_AV_BATCHES),
                    #====Optimization=====
                    learningRate=configGet(modelConfig.LRATE),
                    optimizerSgd0Adam1Rms2=configGet(modelConfig.OPTIMIZER),
                    classicMom0Nesterov1=configGet(modelConfig.MOM_TYPE),
                    momentumValue=configGet(modelConfig.MOM),
                    momNonNormalized0Normalized1=configGet(modelConfig.MOM_NORM_NONNORM),
                    #Adam
                    b1Adam=configGet(modelConfig.B1_ADAM),
                    b2Adam=configGet(modelConfig.B2_ADAM),
                    eAdam=configGet(modelConfig.EPS_ADAM),
                    #Rms
                    rhoRms=configGet(modelConfig.RHO_RMS),
                    eRms=configGet(modelConfig.EPS_RMS)
                    )
    
    
    createModelSessionParameters.sessionLogger.print3("===========    NEW CREATE-MODEL SESSION    ============")
    createModelSessionParameters.printParametersOfThisSession()
    
    createModelSessionParameters.sessionLogger.print3("=========== Creating the CNN model ===============")
    cnn3dInstance = Cnn3d()
    cnn3dInstance.make_cnn_model(*createModelSessionParameters.getTupleForCnnCreation())
    
    cnn3dInstance.initializeTrainingState(*createModelSessionParameters.getTupleForInitializingTrainingState())
    cnn3dInstance.compileTrainFunction(*createModelSessionParameters.getTupleForCompilationOfTrainFunc())
    cnn3dInstance.compileValidationFunction(*createModelSessionParameters.getTupleForCompilationOfValFunc())
    cnn3dInstance.compileTestAndVisualisationFunction(*createModelSessionParameters.getTupleForCompilationOfTestFunc())
    
    filenameAndPathToSaveModel = createModelSessionParameters.getPathAndFilenameToSaveModel() + ".initial." + datetimeNowAsStr()
    filenameAndPathWhereModelWasSaved =  dump_cnn_to_gzip_file_dotSave(cnn3dInstance, filenameAndPathToSaveModel, sessionLogger)
    
    createModelSessionParameters.sessionLogger.print3("=========== Creation of the model: \"" + str(createModelSessionParameters.cnnModelName) +"\" finished =================")
    return (cnn3dInstance, filenameAndPathWhereModelWasSaved)

