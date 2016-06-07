# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

class TrainSessionParameters(object) :
	#THE LOGIC WHETHER I GOT A PARAMETER THAT I NEED SHOULD BE IN HERE!

	#To be called from outside too.
	@staticmethod
	def getDefaultSessionName() :
		return "trainSession"

	#REQUIRED:
	@staticmethod
	def errorRequireChannelsTraining():
		print "ERROR: Parameter \"channelsTraining\" needed but not provided in config file. This parameter should provide paths to files, as many as the channels (modalities) of the task. Each of the files should contain a list of paths, one for each case to train on. These paths in a file should point to the .nii(.gz) files that are the corresponding channel for a patient. Please provide it in the format: channelsTraining = [\"path-to-file-for-channel1\", ..., \"path-to-file-for-channelN\"]. The paths should be given in quotes, separated by commas (list of strings, python-style). Exiting."; exit(1) 
	errReqChansTr = errorRequireChannelsTraining
	@staticmethod
	def errorRequireGtLabelsTraining():
		print "ERROR: Parameter \"gtLabelsTraining\" needed but not provided in config file. This parameter should provide the path to a file. That file should contain a list of paths, one for each case to train on. These paths should point to the .nii(.gz) files that contain the corresponding Ground-Truth labels for a case. Please provide it in the format: gtLabelsTraining = \"path-to-file\". The path should be given in quotes (a string, python-style). Exiting."; exit(1) 
	errReqGtTr = errorRequireGtLabelsTraining


	@staticmethod
	def errorRequireSamplMasksAreProbabMapsTrain() :
		print "ERROR: Parameter \"samplingMasksAreProbabMapsTrain\" needed but not provided in config file. This parameters is needed when parameter \"useDefaultTrainingSamplingFromGtAndRoi\" = False, in order to know whether the provided masks are probability maps or segmentation labels. Please provide parameter in the form: samplingMasksAreProbabMapsTrain = True/False. True if the masks given at \"masksForPos(Neg)SamplingTrain\" are probability maps (can be non-normalized, like weights), or False if they are binary segmentation masks. Exiting."; exit(1) 
	errReqMasksTypeTr = errorRequireSamplMasksAreProbabMapsTrain
	@staticmethod
	def warnDefaultPosSamplMasksTrain() :
		print "WARN: Parameter \"weightedMapsForPosSamplingTrain\" was not provided in config file, even though advanced training options were triggered by setting \"useDefaultTrainingSamplingFromGtAndRoi\" = False. This parameter should point to a file that lists the paths (per case/patient) to weighted-maps that indicate where to perform the sampling of training samples. Can be provided in the format: weightedMapsForPosSamplingTrain = \"path-to-file\". The target file should have one entry per case (patient). Each of them should be pointing to .nii(.gz) files that indicate where to extract the positive samples. \n\tDEFAULT: In the case it is not provided (like now!) the corresponding samples are extracted uniformly from the whole volume!"
		return None 
	warnDefPosMasksTr = warnDefaultPosSamplMasksTrain
	@staticmethod
	def warnDefaultNegSamplMasksTrain() :
		print "WARN: Parameter \"weightedMapsForNegSamplingTrain\" was not provided in config file, even though advanced training options were triggered by setting \"useDefaultTrainingSamplingFromGtAndRoi\" = False. This parameter should point to a file that lists the paths (per case/patient) to weighted-maps that indicate where to perform the sampling of training samples. Can be provided in the format: weightedMapsForNegSamplingTrain = \"path-to-file\". The target file should have one entry per case (patient). Each of them should be pointing to .nii(.gz) files that indicate where to extract the negative samples. \n\tDEFAULT: In the case it is not provided (like now!) the corresponding samples are extracted uniformly from the whole volume!"
		return None
	warnDefNegMasksTr = warnDefaultNegSamplMasksTrain
	@staticmethod
	def errorRequirePredefinedLrSched() :
		print "ERROR: Parameter \"stable0orAuto1orPredefined2orExponential3LrSchedule\" was set to 2, to use Predefined Schedule for Learning-Rate decrease, but no Predefined Schedule was given. Please specify at which epochs to lower the Learning Rate, by providing the corresponding parameter in the format: predefinedSchedule = [epoch-for-1st-decrease, ..., epoch-for-last-decrease], where the epochs are specified by an integer > 0. Exiting."; exit(1) 
	errReqPredLrSch = errorRequirePredefinedLrSched
	@staticmethod
	def errorAutoRequiresValSamples() :
		print "ERROR: Parameter \"stable0orAuto1orPredefined2orExponential3LrSchedule\" was set to 1 (AUTO schedule). This schedule requires performing validation on samples throughout training, because this schedule lowers the Learning Rate when validation-accuracy plateaus. However the parameter \"performValidationOnSamplesThroughoutTraining\" was set to False in the configuration file, or was ommitted, which triggers the default value, False! Please set the parameter performValidationOnSamplesThroughoutTraining = True. You will then need to provide the path to the channels of the validation cases in the format: channelsValidation = [\"path-to-file-that-lists-paths-to-channel-1-for-every-case\", ..., \"path-to-file-that-lists-paths-to-channel-N-for-every-case\"] (python style list-of-strings). Also, you will need to provide the Ground-Truth for the validation cases, in the format:  gtLabelsValidation = \"path-to-file\", where the file lists the paths to the GT labels of each validation case. Exiting!"; exit(1)
	@staticmethod
	def errorRequireChannelsVal() :
		print "ERROR: Parameter \"channelsValidation\" was not provided, although it is required to perform validation, although validation was requested (parameters \"performValidationOnSamplesThroughoutTraining\" or \"performFullInferenceOnValidationImagesEveryFewEpochs\" was set to True). You will need to provide a list with path to files that list where the channels for each validation case can be found. The corresponding parameter must be provided in the format: channelsValidation = [\"path-to-file-that-lists-paths-to-channel-1-for-every-case\", ..., \"path-to-file-that-lists-paths-to-channel-N-for-every-case\"] (python style list-of-strings). Exiting."; exit(1)
	errReqChannsVal = errorRequireChannelsVal
	@staticmethod
	def errorReqGtLabelsVal() :
		print "ERROR: Parameter \"gtLabelsValidation\" was not provided, although it is required to perform validation on training-samples, which was requested (parameter \"performValidationOnSamplesThroughoutTraining\" was set to True). It is also useful so that the DSC score is reported if full-inference on the validation samples is performed (when parameter \"performFullInferenceOnValidationImagesEveryFewEpochs\" is set to True)! You will need to provide the path to a file that lists where the GT labels for each validation case can be found. The corresponding parameter must be provided in the format: gtLabelsValidation = \"path-to-file-that-lists-GT-labels-for-every-case\" (python style string). Exiting."; exit(1)


	#VALIDATION
	@staticmethod
	def errorReqNumberOfEpochsBetweenFullValInfGreaterThan0() :
		print "ERROR: It was requested to perform full-inference on validation images by setting parameter \"performFullInferenceOnValidationImagesEveryFewEpochs\" to True. For this, it is required to specify the number of epochs between two full-inference procedures. This number was given equal to 0. Please specify a number greater than 0, in the format: numberOfEpochsBetweenFullInferenceOnValImages = 1 (Any integer. Default is 1). Exiting!"; exit(1)
	@staticmethod
	def errorRequireNamesOfPredictionsVal() :
		print "ERROR: It was requested to perform full-inference on validation images by setting parameter \"performFullInferenceOnValidationImagesEveryFewEpochs\" to True and then save some of the results (segmentation maps, probability maps or feature maps), either manually or by default. For this, it is required to specify the path to a file, which should contain names to give to the results. Please specify the path to such a file in the format: namesForPredictionsPerCaseVal = \"./validation/validationNamesOfPredictionsSimple.cfg\" (python-style string). Exiting!"; exit(1)
	@staticmethod
	def errorRequirePercentOfPosSamplesVal():
		print "ERROR: Advanced sampling was enabled by setting: useDefaultUniformValidationSampling = False. This requires providing the percentage of validation samples that should be extracted as positives (from the positive weight-map). Please specify a float between 0.0 and 1.0, eg in the format: percentOfSamplesToExtractPositiveVal = 0.5. Exiting!"; exit(1)
	errReqPercPosTrVal = errorRequirePercentOfPosSamplesVal
	@staticmethod
	def warnDefaultPercentOfPosSamplesVal():
		print "WARN: Advanced sampling was enabled by setting: useDefaultUniformValidationSampling = False. This requires providing the percentage of validation samples that should be extracted as positives (from the positive weight-map). Please specify a float between 0.0 and 1.0, eg in the format: percentOfSamplesToExtractPositiveVal = 0.5. \n\tDEFAULT: In the case not given (like now!) default value of 0.5 is used!"
		return 0.5
	warnDefPercPosTrVal = warnDefaultPercentOfPosSamplesVal
	@staticmethod
	def warnDefaultPosSamplMasksVal() :
		print "WARN: Parameter \"weightedMapsForPosSamplingVal\" was not provided in config file, even though advanced validation-sampling options were triggered by setting \"useDefaultUniformValidationSampling\" = False. This parameter should point to a file that lists the paths (per case/patient) to weighted-maps that indicate where to perform the sampling of validation samples. Can be provided in the format: weightedMapsForPosSamplingVal = \"path-to-file\". The target file should have one entry per case (patient). Each of them should be pointing to .nii(.gz) files that indicate where to extract the positive samples. \n\tDEFAULT: In the case it is not provided (like now!) the corresponding samples are extracted uniformly from the whole volume!"
		return None 
	warnDefPosMasksVal = warnDefaultPosSamplMasksVal
	@staticmethod
	def warnDefaultNegSamplMasksVal() :
		print "WARN: Parameter \"weightedMapsForNegSamplingVal\" was not provided in config file, even though advanced  validation-sampling options were triggered by setting \"useDefaultUniformValidationSampling\" = False. This parameter should point to a file that lists the paths (per case/patient) to weighted-maps that indicate where to perform the sampling of validation samples. Can be provided in the format: weightedMapsForNegSamplingVal = \"path-to-file\". The target file should have one entry per case (patient). Each of them should be pointing to .nii(.gz) files that indicate where to extract the negative samples. \n\tDEFAULT: In the case it is not provided (like now!) the corresponding samples are extracted uniformly from the whole volume!"
		return None
	warnDefNegMasksVal = warnDefaultNegSamplMasksVal

	def __init__(	self,
			sessionName,
			sessionLogger,
			mainOutputAbsFolder,
			cnn3dInstance,
			cnnModelFilepath,

			#==================TRAINING====================
			folderForSessionCnnModels,
			listWithAListPerCaseWithFilepathPerChannelTrain,
			gtLabelsFilepathsTrain,

			#[Optionals]
			#~~~~~~~~~Sampling~~~~~~~
			roiMasksFilepathsTrain,
			percentOfSamplesToExtractPositTrain,
			#~~~~~~~~~Advanced Sampling~~~~~~~
			useDefaultTrainingSamplingFromGtAndRoi,
			pathsToWeightMapsOfEachCaseForPosSamplingTrain, #this is the absolute-path to the listing-file. Req if advancedSamplTrain
			pathsToWeightMapsOfEachCaseForNegSamplingTrain, #this is the absolute-path to the listing-file. Req if advancedSamplTrain
			samplingMasksAreProbMapsTrain,
			#~~~~~~~~Training Cycle ~~~~~~~
			numberOfEpochs,
			numberOfSubepochs,
			numOfCasesLoadedPerSubepoch,
			segmentsLoadedOnGpuPerSubepochTrain,

			#~~~~~~~ Learning Rate Schedule ~~~~~~~
			#Auto requires performValidationOnSamplesThroughoutTraining and providedGtForValidationBool
			stable0orAuto1orPredefined2orExponential3LrSchedule,
			
			#Stable + Auto + Predefined.
			whenDecreasingDivideLrBy,
			#Stable + Auto
			numEpochsToWaitBeforeLoweringLr,
			#Auto:
			minIncreaseInValidationAccuracyThatResetsWaiting,
			#Predefined.
			predefinedSchedule,
			#Exponential
			exponentialSchedForLrAndMom,
			
			#~~~~~~~ Augmentation~~~~~~~~~~~~
			reflectImagesPerAxis,
			performIntAugm,
			sampleIntAugmShiftWithMuAndStd,
			sampleIntAugmMultiWithMuAndStd,

			#==================VALIDATION=====================
			performValidationOnSamplesThroughoutTraining,
			performFullInferenceOnValidationImagesEveryFewEpochs,
			#Required:
			listWithAListPerCaseWithFilepathPerChannelVal,
			gtLabelsFilepathsVal,

			validationSegmentsLoadedOnGpuPerSubepoch,

			#[Optionals]
			roiMasksFilepathsVal, #For default sampling and for fast inference. Optional. Otherwise from whole image.

			#~~~~~~~~Full Inference~~~~~~~~
			numberOfEpochsBetweenFullInferenceOnValImages,
			#Output				
			namesToSavePredictionsAndFeaturesVal,
			#predictions
			saveSegmentationVal,
			saveProbMapsBoolPerClassVal,
			folderForPredictionsVal,
			#features:
			saveIndividualFmImagesVal,
			saveMultidimensionalImageWithAllFmsVal,
			indicesOfFmsToVisualisePerPathwayAndLayerVal,
			folderForFeaturesVal,

			#~~~~~~~~ Advanced Validation Sampling ~~~~~~~~~~
			useDefaultUniformValidationSampling,

			percentOfSamplesToExtractPositVal,
			pathsToWeightMapsOfEachCaseForPosSamplingVal, #this is the absolute-path to the listing-file. Req if advancedSamplTrain
			pathsToWeightMapsOfEachCaseForNegSamplingVal, #this is the absolute-path to the listing-file. Req if advancedSamplTrain
			samplingMasksAreProbMapsVal,
			#==============Generic and Preprocessing===============
			padInputImagesBool,

			):

		#Importants for running session.

		self.sessionName = sessionName if sessionName else self.getDefaultSessionName()
		self.sessionLogger = sessionLogger
		self.mainOutputAbsFolder = mainOutputAbsFolder
		self.cnn3dInstance = cnn3dInstance #Must be filled from outside after initialization
		self.cnnModelFilepath = cnnModelFilepath #This is where the model was loaded from.
		self.cnnModelName = cnn3dInstance.cnnModelName if cnn3dInstance.cnnModelName <> None else "defaultCnnModelName"


		#====================TRAINING==========================
		self.pathAndFilenameToSaveTrainedModels = folderForSessionCnnModels + "/" + self.cnnModelName + "." + self.sessionName
		#[[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
		self.channelsFilepathsTrain = listWithAListPerCaseWithFilepathPerChannelTrain if listWithAListPerCaseWithFilepathPerChannelTrain else self.errReqChansTr()
		self.gtLabelsFilepathsTrain = gtLabelsFilepathsTrain if gtLabelsFilepathsTrain else self.errReqGtTr()

		#[Optionals]
		#~~~~~~~~~Sampling~~~~~~~
		self.providedRoiMasksTrain = True if roiMasksFilepathsTrain else False
		self.roiMasksFilepathsTrain = roiMasksFilepathsTrain if roiMasksFilepathsTrain else [] #For Int-Augm and for Mask-Where-Neg if no advanced-sampling-train.
		self.percentOfSamplesToExtractPositTrain = percentOfSamplesToExtractPositTrain if percentOfSamplesToExtractPositTrain else 0.5
		#~~~~~~~~~Advanced Sampling~~~~~~~
		#ADVANCED CONFIG IS DISABLED HERE IF useDefaultSamplingFromGtAndRoi = True!
		self.useDefaultTrainingSamplingFromGtAndRoi = useDefaultTrainingSamplingFromGtAndRoi if useDefaultTrainingSamplingFromGtAndRoi <> None else True
		if self.useDefaultTrainingSamplingFromGtAndRoi :
			self.pathsToWeightMapsOfEachCaseForPosSamplingTrain = self.gtLabelsFilepathsTrain
			self.pathsToWeightMapsOfEachCaseForNegSamplingTrain = self.roiMasksFilepathsTrain if self.roiMasksFilepathsTrain else None
			self.samplingMasksAreProbMapsTrain = False
		else :
			self.pathsToWeightMapsOfEachCaseForPosSamplingTrain = pathsToWeightMapsOfEachCaseForPosSamplingTrain if\
							pathsToWeightMapsOfEachCaseForPosSamplingTrain else self.warnDefPosMasksTr()# Warning returns None.
			self.pathsToWeightMapsOfEachCaseForNegSamplingTrain = pathsToWeightMapsOfEachCaseForNegSamplingTrain if\
							pathsToWeightMapsOfEachCaseForNegSamplingTrain else self.warnDefNegMasksTr()# Warning returns None.
			self.samplingMasksAreProbMapsTrain = samplingMasksAreProbMapsTrain if samplingMasksAreProbMapsTrain <> None else True #This should be kinda hidden. Only for me I can use =False if segments.

		self.providingMaskWhereToGetPosSamplesTrain = True if self.pathsToWeightMapsOfEachCaseForPosSamplingTrain else False #If false, samples from whole image the corresponding samples.
		self.providingMaskWhereToGetNegSamplesTrain = True if self.pathsToWeightMapsOfEachCaseForNegSamplingTrain else False #If false, samples from whole image the corresponding samples.

		#~~~~~~~~ Training Cycle ~~~~~~~~~~~
		self.numberOfEpochs = numberOfEpochs if numberOfEpochs <> None else 35
		self.numberOfSubepochs = numberOfSubepochs if numberOfSubepochs <> None else 20
		self.numOfCasesLoadedPerSubepoch = numOfCasesLoadedPerSubepoch if numOfCasesLoadedPerSubepoch <> None else 50
		self.segmentsLoadedOnGpuPerSubepochTrain = segmentsLoadedOnGpuPerSubepochTrain if segmentsLoadedOnGpuPerSubepochTrain <> None else 1000


		#~~~~~~~ Learning Rate Schedule ~~~~~~~~
		#Auto requires performValidationOnSamplesThroughoutTraining and providedGtForValidationBool
		self.stable0orAuto1orPredefined2orExponential3LrSchedule = stable0orAuto1orPredefined2orExponential3LrSchedule if stable0orAuto1orPredefined2orExponential3LrSchedule <> None else 3
		#Used for Stable + Auto + Predefined.
		self.whenDecreasingDivideLrBy = whenDecreasingDivideLrBy if whenDecreasingDivideLrBy <> None else 2.0
		#Stable + Auto. Set this to 0 to never lower it!
		self.numEpochsToWaitBeforeLoweringLr = numEpochsToWaitBeforeLoweringLr if numEpochsToWaitBeforeLoweringLr <> None else 3
		#Auto:
		self.minIncreaseInValidationAccuracyThatResetsWaiting = minIncreaseInValidationAccuracyThatResetsWaiting if minIncreaseInValidationAccuracyThatResetsWaiting <> None else 0.0005
		#Predefined.
		self.predefinedSchedLowerLrAtEpochs = predefinedSchedule
		if self.stable0orAuto1orPredefined2orExponential3LrSchedule == 2 and self.predefinedSchedLowerLrAtEpochs == None :
			self.errReqPredLrSch()
		#Exponential
		self.exponentialSchedForLrAndMom = exponentialSchedForLrAndMom if exponentialSchedForLrAndMom else [self.numberOfEpochs/2, 1.0/(2**(8)),  0.9]

		#~~~~~~~~~~~~~~ Augmentation~~~~~~~~~~~~~~
		self.reflectImagesPerAxis = reflectImagesPerAxis if reflectImagesPerAxis else [False, False, False]
		self.performIntAugm = performIntAugm if performIntAugm <> None else False
		if self.performIntAugm :
			self.sampleIntAugmShiftWithMuAndStd = sampleIntAugmShiftWithMuAndStd if sampleIntAugmShiftWithMuAndStd else [0.0 , 0.1]
			self.sampleIntAugmMultiWithMuAndStd = sampleIntAugmMultiWithMuAndStd if sampleIntAugmMultiWithMuAndStd else [1.0 , 0.0]
			self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd = [2, 1,
								self.sampleIntAugmShiftWithMuAndStd,self.sampleIntAugmMultiWithMuAndStd]
		else :
			
			self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd = [0, "plcholder", [], []]

		#===================VALIDATION========================
		self.performValidationOnSamplesThroughoutTraining = performValidationOnSamplesThroughoutTraining if performValidationOnSamplesThroughoutTraining <> None else False
		if self.stable0orAuto1orPredefined2orExponential3LrSchedule == 1 and not self.performValidationOnSamplesThroughoutTraining :
			self.errorAutoRequiresValSamples()
		self.performFullInferenceOnValidationImagesEveryFewEpochs = performFullInferenceOnValidationImagesEveryFewEpochs if performFullInferenceOnValidationImagesEveryFewEpochs <> None else False

		#Input:
		if self.performValidationOnSamplesThroughoutTraining or self.performFullInferenceOnValidationImagesEveryFewEpochs :
			#[[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
			self.channelsFilepathsVal = listWithAListPerCaseWithFilepathPerChannelVal if listWithAListPerCaseWithFilepathPerChannelVal else self.errReqChannsVal()
		else :
			self.channelsFilepathsVal = []
		if self.performValidationOnSamplesThroughoutTraining :
			self.gtLabelsFilepathsVal = gtLabelsFilepathsVal if gtLabelsFilepathsVal else self.errorReqGtLabelsVal()
		elif self.performFullInferenceOnValidationImagesEveryFewEpochs :
			self.gtLabelsFilepathsVal = gtLabelsFilepathsVal if gtLabelsFilepathsVal else []
		else : # Dont perform either of the two validations.
			self.gtLabelsFilepathsVal = []
		self.providedGtVal = True if self.gtLabelsFilepathsVal else False
		
		#[Optionals]
		self.providedRoiMasksVal = True if roiMasksFilepathsVal else False #For fast inf.
		self.roiMasksFilepathsVal = roiMasksFilepathsVal if roiMasksFilepathsVal else [] #Also for default sampling of neg segs.

		#~~~~~Validation on Samples~~~~~~~~
		self.validationSegmentsLoadedOnGpuPerSubepoch = validationSegmentsLoadedOnGpuPerSubepoch if validationSegmentsLoadedOnGpuPerSubepoch <> None else 3000

		#~~~~~~~~~Advanced Validation Sampling~~~~~~~~~~~
		#ADVANCED OPTION ARE DISABLED IF useDefaultUniformValidationSampling = True!
		self.useDefaultUniformValidationSampling = useDefaultUniformValidationSampling if useDefaultUniformValidationSampling <> None else True
		if self.useDefaultUniformValidationSampling :
			self.percentOfSamplesToExtractPositVal = 0.0
			self.pathsToWeightMapsOfEachCaseForPosSamplingVal = None
			self.pathsToWeightMapsOfEachCaseForNegSamplingVal = None # To grab from whole image for uniforma sampling.
			self.samplingMasksAreProbMapsVal = False
		else :
			self.percentOfSamplesToExtractPositVal = percentOfSamplesToExtractPositVal if percentOfSamplesToExtractPositVal <> None else self.warnDefPercPosTrVal()
			self.pathsToWeightMapsOfEachCaseForPosSamplingVal = pathsToWeightMapsOfEachCaseForPosSamplingVal if pathsToWeightMapsOfEachCaseForPosSamplingVal else self.warnDefPosMasksVal()
			self.pathsToWeightMapsOfEachCaseForNegSamplingVal = pathsToWeightMapsOfEachCaseForNegSamplingVal if pathsToWeightMapsOfEachCaseForNegSamplingVal else self.warnDefNegMasksVal()
			self.samplingMasksAreProbMapsVal = samplingMasksAreProbMapsVal if samplingMasksAreProbMapsVal <> None else True #This should be kinda hidden. Only for me I can use =False if segmentations.
		self.providingMaskWhereToGetPosSamplesVal = True if self.pathsToWeightMapsOfEachCaseForPosSamplingVal else False #If false, samples from whole image the corresponding samples.
		self.providingMaskWhereToGetNegSamplesVal = True if self.pathsToWeightMapsOfEachCaseForNegSamplingVal else False #If false, samples from whole image the corresponding samples.

		#~~~~~~Full inference on validation image~~~~~~
		self.numberOfEpochsBetweenFullInferenceOnValImages = numberOfEpochsBetweenFullInferenceOnValImages if numberOfEpochsBetweenFullInferenceOnValImages <> None else 1
		if self.numberOfEpochsBetweenFullInferenceOnValImages == 0 and self.performFullInferenceOnValidationImagesEveryFewEpochs :
			self.errorReqNumberOfEpochsBetweenFullValInfGreaterThan0()

		#predictions
		self.saveSegmentationVal = saveSegmentationVal if saveSegmentationVal <> None else True
		self.saveProbMapsBoolPerClassVal = saveProbMapsBoolPerClassVal if saveProbMapsBoolPerClassVal else [True]*cnn3dInstance.numberOfOutputClasses
		self.filepathsToSavePredictionsForEachPatientVal = None #Filled by call to self.makeFilepathsForPredictionsAndFeatures()
		#features:
		self.saveIndividualFmImagesVal = saveIndividualFmImagesVal if saveIndividualFmImagesVal <> None else False
		self.saveMultidimensionalImageWithAllFmsVal = saveMultidimensionalImageWithAllFmsVal if saveMultidimensionalImageWithAllFmsVal <> None else False
		self.indicesOfFmsToVisualisePerPathwayAndLayerVal = [item if item else [] for item in indicesOfFmsToVisualisePerPathwayAndLayerVal] #By default, save none.
		self.indicesOfFmsToVisualisePerPathwayAndLayerVal.append([]) #for the Zoomed-in pathway. HIDDEN
		self.filepathsToSaveFeaturesForEachPatientVal = None #Filled by call to self.makeFilepathsForPredictionsAndFeatures()

		#Output:
		#Given by the config file, and is then used to fill filepathsToSavePredictionsForEachPatient and filepathsToSaveFeaturesForEachPatient.
		self.namesToSavePredictionsAndFeaturesVal = namesToSavePredictionsAndFeaturesVal  
		if not self.namesToSavePredictionsAndFeaturesVal and self.performFullInferenceOnValidationImagesEveryFewEpochs and (self.saveSegmentationVal or True in self.saveProbMapsBoolPerClassVal or self.saveIndividualFmImages or self.saveMultidimensionalImageWithAllFms) :
			self.errorRequireNamesOfPredictionsVal()

		#===================== OTHERS======================
		#Preprocessing
		self.padInputImagesBool = padInputImagesBool if padInputImagesBool <> None else True

		#Others useful internally or for reporting:
		self.numberOfCasesTrain = len(self.channelsFilepathsTrain)
		self.numberOfCasesVal = len(self.channelsFilepathsVal)
		self.numberOfClasses = cnn3dInstance.numberOfOutputClasses
		self.numberOfChannels = cnn3dInstance.numberOfImageChannelsPath1

		#HIDDENS, no config allowed for these at the moment:
		self.useSameSubChannelsAsSingleScale = True
		self.subsampledChannelsFilepathsTrain = "placeholder" #List of Lists with filepaths per patient. Only used when above is False.
		self.subsampledChannelsFilepathsVal = "placeholder" #List of Lists with filepaths per patient. Only used when above is False.
		self.smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage = [None, None]

		self.numberOfEpochsToWeightTheClassesInTheCostFunction = 0

		#layersOfLayerTypesToTrain = [[],[],[0,1,2],[]]# passing "all" to compileNewTrainFunction() trains them all. Else I need to have full entries for all layer-types, with the indices of the layers I want to get trained. Used with: cnn3dInstance.compileNewTrainFunction(myLogger, layersOfLayerTypesToTrain, learning_rate, momentum, costFunctionLetter)

		self._makeFilepathsForPredictionsAndFeaturesVal( folderForPredictionsVal, folderForFeaturesVal )


	def printParametersOfThisSession(self) :
		logPrint = self.sessionLogger.print3
		logPrint("=============================================================")
		logPrint("=============== PARAMETERS FOR THIS SESSION =================")
		logPrint("=============================================================")
		logPrint("Session's name = " + str(self.sessionName))
		logPrint("CNN model's name = " + str(self.cnnModelName))
		logPrint("CNN model was loaded from = " + str(self.cnnModelFilepath))
		logPrint("~~Output~~")
		logPrint("Main output folder = " + str(self.mainOutputAbsFolder))
		logPrint("Path and filename to save trained models = " + str(self.pathAndFilenameToSaveTrainedModels))

		logPrint("~~~~~~~~~~~~~~~~~~Generic Information~~~~~~~~~~~~~~~~")
		logPrint("Number of Classes (incl. background) (from cnnModel) = " + str(self.numberOfClasses))
		logPrint("Number of Channels (Normal Pathway) (from cnnModel) = " + str(self.numberOfChannels))
		logPrint("Number of Cases for Training = " + str(self.numberOfCasesTrain))
		logPrint("Number of Cases for Validation = " + str(self.numberOfCasesVal))

		logPrint("~~~~~~~~~~~~~~~~~~Training parameters~~~~~~~~~~~~~~~~")
		logPrint("Filepaths to Channels of the Training Cases = " + str(self.channelsFilepathsTrain))
		logPrint("Filepaths to Ground-Truth labels of the Training Cases = " + str(self.gtLabelsFilepathsTrain))

		logPrint("~~Sampling~~")
		logPrint("Region-Of-Interest Masks provided = " + str(self.providedRoiMasksTrain))
		logPrint("Filepaths to ROI Masks of the Training Cases = " + str(self.roiMasksFilepathsTrain))
		logPrint("Percent of Samples to extract from positive (foreground) = " + str(self.percentOfSamplesToExtractPositTrain))
		logPrint("~~Advanced Sampling (Sampling method further parameters)~~")
		logPrint("Using default sampling = " + str(self.useDefaultTrainingSamplingFromGtAndRoi))
		logPrint("Providing Mask where to get Positive samples (if this is False, samples will be extracted from whole image) = " + str(self.providingMaskWhereToGetPosSamplesTrain))
		logPrint("Paths to weight-maps/masks for Positive sampling = " + str(self.pathsToWeightMapsOfEachCaseForPosSamplingTrain))
		logPrint("Providing Mask where to get Negative samples (if this is False, samples will be extracted from whole image) = " + str(self.providingMaskWhereToGetNegSamplesTrain))
		logPrint("Paths to weight-maps/masks for Negative sampling = " + str(self.pathsToWeightMapsOfEachCaseForNegSamplingTrain))
		logPrint("Sampling masks are probability maps = " + str(self.samplingMasksAreProbMapsTrain) + ". NOTE: if this is False, we assume they were the GT-labels and the ROI. If Mask for Pos was provided, it is excluded from the NegSampling mask internally (be it ROI or whole image). If this is True (prob maps passed, for weighted sampling) we do not do all this.")

		logPrint("~~Training Cycle~~")
		logPrint("Number of Epochs = " + str(self.numberOfEpochs))
		logPrint("Number of Subepochs per epoch = " + str(self.numberOfSubepochs))
		logPrint("Number of cases to load per Subepoch (for extracting the samples for this subepoch) = " + str(self.numOfCasesLoadedPerSubepoch))
		logPrint("Number of Segments loaded on GPU per subepoch for Training = " + str(self.segmentsLoadedOnGpuPerSubepochTrain) + ". NOTE: This number of segments divided by the batch-size defines the number of optimization-iterations that will be performed every subepoch!")

		logPrint("~~Learning Rate Schedule~~")
		logPrint("Schedule Type: Stable (0), Auto (1), Predefined (2), Exponential (3) = " + str(self.stable0orAuto1orPredefined2orExponential3LrSchedule))
		logPrint("[Stable/Auto/Predefined] When decreasing Learning Rate, divide LR by = " + str(self.whenDecreasingDivideLrBy))
		logPrint("[Stable/Auto] Wait that many epochs before lowering LR = " + str(self.numEpochsToWaitBeforeLoweringLr))
		logPrint("[Auto] Minimum increase in validation accuracy (0. to 1.) that resets the waiting counter = " + str(self.minIncreaseInValidationAccuracyThatResetsWaiting))
		logPrint("[Predefined] Predefined Schedule of Epochs when the LR will be lowered = " + str(self.predefinedSchedLowerLrAtEpochs))
		logPrint("[Exponential] [Number of initial epochs to not change LR and Mom, final value of LR, final value of Momentum] = " + str(self.exponentialSchedForLrAndMom))

		logPrint("~~Data Augmentation During Training~~")
		logPrint("Reflect images per axis = " + str(self.reflectImagesPerAxis))
		logPrint("Perform intensity-augmentation [I'= (I+shift)*mult] = " + str(self.performIntAugm))
		logPrint("[Int. Augm.] Sample Shift from N(mu,std) = " + str(self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd[2]))
		logPrint("[Int. Augm.] Sample Multi from N(mu,std) = " + str(self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd[3]))
		logPrint("[Int. Augm.] (DEBUGGING:) full parameters [no(0)/images(1)/segms(2), nonNorm(0)/norm(1), shift, mult] = " + str(self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd))

		logPrint("~~~~~~~~~~~~~~~~~~Validation parameters~~~~~~~~~~~~~~~~")
		logPrint("Perform Validation on Samples throughout training? = " + str(self.performValidationOnSamplesThroughoutTraining))
		logPrint("Perform Full Inference on validation cases every few epochs? = " + str(self.performFullInferenceOnValidationImagesEveryFewEpochs))
		logPrint("Filepaths to Channels of the Validation Cases (Req for either of the above) = " + str(self.channelsFilepathsVal))
		logPrint("Provided Ground-Truth for Validation = " + str(self.providedGtVal) + ". NOTE: Required for Val on samples. Not Req for Full-Inference, but DSC will be reported if provided.")
		logPrint("Filepaths to Ground-Truth labels of the Validation Cases = " + str(self.gtLabelsFilepathsVal))
		logPrint("Provided ROI masks for Validation = " + str(self.providedRoiMasksVal) + ". NOTE: Validation-sampling and Full-Inference will be limited within this mask if provided. If not provided, Negative Validation samples will be extracted from whole volume, except if advanced-sampling is enabled, and the user provided separate weighting-maps/masks for sampling.")
		logPrint("Filepaths to ROI masks for Validation Cases = " + str(self.roiMasksFilepathsVal))

		logPrint(">>>>>Validation on Samples throughout Training<<<<<")
		logPrint("Number of Segments loaded on GPU per subepoch for Validation = " + str(self.validationSegmentsLoadedOnGpuPerSubepoch))
		logPrint("Use default uniform sampling for validation? = " + str(self.useDefaultUniformValidationSampling) + ". NOTE: Adv.Sampl.Params are auto-set to perform uniform-sampling if True.")
		logPrint("~~Advanced Sampling~~")
		logPrint("Percent of Samples to extract from positive (foreground) = " + str(self.percentOfSamplesToExtractPositVal))
		logPrint("Providing Mask where to get Positive samples (if this is False, samples will be extracted from whole image) = " + str(self.providingMaskWhereToGetPosSamplesVal))
		logPrint("Paths to weight-maps/masks for Positive sampling = " + str(self.pathsToWeightMapsOfEachCaseForPosSamplingVal))
		logPrint("Providing Mask where to get Negative samples (if this is False, samples will be extracted from whole image) = " + str(self.providingMaskWhereToGetNegSamplesVal))
		logPrint("Paths to weight-maps/masks for Negative sampling = " + str(self.pathsToWeightMapsOfEachCaseForNegSamplingTrain))
		logPrint("Sampling masks are probability maps = " + str(self.samplingMasksAreProbMapsVal) + ". NOTE: if this is False, we assume they were the GT-labels and the ROI. In this case, if Pos Mask was provided, it is excluded from the NegSampling mask internally (be it ROI or whole image). If this is True (prob maps passed, for weighted sampling) we do not do this.")

		logPrint(">>>>>Validation with Full Inference on Validation Cases<<<<<")
		logPrint("Perform Full-Inference on Val. cases every that many epochs = " + str(self.numberOfEpochsBetweenFullInferenceOnValImages))
		logPrint("~~Predictions (segmentations and prob maps on val. cases)~~")
		logPrint("Save Segmentations = " + str(self.saveSegmentationVal))
		logPrint("Save Probability Maps for each class = " + str(self.saveProbMapsBoolPerClassVal))
		logPrint("Filepaths to save results per case = " + str(self.filepathsToSavePredictionsForEachPatientVal))
		logPrint("~~Feature Maps~~")
		logPrint("Save Feature Maps = " + str(self.saveIndividualFmImagesVal))
		logPrint("Save FMs in a 4D-image = " + str(self.saveMultidimensionalImageWithAllFmsVal))
		logPrint("Min/Max Indices of FMs to visualise per pathway-type and per layer = " + str(self.indicesOfFmsToVisualisePerPathwayAndLayerVal))
		logPrint("Filepaths to save FMs per case = " + str(self.filepathsToSaveFeaturesForEachPatientVal))

		logPrint("~~~~~~~~~~~~~~~~~~Other Generic Parameters~~~~~~~~~~~~~~~~")
		logPrint("~~Pre Processing~~")
		logPrint("Pad Input Images = " + str(self.padInputImagesBool))

		logPrint("========== Done with printing session's parameters ==========")
		logPrint("=============================================================")
	
	def getTupleForCnnTraining(self) :

		borrowFlag = True
		
		percentThatArePositiveSamplesVal = 0.0

		trainTuple = (self.sessionLogger,
			self.pathAndFilenameToSaveTrainedModels,
			self.cnn3dInstance,

			self.performValidationOnSamplesThroughoutTraining,
			[self.saveSegmentationVal, self.saveProbMapsBoolPerClassVal],

			self.filepathsToSavePredictionsForEachPatientVal,

		        self.channelsFilepathsTrain,
		        self.channelsFilepathsVal,

			self.gtLabelsFilepathsTrain,
			self.providedGtVal,
			self.gtLabelsFilepathsVal,

		        self.providingMaskWhereToGetPosSamplesTrain, #Always true, since either GT labels or advanced-mask-where-to-pos
			self.pathsToWeightMapsOfEachCaseForPosSamplingTrain,
		        self.providingMaskWhereToGetPosSamplesVal, #If false, corresponding samples will be extracted uniformly from whole image.
			self.pathsToWeightMapsOfEachCaseForPosSamplingVal,

			self.providedRoiMasksTrain, #for int-augm.
			self.roiMasksFilepathsTrain,#for int-augm
		        self.providedRoiMasksVal, #for fast inf
		        self.roiMasksFilepathsVal, #fast inf and also for default sampling of neg segs.

		        self.providingMaskWhereToGetNegSamplesTrain, #If false, will extract neg samples from whole image. Else ROI or adv-mask.
			self.pathsToWeightMapsOfEachCaseForNegSamplingTrain,
		        self.providingMaskWhereToGetNegSamplesVal, #If false, corresponding samples will be extracted uniformly from whole image.
			self.pathsToWeightMapsOfEachCaseForNegSamplingVal,

			self.samplingMasksAreProbMapsTrain,
			self.samplingMasksAreProbMapsVal,

		        borrowFlag,
		        self.numberOfEpochs,
		        self.numberOfSubepochs,
		        self.numOfCasesLoadedPerSubepoch,
		        self.segmentsLoadedOnGpuPerSubepochTrain,
			self.validationSegmentsLoadedOnGpuPerSubepoch,
			self.percentOfSamplesToExtractPositTrain,
			self.percentOfSamplesToExtractPositVal,

			#-------Preprocessing-----------
			self.padInputImagesBool,
			self.smoothChannelsWithGaussFilteringStdsForNormalAndSubsampledImage,
			#-------Data Augmentation-------
			self.intAugmOnPairs2Images1None0_imAlreadyNormal_ShiftMuStd_MultiMuStd,
			self.reflectImagesPerAxis,

		        self.useSameSubChannelsAsSingleScale,

		        self.subsampledChannelsFilepathsTrain,
		        self.subsampledChannelsFilepathsVal,
		
			#Learning Rate Schedule:
			self.stable0orAuto1orPredefined2orExponential3LrSchedule,
			self.minIncreaseInValidationAccuracyThatResetsWaiting,
			self.numEpochsToWaitBeforeLoweringLr,
			self.whenDecreasingDivideLrBy,
			self.predefinedSchedLowerLrAtEpochs,
			self.exponentialSchedForLrAndMom,

			#Weighting Classes differently in the CNN's cost function during training:
			self.numberOfEpochsToWeightTheClassesInTheCostFunction,


			self.performFullInferenceOnValidationImagesEveryFewEpochs, #Even if not providedGtForValidationBool, inference will be performed if this == True, to save the results, eg for visual.
			self.numberOfEpochsBetweenFullInferenceOnValImages, # Should not be == 0, except if performFullInferenceOnValidationImagesEveryFewEpochsBool == False

			#--------For FM visualisation---------
			self.saveIndividualFmImagesVal,
			self.saveMultidimensionalImageWithAllFmsVal,
			self.indicesOfFmsToVisualisePerPathwayAndLayerVal,
			self.filepathsToSaveFeaturesForEachPatientVal
		        )

		return trainTuple


	def _makeFilepathsForPredictionsAndFeaturesVal(self,
						absPathToFolderForPredictionsFromSession,
						absPathToFolderForFeaturesFromSession
						) :
		self.filepathsToSavePredictionsForEachPatientVal = []
		self.filepathsToSaveFeaturesForEachPatientVal = []
		for case_i in xrange(self.numberOfCasesVal) :
			if self.saveSegmentationVal :
				filepathForCasePrediction = absPathToFolderForPredictionsFromSession + "/" + self.namesToSavePredictionsAndFeaturesVal[case_i]
				self.filepathsToSavePredictionsForEachPatientVal.append( filepathForCasePrediction )
			

			if True in self.saveProbMapsBoolPerClassVal :
				filepathForCaseFeatures = absPathToFolderForFeaturesFromSession + "/" + self.namesToSavePredictionsAndFeaturesVal[case_i]
				self.filepathsToSaveFeaturesForEachPatientVal.append( filepathForCaseFeatures )








