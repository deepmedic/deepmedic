# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

'''
This script parses training logs and plots the reported accuracy metrics (mean accuracy, sensitivity, specificity and DSC over the predicted samples and DSC achieved by segmentation of the whole scans of the validation subjects.

The script is ugly in its current form. Will be updated soon.
Last update: 25 April 2016
'''

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
import sys
import argparse

def setupArgParser() :
	parser = argparse.ArgumentParser( prog='plotterOfTrainingProgress', formatter_class=argparse.RawTextHelpFormatter,
	description='''This script parses training logs and plots the reported accuracy metrics (mean accuracy, sensitivity, specificity and DSC over the predicted samples and DSC achieved by segmentation of the whole scans of the validation subjects.''')

	parser.add_argument("-logs", dest='inp_logs_list', nargs='+', type=str, help="Paths to training logs. More than one log can be given, to plot accuracy for multiple experiments. \nFormat: -logs log1.txt log2.txt logs3.txt ...")
	parser.add_argument("-class", dest='inp_class_from_each_log_list', nargs='+', type=int, help="Specify which class to plot accuracy from each log (useful for multiclass tasks). Provide as many arguments as the number of logs given. \nFormat: -class 0 0 2 ... \nDefault: class-0 will be plotted for each experiment. \nNOTE: Accuracy for Class-0 corresponds to \"whole\" foreground in the logs. Although NIFTIs for class-0 are actually for the background. To-be fixed.")

	return parser

def runConfigFileAsPythonAndReturnTheTrainingSessionName(pathToLog) :
	envGivenInFile = {}
	execfile(pathToLog, envGivenInFile)

	#In case that the entry is commented-out/not in config file.
	trainSessionName = envGivenInFile["sessionName"] if "sessionName" in envGivenInFile else None
	subepochPerEpoch = envGivenInFile["numberOfSubepochs"] if "numberOfSubepochs" in envGivenInFile else None
	epochsBetweenEachFullInfer = envGivenInFile["numberOfEpochsBetweenFullInferenceOnValImages"] if "numberOfEpochsBetweenFullInferenceOnValImages" in envGivenInFile else None

	return (trainSessionName, subepochPerEpoch, epochsBetweenEachFullInfer)


def parseLogFileAndGetPathToTrainingConfigFile(pathToLog) :
	logLineWithConfigFilePathString = "CONFIG: The configuration file for the training session was loaded from: "
	f = open(pathToLog, 'r')
	newLine = f.readline()	
	while newLine :
		indexWherePatternStringStarts = newLine.rfind(logLineWithConfigFilePathString)
		if indexWherePatternStringStarts > -1 :
			restOfLine = newLine[indexWherePatternStringStarts+len(logLineWithConfigFilePathString):]
			pathToConfigFile = restOfLine.strip()
			return pathToConfigFile

		newLine = f.readline()
	print("ERROR: No log line that holds the path to the config file that was used for this training-session: ", pathToLog)
	print("Exiting!")
	exit(1)

def parseVariablesFromOfTrainingSessionsFromListOfLogs(inLogsList) :
	listOfTrainingSessionNames = []
	listOfSubepochsPerEp = []
	listOfEpochsPerFullInferList = []

	experimentCount = 0
	for pathToLog in inLogsList :
		#Read log file, get the path to the config file. It should be absolute path when logged.
		absPathToTrainingConfigFile = parseLogFileAndGetPathToTrainingConfigFile(pathToLog)

		#Get the variables from the config file.
		trainSessionName, subepochPerEpoch, epochsBetweenEachFullInfer = runConfigFileAsPythonAndReturnTheTrainingSessionName(absPathToTrainingConfigFile)
		#In case the name was not provided at all (eg commented out)
		if not trainSessionName :
			trainSessionName = "trainingSession-" + str(experimentCount)
		if not subepochPerEpoch :
			subepochPerEpoch = 20 #default
		if not trainSessionName :
			epochsBetweenEachFullInfer = 1 #default

		listOfTrainingSessionNames.append(trainSessionName)
		listOfSubepochsPerEp.append(subepochPerEpoch)
		listOfEpochsPerFullInferList.append(epochsBetweenEachFullInfer)

		experimentCount +=1

	return (listOfTrainingSessionNames, listOfSubepochsPerEp, listOfEpochsPerFullInferList)

def plotAccuracy(inLogsList, inClassFromEachLogList, iterationsForRollingAverageAccuracy = 20) :

	if inLogsList :
		logFiles = inLogsList
		classToExtractFromEachLogFile = inClassFromEachLogList
		(sessionNamesList, subepochsPerEpList, epochsPerFullInferList) = parseVariablesFromOfTrainingSessionsFromListOfLogs(inLogsList)
		legendList = sessionNamesList
	else :
		#This is a hack. In case you run the script without any options, it will get here. And run for these hard-coded values.
		logFiles = [
			]
		classToExtractFromEachLogFile = [
						1,
						1
						]
		legendList = [	"Experiment-1",
				"Experiment-2"
				]
		subepochsPerEpList = [20]*len(logFiles)
		epochsPerFullInferList = [5]*len(logFiles)


		if len(logFiles) == 0 :
			print("No input log files. Exiting."); exit(1)

	maxNumOfSubepsPerEpInExperiments = max(subepochsPerEpList)
	maxNumOfEpsBetweenFullInfInExperiments = max(epochsPerFullInferList)
	maxNumOfEpsDurationOfExps = 0

	colors = ["r","g","b","c","m","k"]
	linestyles = ['-', '--', ':', '_', '-.']

	subplotTitles = [ ["Mean Accuracy", "Sensitivity", "Specificity", "DSC (samples)", "DSC (full-segm)"],
			  ["Mean Accuracy", "Sensitivity", "Specificity", "DSC (samples)", "DSC (full-segm)"]
			]


	allAccuracyExperiments = [[],[]]

	for logFile_i in xrange(0, len(logFiles)) :
		for validation0orTraining1 in [0,1] :
			if validation0orTraining1 == 0 : #validation
				accuraciesForAllMeanPosNegDice = [0,0,0,0,0]
			else : #training
				accuraciesForAllMeanPosNegDice = [0,0,0,0]
			for meanPosNegDice1_i in xrange(0,4) :
				accuraciesForAllMeanPosNegDice[meanPosNegDice1_i] = getListWithAccuracyFromLogMulticlass(logFiles[logFile_i], validation0orTraining1, meanPosNegDice1_i, classToExtractFromEachLogFile[logFile_i])
				accuraciesForAllMeanPosNegDice[meanPosNegDice1_i] = moving_average(accuraciesForAllMeanPosNegDice[meanPosNegDice1_i], iterationsForRollingAverageAccuracy)
			#Validation only, the DSC score from Full Inference. No Rolling average.
			if validation0orTraining1 == 0 : #validation
				accuraciesForAllMeanPosNegDice[4] = [0] + getListWithDiceFromLogMulticlass(logFiles[logFile_i], classToExtractFromEachLogFile[logFile_i])
				accuraciesForAllMeanPosNegDice[4] = moving_average(accuraciesForAllMeanPosNegDice[4], 1)
			allAccuracyExperiments[validation0orTraining1].append(accuraciesForAllMeanPosNegDice)


	fontSizeSubplotTitles = 14; fontSizeXTickLabel = 12; fontSizeYTickLabel = 12; fontSizeXAxisLabel = 12; fontSizeYAxisLabel = 14; linewidthInPlots = 1.5;
	legendFontSize = 12; legendNumberOfColumns = 8;
	#plt.close('all')
	#plt.subplots(rows,columns): returns: (figure, axes), where axes is an array, one element for each subplot, of rows and columns as I specify!
	fig, axes = plt.subplots(2, 5, sharex=False, sharey=False)
	inchesForMainPlotPart = 7; inchesForLegend = 0.6; percForMain = inchesForMainPlotPart*1.0/(inchesForMainPlotPart+inchesForLegend); percForLegend = 1.-percForMain
	fig.set_size_inches(15,inchesForMainPlotPart+inchesForLegend); #changes width/height of the figure. VERY IMPORTANT
	fig.set_dpi(100); #changes width/height of the figure.

	fig.subplots_adjust(left=0.05, bottom = 0.1*percForMain + percForLegend, right=0.98, top=0.92*percForMain+percForLegend, wspace=0.25, hspace=0.4*percForMain)
	fig.canvas.set_window_title(os.path.basename(__file__))
	fig.suptitle(os.path.basename(__file__) + ": Rolling Average for #"+ str(iterationsForRollingAverageAccuracy)+" iterations. For each experiments, Class plotted: " + str(classToExtractFromEachLogFile) + ", Subepochs per Epoch: " + str(subepochsPerEpList) + ", Epochs between Full-Segmentations: " + str(epochsPerFullInferList), fontsize=8)#, fontweight='bold')

	for valOrTrain_i in xrange(0, len(allAccuracyExperiments)) :
		for valOrTrainExperiment_i in xrange(0, len(allAccuracyExperiments[valOrTrain_i])) :
			valOrTrainExperiment = allAccuracyExperiments[valOrTrain_i][valOrTrainExperiment_i]
			for meanPosNegDice1_i in xrange(0, len(valOrTrainExperiment)) :
				numberOfSubsPerEpoch = subepochsPerEpList[valOrTrainExperiment_i]
				numberOfEpsBetweenFullInf = epochsPerFullInferList[valOrTrainExperiment_i]

				if meanPosNegDice1_i <> 4 : #Not for DSC full inference.
					numberOfSubepochsRan = len(valOrTrainExperiment[meanPosNegDice1_i])
					numberOfEpochsRan = numberOfSubepochsRan*1.0/numberOfSubsPerEpoch
					maxNumOfEpsDurationOfExps = maxNumOfEpsDurationOfExps if maxNumOfEpsDurationOfExps >= numberOfEpochsRan else numberOfEpochsRan
					xIter = np.linspace(0, numberOfEpochsRan, numberOfSubepochsRan, endpoint=True) #endpoint=True includes it as the final point.
				else : #DSC Full inference.
					#The -1 here is because for the DSC I previously prepended a 0 element (at 0th iteration).
					numberOfFullInfRanPlusOneAt0 = len(valOrTrainExperiment[meanPosNegDice1_i])
					numberOfEpochsRan = (numberOfFullInfRanPlusOneAt0 - 1) * numberOfEpsBetweenFullInf
					maxNumOfEpsDurationOfExps = maxNumOfEpsDurationOfExps if maxNumOfEpsDurationOfExps >= numberOfEpochsRan else numberOfEpochsRan
					xIter = np.linspace(0, numberOfEpochsRan, numberOfFullInfRanPlusOneAt0, endpoint=True)

				axes[valOrTrain_i, meanPosNegDice1_i].plot(xIter, valOrTrainExperiment[meanPosNegDice1_i], color = colors[valOrTrainExperiment_i%len(colors)], linestyle = linestyles[valOrTrainExperiment_i/len(colors)], label=legendList[valOrTrainExperiment_i], linewidth=linewidthInPlots)
				axes[valOrTrain_i, meanPosNegDice1_i].set_title(subplotTitles[valOrTrain_i][meanPosNegDice1_i], fontsize=fontSizeSubplotTitles, y=1.022)
				axes[valOrTrain_i, meanPosNegDice1_i].yaxis.grid(True, zorder=0)
				axes[valOrTrain_i, meanPosNegDice1_i].set_xlim([0,maxNumOfEpsDurationOfExps])
				axes[valOrTrain_i, meanPosNegDice1_i].set_xlabel('Epoch', fontsize=fontSizeXAxisLabel)


	for train0AndValidation1 in [0,1]:
		for axis in axes[train0AndValidation1]:
			#plt.setp(axis.get_xticklabels(), rotation='horizontal', fontsize=fontSizeXTickLabel) #In case I want something vertical, this is how I change it.
			#plt.setp(axis.get_yticklabels(), rotation='horizontal', fontsize=fontSizeYTickLabel)

			#axis.xticks(xCustomTicksEpochs, labels, rotation='vertical') If I d like to also give labels, this is how I do it.

			#In case I want to manually define where to have xticks.
			#axis.set_xticks(xCustomTicksSubepochs)
			#axis.set_xticklabels(xCustomTicksLabelsEpochs, fontsize = fontSizeXTickLabel)
			axis.yaxis.grid(True, linestyle='--', which='major', color='black', alpha=1.0)
			axis.tick_params(axis='y', labelsize=fontSizeYTickLabel)

	axes[0,0].set_ylim(0., 1.); axes[0,1].set_ylim(0., 1.); axes[0,2].set_ylim(0., 1.); axes[0,3].set_ylim(0., 1.); axes[0,4].set_ylim(0., 1.)
	axes[1,0].set_ylim(0., 1.); axes[1,1].set_ylim(0., 1.); axes[1,2].set_ylim(0., 1.); axes[1,3].set_ylim(0., 1.); axes[1,4].set_ylim(0., 1.)

	axes[0,0].set_ylabel('Validation', fontsize=fontSizeYAxisLabel)
	axes[1,0].set_ylabel('Training', fontsize=fontSizeYAxisLabel)

	#axes[0,0].legend()
	"""
	Moving the legend-box:
	- You grab a subplot. (depending on the axis that you ll use at: axis.legend(...))
	- Then, you specify with loc=, the anchor of the LEGENDBOX that you will move in relation to the BOTTOM-LEFT corner of the above axis..
		loc = 'upper right' (1), 'upper left' (2), 'lower left' (3), 'lower right' (4)
	- bbox_to_anchor=(x-from-left, y-from-bottom, width, height). x and y can be negatives. Specify how much to move legend's loc from the bottom left corner of the axis.
		x, y, width and height are floats, giving the percentage of the AXIS's size. Eg x=0.5, y=0.5 moves it at the middle of the subplot.
	"""
	leg = axes[1,0].legend(loc='upper left', bbox_to_anchor=(0., -.25, 0., 0.),#(0., -1.3, 1., 1.),
           		ncol=legendNumberOfColumns, borderaxespad=0. , fontsize=legendFontSize, labelspacing = 0., columnspacing=1.0)#mode="expand",
	#Make the lines in the legend wider.
	for legobj in leg.legendHandles:
		legobj.set_linewidth(6.0)

	#plt.savefig('./foo.pdf', dpi=fig.dpi)#, bbox_inches='tight')

	argv = sys.argv #name-of-script is the first elemet.
	if len(argv) > 1 and argv[1] == "-np" :
		print "Argument -n given. Not rendering figure. Just saved to file."
	else :
		plt.show()


def getStringOfTheListThatForSureStartsInThisLineButMayEndInAnother(restOfTheLineAfterTheEndOfTheWantedPattern, f) :
	#A list starts in the currently already-read line  = restOfTheLineAfterTheEndOfTheWantedPattern. But it may be ending ] in this same line, or one of the next ones.
	#If it does not end in this one, keep reading lilnes from file f, until you find its end. Put the whole list, including [] into the returned resulting string.
	#The file will be read UP UNTIL the line where the list ] ends. This may be the already read line (ie, dont read any more).
	indexWhereListStartsInThisLine = restOfTheLineAfterTheEndOfTheWantedPattern.find("[")
	indexWhereListEndsInThisLine = restOfTheLineAfterTheEndOfTheWantedPattern.find("]")
	if indexWhereListEndsInThisLine > -1 :
		theListInString = restOfTheLineAfterTheEndOfTheWantedPattern[ indexWhereListStartsInThisLine : indexWhereListEndsInThisLine ]
		endOfListFound = True
	else :
		theListInString = restOfTheLineAfterTheEndOfTheWantedPattern[ indexWhereListStartsInThisLine : ]
		endOfListFound = False
				
	while endOfListFound == False :
		newLine = f.readline()
		if newLine :
			indexWhereListEndsInThisLine = newLine.find("]")
			if indexWhereListEndsInThisLine > -1 :
				theListInString += newLine[ : indexWhereListEndsInThisLine ]
				endOfListFound = True
			else :
				theListInString += newLine[ : ]

	theListInString = theListInString.strip() #to get trailing whitespace off.
	return theListInString


def getAListOfStringNumbersAfterSplittingThemFromAStringListWithStringNumbers(theListInString, splittingChar) :
	#gets a string that is a STRING LIST with inside STRING-NUMBERS. It returns an actual list, where the elements are the string-numbers.
	numbersOfListInString = theListInString.strip()
	numbersOfListInString = numbersOfListInString.lstrip('[')
	numbersOfListInString = numbersOfListInString.rstrip(']')
	#print "numbersOfListInString=",numbersOfListInString

	#parse the numbers and put them in a list to return.
	if splittingChar=="" :
		listOfstringNumbersSplitted = numbersOfListInString.split()
	else :
		listOfstringNumbersSplitted = numbersOfListInString.split(splittingChar)

	return listOfstringNumbersSplitted




def getListWithDiceFromLogMulticlass(nameAndPathOfLogFile, class_i) :
	accList = []

	accuracyString = "ACCURACY: (Validation) The Per-Class average DICE Coefficients over all subjects are:"
	dice3String = "DICE3="

	f = open(nameAndPathOfLogFile, 'r')
		
	newLine = f.readline()	
	while newLine :

		indexWhereAccuracyStringStarts = newLine.rfind(accuracyString)
		if indexWhereAccuracyStringStarts > -1 :
			restOfLine = newLine[indexWhereAccuracyStringStarts+len(accuracyString):]
			indexWhereDice3Ends = restOfLine.rfind(dice3String)

			if indexWhereDice3Ends > -1 :
				restOfTheLineAfterTheEndOfTheWantedPattern = restOfLine[indexWhereDice3Ends+len(dice3String):]
				theAccuracyForAllClassesInStringList = getStringOfTheListThatForSureStartsInThisLineButMayEndInAnother(restOfTheLineAfterTheEndOfTheWantedPattern, f)
				
				listOfstringNumbersSplitted = getAListOfStringNumbersAfterSplittingThemFromAStringListWithStringNumbers(theAccuracyForAllClassesInStringList, "")
				
				#check if it is the normal validation/testing or the pseudotesting.
				if not listOfstringNumbersSplitted[0].startswith("999") : #It is PseudoTesting, continue. Forget about this entirely.
					
					accuracyForTheWantedClassInString = listOfstringNumbersSplitted[class_i]
					#check if it is the normal validation/testing or the pseudotesting.

					parseFloatAccuracyNumber = float(accuracyForTheWantedClassInString)
					if parseFloatAccuracyNumber < 1. : #Check if the dice was calculated fine. If > 1.0, it was probably 999 or something, so, just put it to 0/
						accList.append(parseFloatAccuracyNumber)
					else :
						accList.append(0)

		newLine = f.readline()
	return accList


def getListWithAccuracyFromLogMulticlass(nameAndPathOfLogFile, validation0orTraining1, mean0pos1neg2dice3, class_i) :
	valList = []

	if validation0orTraining1 == 0 :
		validationOrTrainingString = "VALIDATION:"
	elif validation0orTraining1 == 1 :
		validationOrTrainingString = "TRAINING:"

	classPrefixString = "Class-" + str(class_i) + ": "

	if mean0pos1neg2dice3 == 0 : #looking for mean accuracy
		patternToLookFor = classPrefixString + "the mean accuracy of each subepoch was: "
	elif mean0pos1neg2dice3 == 1 : #looking for pos accuracy
		patternToLookFor = classPrefixString + "the mean accuracy of each subepoch, for Positive Samples(voxels) was: "
	elif mean0pos1neg2dice3 == 2 : #looking for neg accuracy
		patternToLookFor = classPrefixString + "the mean accuracy of each subepoch, for Negative Samples(voxels) was: "
	elif mean0pos1neg2dice3 == 3 : #looking for dice on samples
		patternToLookFor = classPrefixString + "the mean Dice of each subepoch was: "

	f = open(nameAndPathOfLogFile, 'r')
		
	previousValueOfTheVariableInTheTimeSerie = 0 #This is useful in the case I get a not-valid number, to just use the previous one.
	newLine = f.readline()
	while newLine :

		indexWhereValOrAccStringEnds = newLine.rfind(validationOrTrainingString)
		if indexWhereValOrAccStringEnds > -1 :
			restOfLine = newLine[indexWhereValOrAccStringEnds+len(validationOrTrainingString):]
			indexWherePatternEnds = restOfLine.rfind(patternToLookFor)

			if indexWherePatternEnds > -1 :
				restOfTheLineAfterTheEndOfTheWantedPattern = restOfLine[indexWherePatternEnds+len(patternToLookFor):]
				#parse the list from the string:
				theListInString = getStringOfTheListThatForSureStartsInThisLineButMayEndInAnother(restOfTheLineAfterTheEndOfTheWantedPattern, f)

				listOfstringNumbersSplitted = getAListOfStringNumbersAfterSplittingThemFromAStringListWithStringNumbers(theListInString, "")

				for stringNumber in listOfstringNumbersSplitted :
					stringNumberStrippedOfWhiteSpace = stringNumber.strip()

					parseFloatNumber = float(stringNumberStrippedOfWhiteSpace)

					parseFloatNumber = parseFloatNumber if parseFloatNumber <= 1. else previousValueOfTheVariableInTheTimeSerie
					valList.append(parseFloatNumber)
					previousValueOfTheVariableInTheTimeSerie = parseFloatNumber

		newLine = f.readline()

	return valList



def moving_average(a, n=20) :
    cumsum = np.cumsum(a, dtype=float)

    tempRetComplete = cumsum[n:] - cumsum[:-n]

    retCompletePart = tempRetComplete / n

    # Also calculate the rollAverage for the first n-1 elements, even if it's calculated with less than n elements
    retIncompletePart = cumsum[:n]
    for i in range(0, len(retIncompletePart)) :
	retIncompletePart[i] = retIncompletePart[i] / (i+1)

    return np.concatenate((retIncompletePart, retCompletePart), axis = 0)





if __name__ == '__main__':

	myArgParser = setupArgParser()
	args = myArgParser.parse_args()
	if len(sys.argv) == 1:
		print("For help on the usage of this script, please use the option -h.");

	#Checks by itself that at least one argument is given, if this option is chosen.
	
	
	if args.inp_logs_list :
		inLogsList = args.inp_logs_list
		if args.inp_class_from_each_log_list :
			inClassFromEachLogList = args.inp_class_from_each_log_list
			if len(inLogsList) <> len(inClassFromEachLogList) :
				print("ERROR: The number of log files given is not the same with the number of arguments that specify which class's accuracy to plot from each log file. Give the same number of arguments. Exiting."); exit(1)
		else :
			#Default class when none given
			inClassFromEachLogList = len(inLogsList)*[0]
	else :#if the option -logs is not given at all: Hack. Run the below. And the variables will be passed any hard-coded values.
		inLogsList = None
		inClassFromEachLogList = None

	iterationsForRollingAverageAccuracy = 1
	plotAccuracy(inLogsList, inClassFromEachLogList, iterationsForRollingAverageAccuracy)








