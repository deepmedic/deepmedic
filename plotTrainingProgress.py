# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

'''
This script parses training logs and plots accuracy metrics (mean accuracy, sensitivity, specificity, DSC over samples and DSC of full segmentation of validation subjects).

Last update: 16 June 2016
'''

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
import sys
import argparse

import re

NA_PATTERN = "N/A"
SESSION_NAME_PATTERN = "Session\'s name ="
SUBEPS_PER_EP_PATTERN = "Number of Subepochs per epoch ="
NUM_EPS_BETWEEN_FULLINF_PATTERN = "Perform Full-Inference on Val. cases every that many epochs ="
NUM_OF_CLASSES_PATTERN = "Number of Classes (incl. background) (from cnnModel) ="

# Patterns for extracting detailed metrics:
VALIDATION_PATT = "VALIDATION:"
TRAINING_PATT = "TRAINING:"
CLASS_PREFIX_PATT = "Class-"
MEANACC_SENTENCE = "mean accuracy of each subepoch:"
SENS_SENTENCE = "mean sensitivity of each subepoch:"
SPEC_SENTENCE = "mean specificity of each subepoch:"
DSC_SAMPLES_SENTENCE = "mean Dice of each subepoch:"
# Patterns for extracting basic metrics
OVERALLCLASS_PATT = "Overall"
MEANACC_OVERALL_SENTENCE = "mean accuracy of each subepoch:"
COST_OVERALL_SENTENCE = "mean cost of each subepoch:"

def setupArgParser() :
    parser = argparse.ArgumentParser( prog='plotTrainingProgress', formatter_class=argparse.RawTextHelpFormatter,
    description='''This script parses training logs and plots accuracy metrics (mean accuracy, sensitivity, specificity, DSC over samples, DSC of full segmentation of validation subjects).''')
    parser.add_argument("log_files", nargs='+', type=str, help="Paths to training logs. More than one log can be given, to plot progress of multiple experiments. \nFormat: python ./plotTrainingProgress.py log1.txt log2.txt logs3.txt ...")
    parser.add_argument("-d", "--detailed", dest='detailed_plot', action='store_true', help="By default, only \"overall\" mean empirical accuracy is plotted. Provide this option for a more detailed and \"class-specific\" plot.\nMetrics plotted: mean accuracy, sensitivity, specificity, DSC on samples and DSC on fully-segmented validation subjects.\n***IMPORTANT***\n\"Class-specific\" metrics of the more detailed plot are computed in a \"One-Class Vs All-Others\" fashion!\nIn *Multi-Class* problems, \"overall\" accuracy of the basic plot and \"class-specific\" accuracy of the detailed plot differ significantly because of this!\nOverall accuracy of basic plot: Number of voxels predicted with correct class / number of all voxels.\nClass-specific accuracy of detailed plot: (True Positives + True Negatives with respect to \"the specified class\") / number of all voxels.\n\t>> i.e. voxels predicted with any other class are all considered similar, eg as background.")
    parser.add_argument("-c", "--classes", dest='classes_to_plot', nargs='+', type=int, help="Use only when --detailed plot is activated.\nSpecify for which class(es) to plot metrics.\nFormat: -c 2 |OR| -c 0 0 2 ... (Default: class-0 will be plotted from each log.) \n*NOTE* Plotted metrics for Class-0 correspond to \"whole\" Foreground, although Label-0 in the NIFTIs is supposed to be Background. We consider it more useful.\nUsage cases:\nA single class specified: All given log files will be parsed to plot corresponding training progress for this class. \nMultiple classes and one log file: Log will be parsed for all given classes in order to plot their progress. \nMultiple classes and multiple logs: They will be matched one-to-one for plotting. For this, same number of classes and logs should be given.")
    parser.add_argument("-m", "--movingAv", dest='moving_average', type=int, default=1, help="Plotted values are smoothed with a moving average. Specify over how many values (subepochs) it should extend. \nFormat: -m 20 (Default: 1)\n*NOTE* DSC from full-segmentation of validation images is not smoothed.")
    parser.add_argument("-s", "--saveFigure", dest='save_figure', action='store_true', help="Use to make the script save the figure at the current folder. Takes no arguments.")
    return parser

def getNameOfLogFileWithoutEnding(filePathToLog):
    filenameOfLog = os.path.basename(filePathToLog)
    (filenameWithoutExt, extension1) = os.path.splitext(filenameOfLog)
    return filenameWithoutExt
def getSubepochsPerEpoch(pathToLog) :
    lineWithPattern = getFirstLineInLogWithCertainPattern(pathToLog, SUBEPS_PER_EP_PATTERN)
    if lineWithPattern == None : return None
    return getIntFromStr( lineWithPattern[ lineWithPattern.find(SUBEPS_PER_EP_PATTERN) + len(SUBEPS_PER_EP_PATTERN) : ] )
def getEpochsBetweenFullInf(pathToLog) :
    lineWithPattern = getFirstLineInLogWithCertainPattern(pathToLog, NUM_EPS_BETWEEN_FULLINF_PATTERN)
    if lineWithPattern == None : return None
    return getIntFromStr( lineWithPattern[ lineWithPattern.find(NUM_EPS_BETWEEN_FULLINF_PATTERN) + len(NUM_EPS_BETWEEN_FULLINF_PATTERN) : ] )
def getNumberOfClasses(pathToLog) :
    lineWithPattern = getFirstLineInLogWithCertainPattern(pathToLog, NUM_OF_CLASSES_PATTERN)
    if lineWithPattern == None : return None
    return getIntFromStr( lineWithPattern[ lineWithPattern.find(NUM_OF_CLASSES_PATTERN) + len(NUM_OF_CLASSES_PATTERN) : ] )

def getIntFromStr(string1) : #may be unstripped
    return int(string1.strip())
def getFloatFromStr(string1) : #may be unstripped
    return float(string1.strip())

def parseLogFileAndGetVariablesOfInterest(pathToLog) :
    experimentName = None; subepochsPerEpoch=None; epochsBetweenEachFullInfer=None
    experimentName = getNameOfLogFileWithoutEnding(pathToLog)
    subepochsPerEpoch = getSubepochsPerEpoch(pathToLog)
    epochsBetweenEachFullInfer = getEpochsBetweenFullInf(pathToLog)
    return (experimentName, subepochsPerEpoch, epochsBetweenEachFullInfer)

def parseVariablesOfTrainingSessionsFromListOfLogs(inLogsList) :
    listOfExperimentsNames = []; listOfSubepochsPerEpFromEachLog = []; listOfEpochsPerFullInferFromEachLog = []
    for log_i in xrange(len(inLogsList)) :
        #Get variables from the logfile.
        (experimentName, subepochsPerEpoch, epochsBetweenEachFullInfer) = parseLogFileAndGetVariablesOfInterest(inLogsList[log_i])
        #In case the name was not found:
        if not experimentName : experimentName = "TrainingSession-" + str(log_i)
        if not subepochsPerEpoch : subepochsPerEpoch = 20 #default
        if not epochsBetweenEachFullInfer : epochsBetweenEachFullInfer = 1 #default
        listOfExperimentsNames.append(experimentName)
        listOfSubepochsPerEpFromEachLog.append(subepochsPerEpoch)
        listOfEpochsPerFullInferFromEachLog.append(epochsBetweenEachFullInfer)
    return (listOfExperimentsNames, listOfSubepochsPerEpFromEachLog, listOfEpochsPerFullInferFromEachLog)

def makeLegendList(listOfExperimentsNames, classesFromEachLogFile) :
    legendList = []
    for exper_i in xrange(len(listOfExperimentsNames)) :
        for classFromExper in classesFromEachLogFile[exper_i] :
            legendList.append( listOfExperimentsNames[exper_i] + "-Class" + str(classFromExper) )
    return legendList

def makeHelperVariablesPerExperiment(logFiles, classesFromEachLogFile, subepochsPerEpFromEachLog, epochsPerFullInferFromEachLog) :
    subepochsPerEpOfExpers = []
    epochsPerFullInferOfExpers = []
    for logFile_i in xrange(len(logFiles)) :
        for classForLogFile_i in xrange(len(classesFromEachLogFile[logFile_i])) :
            subepochsPerEpOfExpers.append(subepochsPerEpFromEachLog[logFile_i]) # Essentially just doublicating the same entry again and again for all classes of same logfile/experiment.
            epochsPerFullInferOfExpers.append(epochsPerFullInferFromEachLog[logFile_i])
    return (subepochsPerEpOfExpers, epochsPerFullInferOfExpers)

def getStringOfTheListThatForSureStartsInThisLineButMayEndInAnother(restOfTheLineAfterTheEndOfTheWantedPattern, f) :
    #A list starts in the currently already-read line  = restOfTheLineAfterTheEndOfTheWantedPattern. But it may be ending ] in this same line, or one of the next ones.
    #If it does not end in this one, keep reading lilnes from file f, until you find its end. Put the whole list, including [] into the returned resulting string.
    #The file will be read UP UNTIL the line where the list ] ends. This may be the already read line (ie, dont read any more).
    indexWhereListStartsInThisLine = restOfTheLineAfterTheEndOfTheWantedPattern.find("[")
    indexWhereListEndsInThisLine = restOfTheLineAfterTheEndOfTheWantedPattern.find("]")
    if indexWhereListEndsInThisLine > -1 :
        theListInString = restOfTheLineAfterTheEndOfTheWantedPattern[ indexWhereListStartsInThisLine : indexWhereListEndsInThisLine+1 ]
        endOfListFound = True
    else :
        theListInString = restOfTheLineAfterTheEndOfTheWantedPattern[ indexWhereListStartsInThisLine : ]
        endOfListFound = False
        
    while endOfListFound == False :
        newLine = f.readline()
        if newLine :
            indexWhereListEndsInThisLine = newLine.find("]")
            if indexWhereListEndsInThisLine > -1 :
                theListInString += newLine[ : indexWhereListEndsInThisLine+1 ]
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


def getFirstLineInLogWithCertainPattern(filePathToLog, pattern) :
    foundLine = None
    f = open(filePathToLog, 'r')
    newLine = f.readline()
    while newLine :
        if newLine.find(pattern) > -1 :
            foundLine = newLine
            break
        newLine = f.readline()
    f.close()
    return foundLine # Returns None if not found.


def getRegExprForParsingMetric(validation0orTraining1, basic0detailed1, class_i, intSpecifyingMetric01234) :
    validationOrTrainingString = VALIDATION_PATT if validation0orTraining1 == 0 else TRAINING_PATT
    if basic0detailed1 == 0 : # basic plotting
        classPrefixString = OVERALLCLASS_PATT
        if intSpecifyingMetric01234 == 0 : #looking for mean accuracy
            sentenceToLookFor = MEANACC_OVERALL_SENTENCE
        elif intSpecifyingMetric01234 == 1 : #looking for cost
            sentenceToLookFor = COST_OVERALL_SENTENCE
    else : #detailed plotting
        classPrefixString = CLASS_PREFIX_PATT + str(class_i)
        if intSpecifyingMetric01234 == 0 : #looking for mean accuracy
            sentenceToLookFor = MEANACC_SENTENCE
        elif intSpecifyingMetric01234 == 1 : #looking for pos accuracy
            sentenceToLookFor = SENS_SENTENCE
        elif intSpecifyingMetric01234 == 2 : #looking for neg accuracy
            sentenceToLookFor = SPEC_SENTENCE
        elif intSpecifyingMetric01234 == 3 : #looking for dice on samples
            sentenceToLookFor = DSC_SAMPLES_SENTENCE
    regExp1 = ".*" + validationOrTrainingString + ".*" + classPrefixString + ".*" + sentenceToLookFor
    return (regExp1, sentenceToLookFor)


def getListOfAccNumbersFromListOfStrNumbersAvoidingNotAppl(listOfstringNumbers, previousValueOfTheVariableInTheTimeSerie) :
    listOfAccNumbers = []
    for stringNumber in listOfstringNumbers :
        stringNumberStrippedOfWhiteSpace = stringNumber.strip()
        parseFloatNumber = float(stringNumberStrippedOfWhiteSpace) if stringNumberStrippedOfWhiteSpace <> NA_PATTERN else previousValueOfTheVariableInTheTimeSerie
        previousValueOfTheVariableInTheTimeSerie = parseFloatNumber
        listOfAccNumbers.append(parseFloatNumber)
    return listOfAccNumbers


def movingAverage(a, n=1) :
    cumsum = np.cumsum(a, dtype=float)
    tempRetComplete = cumsum[n:] - cumsum[:-n]
    retCompletePart = tempRetComplete / n
    # Also calculate the rollAverage for the first n-1 elements, even if it's calculated with less than n elements
    retIncompletePart = cumsum[:n]
    for i in range(0, len(retIncompletePart)) :
        retIncompletePart[i] = retIncompletePart[i] / (i+1)
    return np.concatenate((retIncompletePart, retCompletePart), axis = 0)


def movingAverageConv(a, window_size=1) :
    if not a : return a
    window = np.ones(int(window_size))
    result = np.convolve(a, window, 'full')[ : len(a)] # Convolve full returns array of shape ( M + N - 1 ).
    slotsWithIncompleteConvolution = min(len(a), window_size-1)
    result[slotsWithIncompleteConvolution:] = result[slotsWithIncompleteConvolution:]/float(window_size)
    if slotsWithIncompleteConvolution > 1 :
        divisorArr = np.asarray(range(1, slotsWithIncompleteConvolution+1, 1), dtype=float)
        result[ : slotsWithIncompleteConvolution] = result[ : slotsWithIncompleteConvolution] / divisorArr
    return result


################################# PARSING the reported measurements from logs (Optimized for one pass per log) #####################################
# There will be ugly code in here.

def applyMovingAverageToAllButDscFullSeg(detailedPlotBool, measuredMetricsFromAllExperiments, movingAverSubeps ) :
    for valOrTrainExperiments in measuredMetricsFromAllExperiments :
        for experimentToPlot in valOrTrainExperiments : # Number of logs X Classes
            for metric_i in xrange(len(experimentToPlot)) :
                if detailedPlotBool and metric_i == 4 : # We are plotting detailed metrics and this is the DSC-Full-Seg, which we do not smooth with movingAverage
                    continue
                experimentToPlot[metric_i] = movingAverageConv(experimentToPlot[metric_i], movingAverSubeps)
    return measuredMetricsFromAllExperiments


def checkIfLineMatchesAnyRegExpr(string, regExprForEachClassAndMetric) :
    for val0orTrain1 in xrange(len(regExprForEachClassAndMetric)) : #[0,1]
        for class_i in xrange(len(regExprForEachClassAndMetric[val0orTrain1])) :
            for metric_i in xrange(len(regExprForEachClassAndMetric[val0orTrain1][class_i])) :
                regExp1 = regExprForEachClassAndMetric[val0orTrain1][class_i][metric_i]
                matchObj = re.match( regExp1, string, re.M|re.I)
                if matchObj :
                    return matchObj, val0orTrain1, class_i, metric_i 
    return None, None, None, None # No regular expression matches this string.


# THE DATA STRUCTURES HERE are very ugly, with one extra useless dimension, to be consistent with the "detailed" version. For future merging. See parseDetailedMetricsFromThisLog, which was written first!
def parseBasicMetricsFromThisLog( logFile, movingAverSubeps ) :
    ### Initially just form a data structure with the regular expressions for each val/train, class and metric. ###
    # [0] val, [1] train
    # Each has one sublist, because I only have 1 class in the basic-plotting! ( Extra dimension kept just so that this function is consistent with the "detailed" version. For future merging.)
    # Each class-sublist has 1 entry, one for Acc
    regExprForEachClassAndMetric = [ [ [] ], [ [] ] ] #[0] val, [1] train
    sentencesToLookForEachClassAndMetric = [ [ [] ], [ [] ] ] #[0] val, [1] train
    
    for val0orTrain1 in [0,1] :
        (regExprForMeanAcc, sentenceForMeanAcc) = getRegExprForParsingMetric(val0orTrain1, 0, None, 0)
        regExprForEachClassAndMetric[val0orTrain1][0].append( regExprForMeanAcc )
        sentencesToLookForEachClassAndMetric[val0orTrain1][0].append( sentenceForMeanAcc )
        
    ### Form the data structure where we ll put all the measurements from this logfile for each val/train, class and metric. ###
    #[0] val, [1] train. Each has one sublist, because in basic I have only 1 class to be plotted. The class-sublist has 1 sublist, because here I have only 1 plotted metric.
    measurementsForEachClassAndMetric = [ [ [] ], [ [] ] ]
    previousMeasurementForEachClassAndMetric = [ [ [] ], [ [] ] ] #This is useful in the case I get a not-valid number, to just use the previous one.
    for val0orTrain1 in [0,1] :
        for metric_i in xrange(1) : 
            measurementsForEachClassAndMetric[val0orTrain1][0].append([]) # Add a sublist in the class, per metric.
            previousMeasurementForEachClassAndMetric[val0orTrain1][0].append(0)
            
    ### Read the file and start parsing each line.
    f = open(logFile, 'r')
    newLine = f.readline()
    while newLine :
        matchObj, matchVal0Train1, matchClass_i, matchMetric_i = checkIfLineMatchesAnyRegExpr(newLine, regExprForEachClassAndMetric)
        
        if matchObj : #matched the reg-expression for Acc
            sentenceToLookFor = sentencesToLookForEachClassAndMetric[matchVal0Train1][matchClass_i][matchMetric_i]
            restOfLineAfterPattern = newLine[ newLine.find(sentenceToLookFor)+len(sentenceToLookFor) : ]
            theListInString = getStringOfTheListThatForSureStartsInThisLineButMayEndInAnother(restOfLineAfterPattern, f)
            listOfstringNumbersSplitted = getAListOfStringNumbersAfterSplittingThemFromAStringListWithStringNumbers(theListInString, "")
            
            previousMeasurementForClassAndMetric = previousMeasurementForEachClassAndMetric[matchVal0Train1][matchClass_i][matchMetric_i]
            listOfMeasurements = getListOfAccNumbersFromListOfStrNumbersAvoidingNotAppl(listOfstringNumbersSplitted, previousMeasurementForClassAndMetric)
            
            previousMeasurementForEachClassAndMetric[matchVal0Train1][matchClass_i][matchMetric_i] = listOfMeasurements[-1] # LHS use the list itself, not an intermediate immutable int-variable!
            measurementsForEachClassAndMetric[matchVal0Train1][matchClass_i][matchMetric_i] += listOfMeasurements
            
        newLine = f.readline()
        
    f.close()
    
    return ( measurementsForEachClassAndMetric[0], measurementsForEachClassAndMetric[1] )


# THIS IS A VERY UGLY FUNCTION because it has hardcoded 0/4/5 integers for each of the metric. I need to make an enumerated class for this!
def parseDetailedMetricsFromThisLog( logFile, classesFromThisLog, movingAverSubeps ) :
    ### Initially just form a data structure with the regular expressions for each val/train, class and metric. ###
    # [0] val, [1] train
    # Each has one sublist per class to be plotted
    # Each class-sublist has 4 entries, one for each of the plotted metrics Acc,Sens,Spec,DSC-samples. NOT FOR DSC-Full-Seg cause it's not reported per class/subepoch.
    regExprForEachClassAndMetric = [ [], [] ] #[0] val, [1] train
    sentencesToLookForEachClassAndMetric = [ [], [] ] #[0] val, [1] train
    regExprForDscFullSeg = ".*ACCURACY:.*Validation.*The Per-Class average DICE Coefficients over all subjects are:.*DICE3="  # Special case, because it's not reported per class.
    sentenceForDscFullSeg = "DICE3="
    
    for val0orTrain1 in [0,1] :
        for classInt in classesFromThisLog :
            # mean acc, sens, spec, dsc samples, dsc full.
            regExprForClassAllMetrics = [0,0,0,0]
            sentencesForClassAllMetrics = [0,0,0,0]
            for metric_i in xrange(len(regExprForClassAllMetrics)) :
                (regExprForClassAllMetrics[metric_i], sentencesForClassAllMetrics[metric_i]) = getRegExprForParsingMetric(val0orTrain1, 1, classInt, metric_i)
            regExprForEachClassAndMetric[val0orTrain1].append( regExprForClassAllMetrics )
            sentencesToLookForEachClassAndMetric[val0orTrain1].append( sentencesForClassAllMetrics )
            
    ### Form the data structure where we ll put all the measurements from this logfile for each val/train, class and metric. ###
    #[0] val, [1] train. Each has one sublist per class to be plotted. Each class-sublist has 5 sublists, one for each of the plotted metrics.
    measurementsForEachClassAndMetric = [ [], [] ]
    previousMeasurementForEachClassAndMetric = [ [], [] ] #This is useful in the case I get a not-valid number, to just use the previous one.
    for val0orTrain1 in [0,1] :
        for class_i in xrange(len(classesFromThisLog)) :
            measurementsForEachClassAndMetric[val0orTrain1].append([]) # add a sublist in the val/train for each class
            previousMeasurementForEachClassAndMetric[val0orTrain1].append([])
            for metric_i in xrange(0,5) : # CAREFUL WITH THIS >> 5 <<
                measurementsForEachClassAndMetric[val0orTrain1][class_i].append([]) # Add a sublist in the class, per metric.
                if metric_i == 4 : # If it's the DSC-full-segm, add an initial 0 measurement!
                    measurementsForEachClassAndMetric[val0orTrain1][class_i][-1].append(0)
                previousMeasurementForEachClassAndMetric[val0orTrain1][class_i].append(0)
                
    ### Read the file and start parsing each line.
    f = open(logFile, 'r')
    newLine = f.readline()
    while newLine :
        matchObj, matchVal0Train1, matchClass_i, matchMetric_i = checkIfLineMatchesAnyRegExpr(newLine, regExprForEachClassAndMetric)
        
        if matchObj : #matched one of the reg-expressions for Acc/Sens/Spec/Dsc-Samples.
            sentenceToLookFor = sentencesToLookForEachClassAndMetric[matchVal0Train1][matchClass_i][matchMetric_i]
            restOfLineAfterPattern = newLine[ newLine.find(sentenceToLookFor)+len(sentenceToLookFor) : ]
            theListInString = getStringOfTheListThatForSureStartsInThisLineButMayEndInAnother(restOfLineAfterPattern, f)
            listOfstringNumbersSplitted = getAListOfStringNumbersAfterSplittingThemFromAStringListWithStringNumbers(theListInString, "")
            
            previousMeasurementForClassAndMetric = previousMeasurementForEachClassAndMetric[matchVal0Train1][matchClass_i][matchMetric_i]
            listOfMeasurements = getListOfAccNumbersFromListOfStrNumbersAvoidingNotAppl(listOfstringNumbersSplitted, previousMeasurementForClassAndMetric)
            
            previousMeasurementForEachClassAndMetric[matchVal0Train1][matchClass_i][matchMetric_i] = listOfMeasurements[-1] # LHS use the list itself, not an intermediate immutable int-variable!
            measurementsForEachClassAndMetric[matchVal0Train1][matchClass_i][matchMetric_i] += listOfMeasurements
            
        elif re.match( regExprForDscFullSeg, newLine, re.M|re.I) : # Did not match the reg-expressions for Acc/Sens/Spec/Dsc-Samples. But matches DSC-Full-Inf!
            sentenceToLookFor = sentenceForDscFullSeg
            restOfLineAfterPattern = newLine[ newLine.find(sentenceToLookFor)+len(sentenceToLookFor) : ]
            theListInString = getStringOfTheListThatForSureStartsInThisLineButMayEndInAnother(restOfLineAfterPattern, f)
            listOfstringNumbersSplitted = getAListOfStringNumbersAfterSplittingThemFromAStringListWithStringNumbers(theListInString, "")
            
            for class_i in xrange(len(classesFromThisLog)) :
                previousMeasurement = previousMeasurementForEachClassAndMetric[0][class_i][4] # get last value found for DSC of this class.
                dscForTheWantedClassInString = listOfstringNumbersSplitted[ classesFromThisLog[class_i] ] # Reported list with DICE is different than others and has a float per class.
                # listOfMeasurements = [float], just a list with one float in this case of DSC-full-seg.
                listOfMeasurements = getListOfAccNumbersFromListOfStrNumbersAvoidingNotAppl( [dscForTheWantedClassInString], previousMeasurement) # just returns the str number as float here.
                
                previousMeasurementForEachClassAndMetric[0][class_i][4] = listOfMeasurements[-1] # DONT replace LHS with any intermediate Int immutable variable!
                measurementsForEachClassAndMetric[0][class_i][4] += listOfMeasurements
                
        newLine = f.readline()
        
    f.close()
    
    return ( measurementsForEachClassAndMetric[0], measurementsForEachClassAndMetric[1] )


def optimizedParseMetricsFromLogs(logFiles, detailedPlotBool, classesFromEachLogFile, movingAverSubeps) :
    # Two rows, Validation and Accuracy
    # Each of these has as many sublists as the number of experiments (logFiles) X Classes!
    # Each of these sublists has a 5/4-entries sublist. Mean Accuracy/Sens/Spec/DSC-on-samples/DSC-from-full-segm-of-volumes (val only). OR just 1, if basic.
    measuredMetricsFromAllExperiments = [[],[]] #[0] validation, [1] training measurements.
    for logFile_i in xrange(0, len(logFiles)) :
        if not detailedPlotBool :
            ( measuredMetricsFromThisLogValidation,
            measuredMetricsFromThisLogTraining ) = parseBasicMetricsFromThisLog( logFiles[logFile_i], movingAverSubeps )
        else :
            ( measuredMetricsFromThisLogValidation,
            measuredMetricsFromThisLogTraining ) = parseDetailedMetricsFromThisLog( logFiles[logFile_i], classesFromEachLogFile[logFile_i], movingAverSubeps )
            
        measuredMetricsFromAllExperiments[0] += measuredMetricsFromThisLogValidation
        measuredMetricsFromAllExperiments[1] += measuredMetricsFromThisLogTraining
    measuredMetricsFromAllExperiments = applyMovingAverageToAllButDscFullSeg(detailedPlotBool, measuredMetricsFromAllExperiments, movingAverSubeps )
    return measuredMetricsFromAllExperiments

################################# END OF FUNCTIONS FOR THE PARSING OF MEASUREMENTS ########################################################


#========================================
#                PLOTTING
#========================================

def plotProgressBasic(measuredMetricsFromAllExperiments, legendList, movingAverSubeps, subepochsPerEpOfExpers, saveFigureBool) :
    colors = ["r","g","b","c","m","k"]
    linestyles = ['-', '--', ':', '_', '-.']
    
    subplotTitles = [ ["Mean Accuracy"], # Validation
                      ["Mean Accuracy"] # Training
                    ]
    
    fontSizeSubplotTitles = 14; fontSizeXTickLabel = 12; fontSizeYTickLabel = 12; fontSizeXAxisLabel = 12; fontSizeYAxisLabel = 14; linewidthInPlots = 1.5;
    legendFontSize = 12; legendNumberOfColumns = 8;
    #plt.close('all')
    #plt.subplots(rows,columns): returns: (figure, axes), where axes is an array, one element for each subplot, of rows and columns as I specify!
    numberOfMetricsPlotted = len(measuredMetricsFromAllExperiments[0][0])
    fig, axes = plt.subplots(2, numberOfMetricsPlotted, sharex=False, sharey=False)
    inchesForMainPlotPart = 7; inchesForLegend = 0.6; percForMain = inchesForMainPlotPart*1.0/(inchesForMainPlotPart+inchesForLegend); percForLegend = 1.-percForMain
    fig.set_size_inches(15,inchesForMainPlotPart+inchesForLegend); #changes width/height of the figure. VERY IMPORTANT
    fig.set_dpi(100); #changes width/height of the figure.
    
    fig.subplots_adjust(left=0.05, bottom = 0.1*percForMain + percForLegend, right=0.98, top=0.92*percForMain+percForLegend, wspace=0.25, hspace=0.4*percForMain)
    fig.canvas.set_window_title(os.path.basename(__file__))
    fig.suptitle(os.path.basename(__file__) + ": Moving Average over ["+ str(movingAverSubeps)+"] value. For each plotted experiment, Subepochs per Epoch: " + str(subepochsPerEpOfExpers), fontsize=8)#, fontweight='bold')
    
    maxNumOfEpsDurationOfExps = 0 # The number of epochs that the longest experiment lasted.
    
    for valOrTrain_i in xrange(0, len(measuredMetricsFromAllExperiments)) :
        for valOrTrainExperiment_i in xrange(0, len(measuredMetricsFromAllExperiments[valOrTrain_i])) :
            valOrTrainExperiment = measuredMetricsFromAllExperiments[valOrTrain_i][valOrTrainExperiment_i]
            for metric_i in xrange(0, len(valOrTrainExperiment)) :
                numberOfSubsPerEpoch = subepochsPerEpOfExpers[valOrTrainExperiment_i]
                
                numberOfSubepochsRan = len(valOrTrainExperiment[metric_i])
                numberOfEpochsRan = numberOfSubepochsRan*1.0/numberOfSubsPerEpoch
                maxNumOfEpsDurationOfExps = maxNumOfEpsDurationOfExps if maxNumOfEpsDurationOfExps >= numberOfEpochsRan else numberOfEpochsRan
                xIter = np.linspace(0, numberOfEpochsRan, numberOfSubepochsRan, endpoint=True) #endpoint=True includes it as the final point.
                
                axis = axes[valOrTrain_i] if numberOfMetricsPlotted == 1 else axes[valOrTrain_i, metric_i] # No 2nd index when subplot(X, 1, ...)
                axis.plot(xIter, valOrTrainExperiment[metric_i], color = colors[valOrTrainExperiment_i%len(colors)], linestyle = linestyles[valOrTrainExperiment_i/len(colors)], label=legendList[valOrTrainExperiment_i], linewidth=linewidthInPlots)
                axis.set_title(subplotTitles[valOrTrain_i][metric_i], fontsize=fontSizeSubplotTitles, y=1.022)
                axis.yaxis.grid(True, zorder=0)
                axis.set_xlim([0,maxNumOfEpsDurationOfExps])
                axis.set_xlabel('Epoch', fontsize=fontSizeXAxisLabel)
                
    for train0AndValidation1 in [0,1]:
        axis = axes[train0AndValidation1] if numberOfMetricsPlotted == 1 else axes[train0AndValidation1][axis_i]
        axis.yaxis.grid(True, linestyle='--', which='major', color='black', alpha=1.0)
        axis.tick_params(axis='y', labelsize=fontSizeYTickLabel)
        
    axes[0].set_ylim(0., 1.);
    axes[1].set_ylim(0., 1.);
    
    axes[0].set_ylabel('Validation', fontsize=fontSizeYAxisLabel)
    axes[1].set_ylabel('Training', fontsize=fontSizeYAxisLabel)
    
    """
    Moving the legend-box:
    - You grab a subplot. (depending on the axis that you ll use at: axis.legend(...))
    - Then, you specify with loc=, the anchor of the LEGENDBOX that you will move in relation to the BOTTOM-LEFT corner of the above axis..
        loc = 'upper right' (1), 'upper left' (2), 'lower left' (3), 'lower right' (4)
    - bbox_to_anchor=(x-from-left, y-from-bottom, width, height). x and y can be negatives. Specify how much to move legend's loc from the bottom left corner of the axis.
        x, y, width and height are floats, giving the percentage of the AXIS's size. Eg x=0.5, y=0.5 moves it at the middle of the subplot.
    """
    leg = axes[1].legend(loc='upper left', bbox_to_anchor=(0., -.25, 0., 0.),#(0., -1.3, 1., 1.),
                       ncol=legendNumberOfColumns, borderaxespad=0. , fontsize=legendFontSize, labelspacing = 0., columnspacing=1.0)#mode="expand",
    #Make the lines in the legend wider.
    for legobj in leg.legendHandles:
        legobj.set_linewidth(6.0)
        
    if saveFigureBool :
        plt.savefig('./trainingProgress.pdf', dpi=fig.dpi)#, bbox_inches='tight')
        
    plt.show()
    
def plotProgressDetailed(measuredMetricsFromAllExperiments, legendList, movingAverSubeps, subepochsPerEpOfExpers, epochsPerFullInferOfExpers, saveFigureBool) :
    colors = ["r","g","b","c","m","k"]
    linestyles = ['-', '--', ':', '_', '-.']
    
    subplotTitles = [ ["Mean Accuracy", "Sensitivity", "Specificity", "DSC (samples)", "DSC (full-segm)"], # Validation
                      ["Mean Accuracy", "Sensitivity", "Specificity", "DSC (samples)", "DSC (full-segm)"] # Training
                    ]
    
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
    fig.suptitle(os.path.basename(__file__) + ": Moving Average over ["+ str(movingAverSubeps)+"] value. For each plotted experiment, Subepochs per Epoch: " + str(subepochsPerEpOfExpers) + ", Epochs between Full-Segmentations: " + str(epochsPerFullInferOfExpers), fontsize=8)#, fontweight='bold')
    
    maxNumOfEpsDurationOfExps = 0 # The number of epochs that the longest experiment lasted.
    
    for valOrTrain_i in xrange(0, len(measuredMetricsFromAllExperiments)) :
        for valOrTrainExperiment_i in xrange(0, len(measuredMetricsFromAllExperiments[valOrTrain_i])) :
            valOrTrainExperiment = measuredMetricsFromAllExperiments[valOrTrain_i][valOrTrainExperiment_i]
            for meanPosNegDice1_i in xrange(0, len(valOrTrainExperiment)) :
                numberOfSubsPerEpoch = subepochsPerEpOfExpers[valOrTrainExperiment_i]
                numberOfEpsBetweenFullInf = epochsPerFullInferOfExpers[valOrTrainExperiment_i]

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
            
    if saveFigureBool :
        plt.savefig('./trainingProgress.pdf', dpi=fig.dpi)#, bbox_inches='tight')
        
    plt.show()





if __name__ == '__main__':

    myArgParser = setupArgParser()
    args = myArgParser.parse_args()
    if len(sys.argv) == 1:
        print("For help on the usage of this script, please use the option [-h]."); exit(1)
        
    detailedPlotBool = args.detailed_plot
    movingAverSubeps = args.moving_average
    saveFigBool = args.save_figure
    
    logFiles = args.log_files
    (listOfExperimentsNames, subepochsPerEpFromEachLog, epochsPerFullInferFromEachLog) = parseVariablesOfTrainingSessionsFromListOfLogs(logFiles)
    
    if not detailedPlotBool: # basic plot
        if args.classes_to_plot :
            print "ERROR: -c/--classes option should only be provided when -d/--detailed plotting is specified. Default basic plotting parses and shows overall and not class-specific accuracy."
            print "Exiting!"; exit(1)
            
        measuredMetricsFromAllExperiments = optimizedParseMetricsFromLogs(logFiles, detailedPlotBool, None, movingAverSubeps)
        plotProgressBasic(measuredMetricsFromAllExperiments, listOfExperimentsNames, movingAverSubeps, subepochsPerEpFromEachLog, saveFigBool)
        
    else : # detailed plot
        if not args.classes_to_plot : #Default class when none given as argument
            classesFromEachLogFile = len(logFiles)*[[0]]
        elif len(logFiles) == 1 : # 1 log file only
            classesFromEachLogFile = [ args.classes_to_plot ] # [ [class0, class1, ...] ]
        elif len(args.classes_to_plot) == 1 : # multiple logs provided, and 1 class argument
            classesFromEachLogFile = [ [ args.classes_to_plot[0] ] for i in xrange(len(logFiles)) ] # [ [class0], [class0], ... ]
        elif len(args.classes_to_plot) == len(logFiles) :
            classesFromEachLogFile = [ [ args.classes_to_plot[i] ] for i in xrange(len(logFiles)) ] # [ [class0], [class1], [class2], ...] 
        else : # logFiles provided, multiple classes provided, but not the same number as the log files.
            print("ERROR:\tThe number of log files given is not the same with the number of arguments that specify which class's accuracy to plot from each log file.")
            print("\tPlease provide the same number of Class arguments, or just 1, if the same class is to be plotted from all log files. Exiting."); exit(1)
        # Parse the logs and get the names of the files to put in legend, the subepochs per epoch in each session and the number of epochs between full-segmentation.
        
        """
        # Hack for convenience. Comment out the above, uncomment this, pass sth random as logfile and it will give the below hard-coded values to the variables. In case I want to use it this way.
        logFiles = [ "path-to-log-file1", "path-to-log-file2" ]
        classesFromEachLogFile = [ [1,2], [1] ]
        subepochsPerEpFromEachLog = [20]*len(logFiles)
        epochsPerFullInferFromEachLog = [5]*len(logFiles)
        listOfExperimentsNames = [ "TrainingSession1", "TrainingSession2" ]
        """
        measuredMetricsFromAllExperiments = optimizedParseMetricsFromLogs( logFiles, detailedPlotBool, classesFromEachLogFile, movingAverSubeps )
        legendList = makeLegendList(listOfExperimentsNames, classesFromEachLogFile)
        (subepochsPerEpOfExpers, epochsPerFullInferOfExpers) = makeHelperVariablesPerExperiment(logFiles, classesFromEachLogFile, subepochsPerEpFromEachLog, epochsPerFullInferFromEachLog)
        
        plotProgressDetailed(measuredMetricsFromAllExperiments, legendList, movingAverSubeps, subepochsPerEpOfExpers, epochsPerFullInferOfExpers, saveFigBool)







