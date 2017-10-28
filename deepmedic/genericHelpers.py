# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange
import os
import gzip
import datetime
import nibabel as nib
import numpy as np
import random
from math import ceil

import pickle
try:
    import cPickle
except ImportError:
    # python3 compatibility
    import _pickle as cPickle
    

def strFlXDec(flNum, numDec) :
    #ala "{0:.2f}".format(13.94999)
    stringThatSaysHowManyDecimals = "{0:." + str(numDec) + "f}"
    return stringThatSaysHowManyDecimals.format(flNum)
def strFl4Dec(flNum) :
    return strFlXDec(flNum, 4)
def strFl5Dec(flNum) :
    return strFlXDec(flNum, 5)
#This function is used to get a STRING-list-of-float-with-X-decimal-points. Used for printing.
def strFlListXDec(listWithFloats, numDec) :
    stringThatSaysHowManyDecimals = "%."+str(numDec)+"f"
    strList = "[ "
    for num in listWithFloats :
        strList += stringThatSaysHowManyDecimals % num + " "
    strList += "]"
    #listWithStringFloatsWithLimitedDecimals = [ stringThatSaysHowManyDecimals % num for num in listWithFloats]
    return strList
def strFlList4Dec(listWithFloats) :
    return strFlListXDec(listWithFloats, 4)
    
# Gets floatOrNotAppl that is either a float or a "Not Applicable" string.
# Returns a string, which is the float with 4 decimals, or the "Not Applicable" string if that's the case.
def strFlXfNA(floatOrNotAppl, numDec, notApplicPattern) :
    stringThatSaysHowManyDecimals = "{:." + str(numDec) + "f}"
    return stringThatSaysHowManyDecimals.format(floatOrNotAppl) if floatOrNotAppl != notApplicPattern else notApplicPattern
def strFl4fNA(floatOrNotAppl, notApplicPattern) :
    return strFlXfNA(floatOrNotAppl, 4, notApplicPattern)
def strFl5fNA(floatOrNotAppl, notApplicPattern) :
    return strFlXfNA(floatOrNotAppl, 5, notApplicPattern)
    
def strListFlXfNA(listWithFloatsOrNotAppl, numDec, notApplicPattern) : # If you just do a normal list, the internal string-floats are printed with quotes ' '
    stringThatSaysHowManyDecimals = "%." + str(numDec) + "f"
    strList = "[ "
    for element in listWithFloatsOrNotAppl :
        strList += stringThatSaysHowManyDecimals % element + " " if element != notApplicPattern else notApplicPattern + " "
    strList += "]"
    return strList

def strListFl4fNA(listWithFloatsOrNotAppl, notApplicPattern) : # If you just print a normal list, the internal string-floats are printed with quotes ' '. To avoit it we use this.
    return strListFlXfNA(listWithFloatsOrNotAppl, 4, notApplicPattern)
def strListFl5fNA(listWithFloatsOrNotAppl, notApplicPattern) : # If you just print a normal list, the internal string-floats are printed with quotes ' '. To avoit it we use this.
    return strListFlXfNA(listWithFloatsOrNotAppl, 5, notApplicPattern)

def getMeanOfListExclNA(list1, notApplicPattern) :
    # Calculates mean over the list's entries that are applicable. i.e. the ones that are not notApplicPattern == "N/A".
    # Returns NotApplicablePattern if all entries are not-applicable.
    addedValuesForMeanCalc = 0
    numberOfApplicableEntries = 0
    for subep_i in xrange(len(list1)) :
        if list1[subep_i] != notApplicPattern :
            addedValuesForMeanCalc += list1[subep_i]
            numberOfApplicableEntries += 1
    return addedValuesForMeanCalc / float(numberOfApplicableEntries) if numberOfApplicableEntries > 0 else notApplicPattern

def getMeanPerColOf2dListExclNA(list2d, notApplicPattern) :
    # list has dimensions: rows X col.
    # Return a list, which has 1 element per column, which is the average of the column, excluding any "NotApplicable" elements.
    listWithMeanPerColumn = []
    for col_i in xrange(len(list2d[0])) :
        listColumn = [list2d[i][col_i] for i in xrange(len(list2d))] # make a list of a column's elements.
        meanOfColumn = getMeanOfListExclNA(listColumn, notApplicPattern) #will return NA-pattern if all elements are NA.
        listWithMeanPerColumn.append(meanOfColumn)
    return listWithMeanPerColumn


def calculateDiceCoefficient(predictedBinaryLabels, groundTruthBinaryLabels) :
    unionCorrectlyPredicted = predictedBinaryLabels * groundTruthBinaryLabels
    numberOfTruePositives = np.sum(unionCorrectlyPredicted)
    numberOfGtPositives = np.sum(groundTruthBinaryLabels)
    diceCoeff = (2.0 * numberOfTruePositives) / (np.sum(predictedBinaryLabels) + numberOfGtPositives) if numberOfGtPositives!=0 else -1
    return diceCoeff

def load_object_from_file(filenameWithPath) :
    f = file(filenameWithPath, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

def dump_object_to_file(my_obj, filenameWithPath) :
    """
    my_obj = object to pickle
    filenameWithPath = a string with the full path+name
    
    The function uses the 'highest_protocol' which is supposed to be more storage efficient.
    It uses cPickle, which is coded in c and is supposed to be faster than pickle.
    Remember, this instance is safe to load only from a code which is fully-compatible (same version)
    ...with the code this was saved from, i.e. same classes define.
    If I need forward compatibility, read this: http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
    """
    f = file(filenameWithPath, 'wb')
    cPickle.dump(my_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
def load_object_from_gzip_file(filenameWithPath) :
    f = gzip.open(filenameWithPath, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

def dump_object_to_gzip_file(my_obj, filenameWithPath) :
    f = gzip.open(filenameWithPath, 'wb')
    cPickle.dump(my_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
#This could be renamed to be more generic.
def get_random_subject_indices_to_load_on_GPU(total_number_of_subjects, 
                                            max_subjects_on_gpu_for_subepoch, 
                                            get_max_subjects_for_gpu_even_if_total_less=False,
                                            myLogger=None):
    
    subjects_indices = list(range(total_number_of_subjects)) #list() for python3 compatibility, as range cannot get assignment in shuffle()
    random_order_chosen_subjects=[]
    
    random.shuffle(subjects_indices) #does it in place. Now they are shuffled
    
    if max_subjects_on_gpu_for_subepoch>=total_number_of_subjects:
        random_order_chosen_subjects += subjects_indices
        
        if get_max_subjects_for_gpu_even_if_total_less : #This is if I want to have a certain amount on GPU, even if total subjects are less.
            while (len(random_order_chosen_subjects)<max_subjects_on_gpu_for_subepoch):
                random.shuffle(subjects_indices)
                number_of_extra_subjects_to_get_to_fill_gpu = min(max_subjects_on_gpu_for_subepoch - len(random_order_chosen_subjects), total_number_of_subjects)
                random_order_chosen_subjects += (subjects_indices[:number_of_extra_subjects_to_get_to_fill_gpu])
            if len(random_order_chosen_subjects)!=max_subjects_on_gpu_for_subepoch :
                if myLogger!=None :
                    myLogger.print3("ERROR: in get_random_subjects_indices_to_load_on_GPU(), something is wrong!")
                else :
                    print("ERROR: in get_random_subjects_indices_to_load_on_GPU(), something is wrong!")
                exit(1)
    else:
        random_order_chosen_subjects += subjects_indices[:max_subjects_on_gpu_for_subepoch]
        
    return random_order_chosen_subjects

    
def datetimeNowAsStr() :
    #datetime returns in the format: YYYY-MM-DD HH:MM:SS.millis but ':' is not supported for Windows' naming convention.
    dateTimeNowStr = str(datetime.datetime.now())
    dateTimeNowStr = dateTimeNowStr.replace(':','.')
    dateTimeNowStr = dateTimeNowStr.replace(' ','.')
    return dateTimeNowStr

