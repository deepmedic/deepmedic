# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import os
import datetime
    

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
    for subep_i in range(len(list1)) :
        if list1[subep_i] != notApplicPattern :
            addedValuesForMeanCalc += list1[subep_i]
            numberOfApplicableEntries += 1
    return addedValuesForMeanCalc / float(numberOfApplicableEntries) if numberOfApplicableEntries > 0 else notApplicPattern

def getMeanPerColOf2dListExclNA(list2d, notApplicPattern) :
    # list has dimensions: rows X col.
    # Return a list, which has 1 element per column, which is the average of the column, excluding any "NotApplicable" elements.
    listWithMeanPerColumn = []
    for col_i in range(len(list2d[0])) :
        listColumn = [list2d[i][col_i] for i in range(len(list2d))] # make a list of a column's elements.
        meanOfColumn = getMeanOfListExclNA(listColumn, notApplicPattern) #will return NA-pattern if all elements are NA.
        listWithMeanPerColumn.append(meanOfColumn)
    return listWithMeanPerColumn



def datetimeNowAsStr() :
    #datetime returns in the format: YYYY-MM-DD HH:MM:SS.millis but ':' is not supported for Windows' naming convention.
    dateTimeNowStr = str(datetime.datetime.now())
    dateTimeNowStr = dateTimeNowStr.replace(':','.')
    dateTimeNowStr = dateTimeNowStr.replace(' ','.')
    return dateTimeNowStr

