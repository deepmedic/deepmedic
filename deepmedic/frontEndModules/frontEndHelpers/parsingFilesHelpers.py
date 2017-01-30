# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

import os

def getAbsPathEvenIfRelativeIsGiven(pathGiven, absolutePathToWhereRelativePathRelatesTo) :
    #os.path.normpath "cleans" Additional ../.// etc.
    if os.path.isabs(pathGiven) : 
        return os.path.normpath(pathGiven)
    else : #relative path given. Need to make absolute path
        if os.path.isdir(absolutePathToWhereRelativePathRelatesTo) :
            relativePathToWhatGiven = absolutePathToWhereRelativePathRelatesTo
        elif os.path.isfile(absolutePathToWhereRelativePathRelatesTo) :
            relativePathToWhatGiven = os.path.dirname(absolutePathToWhereRelativePathRelatesTo)
        else : #not file, not dir, exit.
            print "ERROR: in [func:returnAbsolutePathEvenIfRelativePathIsGiven()] Given path :", absolutePathToWhereRelativePathRelatesTo, " does not correspond to neither an existing file nor a directory. Exiting!"; exit(1)
        return os.path.normpath(relativePathToWhatGiven + "/" + pathGiven)
    
def checkIfAllElementsOfAListAreFilesAndExitIfNot(pathToTheListingFile, list1) :
    for filepath in list1 :
        if not os.path.isfile(filepath) :
            print "ERROR: in [checkIfAllElementsOfAListExistAndExitIfNot()] path:", filepath, " given in :", pathToTheListingFile," does not correspond to a file. Exiting!"
            print exit(1)
            
def parseFileLinesInList(pathToListingFile) :
    list1 = []
    with open(pathToListingFile, "r") as inp :
        for line in inp :
            if not line.startswith("#") and line.strip() <> "" :
                list1.append(line.strip())
    return list1

def parseAbsFileLinesInList(pathToListingFile) :
    # os.path.normpath below is to "clean" the paths from ./..//...
    pathToFolderContainingThisListFile = os.path.dirname(pathToListingFile)
    list1 = []
    with open(pathToListingFile, "r") as inp :
        for line in inp :
            if not line.startswith("#") and line.strip() <> "" :
                pathToFileParsed = line.strip()
                if os.path.isabs(pathToFileParsed) : #abs path.
                    list1.append(os.path.normpath(pathToFileParsed))
                else : #relative path to this listing-file.
                    list1.append(os.path.normpath(pathToFolderContainingThisListFile + "/" + pathToFileParsed))
    return list1

def checkListContainsCorrectNumberOfCasesOtherwiseExitWithError(numberOfCasesPreviously, pathToGivenListFile, listOfFilepathsToChannelIForEachCase) :
    numberOfContainedCasesInList = len(listOfFilepathsToChannelIForEachCase)
    if numberOfCasesPreviously <> numberOfContainedCasesInList :
        print "ERROR: Given file:", pathToGivenListFile, " contains #", numberOfContainedCasesInList," entries, whereas previously checked files contained #", numberOfCasesPreviously,". All listing-files for channels, masks, etc, should contain the same number of entries, one for each case.\nExiting!"
        exit(1)
        
def checkThatAllEntriesOfAListFollowNameConventions(listOfPredictionNamesForEachCaseInListingFile) :
    for entry in listOfPredictionNamesForEachCaseInListingFile :
        if entry.find("/") > -1 or entry.startswith(".") :
            print "ERROR: in [checkThatAllEntriesOfAListFollowNameConventions()] while checking that all entries follow name-conventions. Entry \"", entry, "\" was found to begin with \'.\' or contain \'/\'. Please correct. Exiting!"
            print exit(1)
