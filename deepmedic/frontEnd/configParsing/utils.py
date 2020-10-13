# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import input
import os


def abs_from_rel_path(pathGiven, absolutePathToWhereRelativePathRelatesTo):
    #os.path.normpath "cleans" Additional ../.// etc.
    if os.path.isabs(pathGiven) : 
        return os.path.normpath(pathGiven)
    else : #relative path given. Need to make absolute path
        if os.path.isdir(absolutePathToWhereRelativePathRelatesTo) :
            relativePathToWhatGiven = absolutePathToWhereRelativePathRelatesTo
        elif os.path.isfile(absolutePathToWhereRelativePathRelatesTo) :
            relativePathToWhatGiven = os.path.dirname(absolutePathToWhereRelativePathRelatesTo)
        else : #not file, not dir, exit.
            print("ERROR: in [func:returnAbsolutePathEvenIfRelativePathIsGiven()] Given path :", absolutePathToWhereRelativePathRelatesTo, " does not correspond to neither an existing file nor a directory. Exiting!"); exit(1)
        return os.path.normpath(relativePathToWhatGiven + "/" + pathGiven)


def checkIfAllElementsOfAListAreFilesAndExitIfNot(pathToTheListingFile, list1):
    for filepath in list1 :
        if not os.path.isfile(filepath) :
            print("ERROR: in [checkIfAllElementsOfAListExistAndExitIfNot()] path:", filepath, " given in :", pathToTheListingFile," does not correspond to a file. Exiting!")
            exit(1)


def parse_filelist(filelist_path, make_abs=False):
    # os.path.normpath below is to "clean" the paths from ./..//...
    path_to_folder_w_filelist = os.path.dirname(filelist_path)
    list1 = []
    with open(filelist_path, "r") as inp:
        for line in inp:
            if line.strip() == "-":  # Special char. E.g. indicating the non existence of channel, to be zero-filled.
                list1.append("-")
            elif not line.startswith("#") and line.strip() != "":
                path_to_file = line.strip()

                if (not make_abs) or os.path.isabs(path_to_file):
                    list1.append(os.path.normpath(path_to_file))
                else:  # relative path to this listing-file.
                    list1.append(os.path.normpath(path_to_folder_w_filelist + "/" + path_to_file))
    return list1


def checkListContainsCorrectNumberOfCasesOtherwiseExitWithError(numberOfCasesPreviously, pathToGivenListFile, listOfFilepathsToChannelIForEachCase) :
    numberOfContainedCasesInList = len(listOfFilepathsToChannelIForEachCase)
    if numberOfCasesPreviously != numberOfContainedCasesInList :
        raise IOError("ERROR: Given file:", pathToGivenListFile +\
              "\n\t contains #", numberOfContainedCasesInList," entries, whereas previously checked files contained #", numberOfCasesPreviously,"."+\
              "\n\t All listing-files for channels, masks, etc, should contain the same number of entries, one for each case.")


def checkThatAllEntriesOfAListFollowNameConventions(listOfPredictionNamesForEachCaseInListingFile) :
    for entry in listOfPredictionNamesForEachCaseInListingFile :
        if entry.find("/") > -1 or entry.startswith(".") :
            raise IOError("ERROR: Check that all entries follow name-conventions failed."+\
                          "\n\t Entry \"", entry, "\" was found to begin with \'.\' or contain \'/\'. Please correct this.")


def check_and_adjust_path_to_ckpt( log, filepath_to_ckpt ):
    STR_DM_CKPT = ".model.ckpt"
    index_of_str = filepath_to_ckpt.rfind(STR_DM_CKPT)
    if index_of_str > -1 and len(filepath_to_ckpt) > len(filepath_to_ckpt[:index_of_str])+len(STR_DM_CKPT) : # found.
        
        user_input = None
        string_warn = "It seems that the path to the model to load paramters from, a tensorflow checkpoint, was given wrong."+\
                       "\n\t The path to checkpoint should be of the form: [...name...date...model.ckpt] (finishing with .ckpt)"+\
                       "\n\t Note that you should not point to the .data, .index or .meta files that are saved. Rather, shorten their names till the .ckpt"+\
                       "\n\t Given path seemed longer: " + str(filepath_to_ckpt)
        try:
            user_input = input(">>\t " + string_warn +\
                                   "\n\t Do you wish that we shorten the path to end with [.ckpt] as expected? [y/n] : ")
            while user_input not in ['y','n']: 
                user_input = input("Please specify 'y' or 'n': ")
        except:
            log.print3("\nWARN:\t " + string_warn +\
                       "\n\t We tried to request command line input from user whether to shorten it after [.ckpt] but failed (remote use? nohup?"+\
                       "\n\t Continuing without doing anything. If this fails, try to give the correct path, ending with [.ckpt]")
        if user_input == 'y':
            filepath_to_ckpt = filepath_to_ckpt[ : index_of_str+len(STR_DM_CKPT)]
            log.print3("Changed path to load parameters from: "+str(filepath_to_ckpt))
        else:
            log.print3("Continuing without doing anything.")
            
    return filepath_to_ckpt


def normfullpath(abspath, relpath):
    if os.path.isabs(relpath):
        return relpath
    else:
        return os.path.normpath(os.path.join(abspath, relpath))


def get_paths_from_df(log, df, abs_path, req_gt=True):
    # df: Pandas dataframe, or one with same API.
    # channels are sorted alphabetically to ensure consistency
    c_names = sorted([c for c in list(df.columns) if c.startswith('channel_')])
    if len(c_names) == 0:
        # no channels error raise - move to function later
        log.print3('No channel columns in dataframe. Columns should be named "channel_[channel_name]". Exiting')
        exit(1)
    # [[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
    channels = [[normfullpath(abs_path, c) for c in list(item[c_names])] for _, item in df.iterrows()]

    try:
        gt = [normfullpath(abs_path, g) for g in list(df['ground_truth'])]
    except KeyError:
        gt = None
        if req_gt:
            raise Exception('No ground truth column in dataframe, as required. Column should be named "ground_truth".')
    try:
        roi = [normfullpath(abs_path, r) for r in list(df['roi_mask'])]
    except KeyError:
        roi = None
        log.print3('No roi masks column in dataframe. Column should be named "roi_mask".')

    try:
        pred = [p for p in list(df['prediction_filename'])]
    except KeyError:
        pred = None

    return channels, gt, roi, pred


def parse_fpaths_of_channs_from_filelists(list_of_filelists, abs_path_root):
    """
    list_of_filelists: list with one filepath per channel. Each path points to the file-list for the channel.
                       The file-list is a list with 1 row per case. Each row is the path to the corresponding
                       image of the case for the specific channel.
    abs_path_root: list_of_filelists may contain relative paths. This is the absolute path that they relate to.
    """
    chan_fpaths_per_chan_per_case = []
    for path_to_filelist_for_chan in list_of_filelists:
        # The below: [[case1-ch1, ..., caseN-ch1], [case1-ch2,...,caseN-ch2]]
        abs_path_to_filelist_for_chan = abs_from_rel_path(path_to_filelist_for_chan, abs_path_root)
        chan_fpaths_per_chan_per_case.append(parse_filelist(abs_path_to_filelist_for_chan, make_abs=True))
        # The below [[case1-ch1, case1-ch2], ..., [caseN-ch1, caseN-ch2]]
    channels_fpaths = [list(item) for item in zip(*tuple(chan_fpaths_per_chan_per_case))]
    return channels_fpaths
