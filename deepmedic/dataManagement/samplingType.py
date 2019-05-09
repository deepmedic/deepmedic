# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np

class SamplingType(object) :
    def __init__(self, log, samplingType, numberOfClassesInclBackgr):
        self.log = log
        # foreBackgr=0 , uniform=1, fullImage=2, targettedPerClass=3
        self.samplingType = samplingType
        
        if self.samplingType == 0 :
            self.stringOfSamplingType = "Fore/Background"
            self.arrayWithStringPerCategoryToSample = ["Foreground", "Background"]
        elif self.samplingType == 1 :
            self.stringOfSamplingType = "Uniform"
            self.arrayWithStringPerCategoryToSample = ["Uniform"]
        elif self.samplingType == 2 :
            self.stringOfSamplingType = "Whole-Image"
            self.arrayWithStringPerCategoryToSample = ["WholeImage"]
        elif self.samplingType == 3 :
            self.stringOfSamplingType = "Per-Class"
            self.arrayWithStringPerCategoryToSample = ["Class-" + str(i) for i in range(numberOfClassesInclBackgr) ]
        else :
            self.log.print3("ERROR: Tried to create a SamplingType instance, but the samplingType passed was invalid. Should be [0,1,2,3]. Exiting!"); exit(1)
            
    def setPercentOfSamplesPerCategoryToSample(self, percentageOfSamplesPerCategoryOfSampling) :
        if self.samplingType in [0,3] and len(percentageOfSamplesPerCategoryOfSampling) != self.getNumberOfCategoriesToSample() :
            self.log.print3("ERROR: In class [SamplingType], the list percentageOfSamplesPerCategoryOfSampling had [" + str(len(percentageOfSamplesPerCategoryOfSampling)) + "] elements. In this case of [" + self.stringOfSamplingType + "], it should have [" + str(self.getNumberOfCategoriesToSample()) + "]! Exiting!"); exit(1)
            
        if self.samplingType == 0 :
            self.percentOfSamplesPerCategoryToSample = self.normalizePercentages(percentageOfSamplesPerCategoryOfSampling)
        elif self.samplingType == 1 :
            self.percentOfSamplesPerCategoryToSample = [1.0]
        elif self.samplingType == 2 :
            self.percentOfSamplesPerCategoryToSample = [1.0]
        elif self.samplingType == 3 :
            self.percentOfSamplesPerCategoryToSample = self.normalizePercentages(percentageOfSamplesPerCategoryOfSampling)
        else :
            self.log.print3("ERROR: in [SamplingType]. self.samplingType was invalid. Should be [0,1,2,3]. Exiting!"); exit(1)
            
    def normalizePercentages(self, listOfWeights) :
        arrayOfWeights = np.asarray(listOfWeights, dtype="float32")
        return arrayOfWeights / (1.0*np.sum(arrayOfWeights))
    # API
    def getIntSamplingType(self) :
        return self.samplingType
    def getStringOfSamplingType(self) :
        return self.stringOfSamplingType
    def getStringsPerCategoryToSample(self) :
        return self.arrayWithStringPerCategoryToSample
    def getPercentOfSamplesPerCategoryToSample(self) :
        return self.percentOfSamplesPerCategoryToSample
    def getNumberOfCategoriesToSample(self) :
        return len(self.arrayWithStringPerCategoryToSample)
    
    
    def logicDecidingSamplingMapsPerCategory(   self,
                                                weightmaps_to_sample_per_cat,
                                                gt_lbl_img,
                                                roi_mask,
                                                dims_of_scan
                                                ) :
        # If weight-maps are provided (not None), they should be provided for *every* class of the sampling-type.
        
        # a) Check if weighted maps are given. In that case, use them.
        # b) If no weightMaps, and ROI/GT given, use them.
        # c) Otherwise, whole image.
        # Depending on the SamplingType, different behaviour as in how many categories.
        # dims_of_scan: [H, W, D]
        if self.samplingType == 0 : # fore/background
            if weightmaps_to_sample_per_cat is not None : #Both weight maps should be provided currently.
                numOfProvidedWeightMaps = len(weightmaps_to_sample_per_cat)
                if numOfProvidedWeightMaps != self.getNumberOfCategoriesToSample() :
                    self.log.print3("ERROR: For SamplingType = Fore/Background(0), [" + str(numOfProvidedWeightMaps) + "] weight maps were provided! "+\
                                    "Two (fore/back) were expected! Exiting!"); exit(1)
                finalWeightMapsToSampleFromPerCategoryForSubject = weightmaps_to_sample_per_cat
            elif gt_lbl_img is None:
                self.log.print3("ERROR: For SamplingType=[" + self.stringOfSamplingType + "], if weighted-maps are not provided, "+\
                                "at least Ground Truth labels should be given to extract foreground! Exiting!"); exit(1)
            elif roi_mask is not None : # and provided GT
                maskForForegroundSampling = ( (gt_lbl_img>0)*(roi_mask>0) ).astype(int)
                maskForBackgroundSampling_roiMinusGtLabels = ( (maskForForegroundSampling==0)*(roi_mask>0) ).astype(int)
                finalWeightMapsToSampleFromPerCategoryForSubject = [ maskForForegroundSampling, maskForBackgroundSampling_roiMinusGtLabels ] #Foreground / Background (in sequence)
            else : # no weightmaps, gt provided and roi is not provided.
                maskForForegroundSampling = (gt_lbl_img>0).astype(int)
                maskForBackgroundSampling_roiMinusGtLabels = np.ones(dims_of_scan, dtype="int16") * (maskForForegroundSampling==0)
                finalWeightMapsToSampleFromPerCategoryForSubject = [ maskForForegroundSampling, maskForBackgroundSampling_roiMinusGtLabels ] #Foreground / Background (in sequence)
        elif self.samplingType == 1 : # uniform
            if weightmaps_to_sample_per_cat is not None :
                numOfProvidedWeightMaps = len(weightmaps_to_sample_per_cat)
                if numOfProvidedWeightMaps != 1 :
                    self.log.print3("ERROR: For SamplingType=[" + self.stringOfSamplingType + "], [" + str(numOfProvidedWeightMaps) + "] weight maps were provided! "+\
                                    "One was expected! Exiting!"); exit(1)
                finalWeightMapsToSampleFromPerCategoryForSubject = weightmaps_to_sample_per_cat #Should be an array with dim1==1 already.
            elif roi_mask is not None :
                finalWeightMapsToSampleFromPerCategoryForSubject = [ roi_mask ] #Be careful to not change either of the two arrays later or there'll be a problem.
            else :
                finalWeightMapsToSampleFromPerCategoryForSubject = [ np.ones(dims_of_scan, dtype="int16") ]
        elif self.samplingType == 2 : # full image. SAME AS UNIFORM?
            if weightmaps_to_sample_per_cat is not None :
                numOfProvidedWeightMaps = len(weightmaps_to_sample_per_cat)
                if numOfProvidedWeightMaps != 1 :
                    self.log.print3("ERROR: For SamplingType=[" + self.stringOfSamplingType + "], [" + str(numOfProvidedWeightMaps) + "] weight maps were provided! "+\
                                    "One was expected! Exiting!"); exit(1)
                finalWeightMapsToSampleFromPerCategoryForSubject = weightmaps_to_sample_per_cat #Should be an array with dim1==1 already.
            elif roi_mask is not None :
                finalWeightMapsToSampleFromPerCategoryForSubject = [ roi_mask ] #Be careful to not change either of the two arrays later or there'll be a problem.
            else :
                finalWeightMapsToSampleFromPerCategoryForSubject = [ np.ones(dims_of_scan, dtype="int16") ]
        elif self.samplingType == 3 : # Targeted per class.
            if weightmaps_to_sample_per_cat is not None :
                numOfProvidedWeightMaps = len(weightmaps_to_sample_per_cat)
                if numOfProvidedWeightMaps != self.getNumberOfCategoriesToSample() :
                    self.log.print3("ERROR: For SamplingType=[" + self.stringOfSamplingType + "], [" + str(numOfProvidedWeightMaps) + "] weight maps were provided! "+\
                                    "As many as the classes [" + str(self.getNumberOfCategoriesToSample()) + "] (incl Background) were expected! Exiting!"); exit(1)
                finalWeightMapsToSampleFromPerCategoryForSubject = weightmaps_to_sample_per_cat #Should have as many entries as classes (incl backgr).
            elif gt_lbl_img is None:
                self.log.print3("ERROR: For SamplingType=TargettedPerClass(3), either weightMaps for each class or GT labels should be given! Exiting!"); exit(1)
            elif roi_mask is not None : # and provided GT
                finalWeightMapsToSampleFromPerCategoryForSubject = []
                for cat_i in range( self.getNumberOfCategoriesToSample() ) : # Should be the same number as the number of actual classes, including background.
                    finalWeightMapsToSampleFromPerCategoryForSubject.append( ( (gt_lbl_img==cat_i)*(roi_mask>0) ).astype(int) )
            else : # no weightmaps, gt provided and roi is not provided.
                finalWeightMapsToSampleFromPerCategoryForSubject = []
                for cat_i in range( self.getNumberOfCategoriesToSample() ) : # Should be the same number as the number of actual classes, including background.
                    finalWeightMapsToSampleFromPerCategoryForSubject.append( (gt_lbl_img==cat_i).astype(int) )
                
        else :
            self.log.print3("ERROR: Sampling-type-number passed in [logicDecidingAndGivingFinalSamplingMapsForEachCategory] was invalid. Should be [0,1,2,3]. Exiting!"); exit(1)
            
        return finalWeightMapsToSampleFromPerCategoryForSubject
    

