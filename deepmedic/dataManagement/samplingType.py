# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np

class SamplingType(object):
    def __init__(self, log, sampling_type, n_classes_incl_bgr):
        self._log = log
        # foreBackgr=0 , uniform=1, fullImage=2, targettedPerClass=3
        self._sampling_type = sampling_type
        self._perc_to_sample_per_cat = None
        
        assert self._sampling_type in [0,1,2,3]
        if self._sampling_type == 0:
            self._sampling_type_str = "Fore/Background"
            self._array_w_str_per_sampling_category = ["Foreground", "Background"]
        elif self._sampling_type == 1:
            self._sampling_type_str = "Uniform"
            self._array_w_str_per_sampling_category = ["Uniform"]
        elif self._sampling_type == 2:
            self._sampling_type_str = "Whole-Image"
            self._array_w_str_per_sampling_category = ["WholeImage"]
        elif self._sampling_type == 3:
            self._sampling_type_str = "Per-Class"
            self._array_w_str_per_sampling_category = ["Class-" + str(i) for i in range(n_classes_incl_bgr) ]
        else:
            raise ValueError("Invalid value for sampling type.")
        
    
    def set_perc_of_samples_per_cat(self, perc_of_samples_per_cat):
        # For each category to sample, set the percent of the samples that should be taken from this category.
        # Categories to sample are different for each sampling-type.
        if self._sampling_type in [0,3] and len(perc_of_samples_per_cat) != self.get_n_sampling_cats():
            self._log.print3("ERROR: The list perc_of_samples_per_cat had [" + str(len(perc_of_samples_per_cat)) + "] elements." +\
                             " For type [" + self._sampling_type_str + "], it requires [" + str(self.get_n_sampling_cats()) + "]! Exiting!"); exit(1)
            
        if self._sampling_type == 0:
            self._perc_to_sample_per_cat = self._normalize_percentages(perc_of_samples_per_cat)
        elif self._sampling_type == 1:
            self._perc_to_sample_per_cat = [1.0]
        elif self._sampling_type == 2:
            self._perc_to_sample_per_cat = [1.0]
        elif self._sampling_type == 3:
            self._perc_to_sample_per_cat = self._normalize_percentages(perc_of_samples_per_cat)
        else:
            raise ValueError("Invalid value for sampling type.")
            
    def _normalize_percentages(self, list_of_weights):
        array_of_weights = np.asarray(list_of_weights, dtype="float32")
        return array_of_weights / (1.0*np.sum(array_of_weights))
    # API
    def get_type_as_int(self):
        return self._sampling_type
    def get_type_as_str(self):
        return self._sampling_type_str
    def get_sampling_cats_as_str(self):
        return self._array_w_str_per_sampling_category
    def get_perc_to_sample_per_cat(self):
        return self._perc_to_sample_per_cat
    def get_n_sampling_cats(self):
        return len(self._array_w_str_per_sampling_category)
    
    
    def derive_sampling_maps_per_cat(self, wmaps_to_sample_per_cat, gt_lbl_img, roi_mask, dims_of_scan):
        # wmaps_to_sample_per_cat: If weight-maps are provided (not None), they should be provided for *every* class of the sampling-type.
        # a) Check if weighted maps are given. In that case, use them.
        # b) If no weightMaps, and ROI/GT given, use them.
        # c) Otherwise, whole image.
        # Depending on the sampling_type, different behaviour as in how many categories.
        # dims_of_scan: [H, W, D]
        
        n_wmaps = 0 if wmaps_to_sample_per_cat is None else len(wmaps_to_sample_per_cat)
                
        if self._sampling_type == 0: # fore/background
            if n_wmaps > 0: #Both weight maps should be provided currently.
                if n_wmaps != self.get_n_sampling_cats():
                    self._log.print3("ERROR: For sampling_type = Fore/Background(0), [" + str(n_wmaps) + "] weight maps were provided! "+\
                                    "Two (fore/back) were expected! Exiting!"); exit(1)
                sampling_maps_per_cat = wmaps_to_sample_per_cat
            elif gt_lbl_img is None:
                self._log.print3("ERROR: For sampling_type=[" + self._sampling_type_str + "], if weighted-maps are not provided, "+\
                                "at least Ground Truth labels should be given to extract foreground! Exiting!"); exit(1)
            elif roi_mask is not None: # and provided GT
                mask_to_sample_foregr = ( (gt_lbl_img>0)*(roi_mask>0) ).astype("int8")
                mask_to_sample_backgr = ( (mask_to_sample_foregr==0)*(roi_mask>0) ).astype("int8") # ROI minus foregr.
                sampling_maps_per_cat = [ mask_to_sample_foregr, mask_to_sample_backgr ] #Foreground / Background (in sequence)
            else: # no weightmaps, gt provided and roi is not provided.
                mask_to_sample_foregr = (gt_lbl_img>0).astype("int8")
                mask_to_sample_backgr = np.ones(dims_of_scan, dtype="int8") * (mask_to_sample_foregr==0)
                sampling_maps_per_cat = [ mask_to_sample_foregr, mask_to_sample_backgr ] #Foreground / Background (in sequence)
        elif self._sampling_type == 1: # uniform
            if n_wmaps > 0:
                if n_wmaps != 1:
                    self._log.print3("ERROR: For sampling_type=[" + self._sampling_type_str + "], [" + str(n_wmaps) + "] weight maps were provided! "+\
                                    "One was expected! Exiting!"); exit(1)
                sampling_maps_per_cat = wmaps_to_sample_per_cat #Should be an array with dim1==1 already.
            elif roi_mask is not None:
                sampling_maps_per_cat = [ roi_mask ] #Be careful to not change either of the two arrays later or there'll be a problem.
            else:
                sampling_maps_per_cat = [ np.ones(dims_of_scan, dtype="int8") ]
        elif self._sampling_type == 2: # full image. SAME AS UNIFORM?
            if n_wmaps > 0:
                if n_wmaps != 1:
                    self._log.print3("ERROR: For sampling_type=[" + self._sampling_type_str + "], [" + str(n_wmaps) + "] weight maps were provided! "+\
                                    "One was expected! Exiting!"); exit(1)
                sampling_maps_per_cat = wmaps_to_sample_per_cat #Should be an array with dim1==1 already.
            elif roi_mask is not None:
                sampling_maps_per_cat = [ roi_mask ] #Be careful to not change either of the two arrays later or there'll be a problem.
            else:
                sampling_maps_per_cat = [ np.ones(dims_of_scan, dtype="int8") ]
        elif self._sampling_type == 3: # Targeted per class.
            if n_wmaps > 0:
                if n_wmaps != self.get_n_sampling_cats():
                    self._log.print3("ERROR: Sampling-Type [" + self._sampling_type_str + "], [" + str(n_wmaps) + "] weight maps were provided! "+\
                                     "As many as classes [" + str(self.get_n_sampling_cats()) + "] (incl Background) were expected! Exiting!"); exit(1)
                sampling_maps_per_cat = wmaps_to_sample_per_cat #Should have as many entries as classes (incl backgr).
            elif gt_lbl_img is None:
                self._log.print3("ERROR: Sampling type [3, per-class] requires weight-maps or manual labels for each class! Exiting!"); exit(1)
            elif roi_mask is not None: # and provided GT
                sampling_maps_per_cat = []
                for cat_i in range( self.get_n_sampling_cats() ): # Should be same number as actual classes, including background.
                    sampling_maps_per_cat.append( ( (gt_lbl_img==cat_i)*(roi_mask>0) ).astype("int8") )
            else: # no weightmaps, gt provided and roi is not provided.
                sampling_maps_per_cat = []
                for cat_i in range( self.get_n_sampling_cats() ): # Should be same number as actual classes, including background.
                    sampling_maps_per_cat.append( (gt_lbl_img==cat_i).astype("int8") )
                
        else:
            raise ValueError("Invalid value for sampling type.")
            
        return sampling_maps_per_cat
    
    
    def distribute_n_samples_to_categs(self, n_samples, sampling_maps_per_cat):
        # sampling_maps_per_cat: returned by self.derive_sampling_maps_per_cat(...)
        # The below is a list of booleans, where False if a sampling_map is all 0.
        valid_cats = [ np.sum(s_map) > 0 for s_map in sampling_maps_per_cat ]
        
        # Set weight for sampling a category to 0 if it's not valid.
        perc_samples_per_valid_cat = [p if v else 0. for p,v in zip(self._perc_to_sample_per_cat, valid_cats) ]
        # Renormalize probabilities.
        perc_samples_per_valid_cat = self._normalize_percentages(perc_samples_per_valid_cat)

        # First, distribute n_samples to the categories:
        n_sampl_cats = len(perc_samples_per_valid_cat)
        n_samples_per_cat = np.zeros( n_sampl_cats, dtype="int32" )
        for cat_i in range(n_sampl_cats):
            n_samples_per_cat[cat_i] += int(n_samples * perc_samples_per_valid_cat[cat_i])
        # Distribute samples that were left if perc dont exactly add to 1.
        n_undistributed_samples = n_samples - np.sum(n_samples_per_cat)
        cats_to_distribute_samples = np.random.choice(n_sampl_cats,
                                                      size=n_undistributed_samples,
                                                      replace=True,
                                                      p=perc_samples_per_valid_cat )
        for cat_i in cats_to_distribute_samples:
            n_samples_per_cat[cat_i] += 1
                
        return n_samples_per_cat, valid_cats





