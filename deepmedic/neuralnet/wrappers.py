# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division


# The API for these classes should resemble the API of Pathway and Cnn3d classes.
# But only what is needed by the sampling process of the training procedure.
class PathwayWrapperForSampling(object):
    # For CnnWrapperForSampling class.
    def __init__(self, pathwayInstance):
        self._pType = pathwayInstance.pType()
        self._subs_factor = pathwayInstance.subs_factor()
        self._rec_field = pathwayInstance.rec_field()

    def pType(self):
        return self._pType
    def subs_factor(self):
        return self._subs_factor
    def rec_field(self): # Used by sampling of low-res context (old version) during training.
        return self._rec_field

class CnnWrapperForSampling(object):
    # Only for the parallel process used during training. So that it won't re-load theano/tensorflow etc.
    # There was a problem with cnmem when reloading theano.
    def __init__(self, cnn3d) :
        # Cnn
        self.num_classes = cnn3d.num_classes
        # Pathways related
        self._numPathwaysThatRequireInput = cnn3d.getNumPathwaysThatRequireInput()
        self.numSubsPaths = cnn3d.numSubsPaths
        
        self.pathways = []
        for pathway_i in range(len(cnn3d.pathways)) :
            self.pathways.append( PathwayWrapperForSampling(cnn3d.pathways[pathway_i]) )
        
    def getNumPathwaysThatRequireInput(self) :
        return self._numPathwaysThatRequireInput
    
    
