# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from deepmedic.neuralnet.cnn3d import Cnn3d
from deepmedic.neuralnet.pathways import Pathway


# The API for these classes should resemble the API of Pathway and Cnn3d classes.
# But only what is needed by the sampling process of the training procedure.
class PathwayWrapperForSampling:
    # For CnnWrapperForSampling class.
    def __init__(self, pathway: Pathway):
        self._pType = pathway.pType()
        self._subs_factor = pathway.subs_factor()
        self._rec_field = pathway.rec_field()

    def pType(self):
        return self._pType

    def subs_factor(self):
        return self._subs_factor

    def rec_field(self):
        # Used by sampling of low-res context (old version) during training.
        return self._rec_field


class CnnWrapperForSampling:
    # Only for the parallel process used during training. So that it won't re-load theano/tensorflow etc.
    # There was a problem with cnmem when reloading theano.
    def __init__(self, cnn3d: Cnn3d):
        # Cnn
        self.num_classes = cnn3d.num_classes
        # Pathways related
        self._num_pathways_that_require_input = cnn3d.get_num_pathways_that_require_input()
        self.num_subs_paths = cnn3d.num_subs_paths
        
        self.pathways = []
        for pathway_i in range(len(cnn3d.pathways)):
            self.pathways.append(PathwayWrapperForSampling(cnn3d.pathways[pathway_i]))
        
    def get_num_pathways_that_require_input(self):
        return self._num_pathways_that_require_input
