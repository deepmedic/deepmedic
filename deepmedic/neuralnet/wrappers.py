# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division


# The API for these classes should resemble the API of Pathway and Cnn3d classes. But only what is needed by the sampling process of the training procedure.
class PathwayWrapperForSampling(object):
    # For CnnWrapperForSampling class.
    def __init__(self, pathwayInstance) :
        self._pType = pathwayInstance.pType()
        self._subsFactor = pathwayInstance.subsFactor()
        self._inputShape = {"train": pathwayInstance.getShapeOfInput("train"),
                              "val": pathwayInstance.getShapeOfInput("val"),
                              "test": pathwayInstance.getShapeOfInput("test")}
    def pType(self):
        return self._pType
    def subsFactor(self):
        return self._subsFactor
    def getShapeOfInput(self, train_val_test_str):
        assert train_val_test_str in ["train", "val", "test"]
        return self._inputShape[train_val_test_str]
        
class CnnWrapperForSampling(object):
    # Only for the parallel process used during training. So that it won't re-load theano/tensorflow etc. There was a problem with cnmem when reloading theano.
    def __init__(self, cnn3d) :
        # Cnn
        self.num_classes = cnn3d.num_classes
        self.recFieldCnn = cnn3d.recFieldCnn
        self.finalTargetLayer_outputShape = {"train": cnn3d.finalTargetLayer.outputShape["train"],
                                             "val": cnn3d.finalTargetLayer.outputShape["val"],
                                             "test": cnn3d.finalTargetLayer.outputShape["test"]}
        # Pathways related
        self._numPathwaysThatRequireInput = cnn3d.getNumPathwaysThatRequireInput()
        self.numSubsPaths = cnn3d.numSubsPaths
        
        self.pathways = []
        for pathway_i in range(len(cnn3d.pathways)) :
            self.pathways.append( PathwayWrapperForSampling(cnn3d.pathways[pathway_i]) )
        
    def getNumPathwaysThatRequireInput(self) :
        return self._numPathwaysThatRequireInput
    
    