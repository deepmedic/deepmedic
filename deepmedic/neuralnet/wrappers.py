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
        self._shapeOfInputTrainValTest = pathwayInstance.getShapeOfInput()
        self._shapeOfOutputTrainValTest = pathwayInstance.getShapeOfOutput()
    def pType(self):
        return self._pType
    def subsFactor(self):
        return self._subsFactor
    def getShapeOfInput(self):
        return self._shapeOfInputTrainValTest
    def getShapeOfOutput(self):
        return self._shapeOfOutputTrainValTest
        
class CnnWrapperForSampling(object):
    # Only for the parallel process used during training. So that it won't re-load theano etc. There was a problem with cnmem when reloading theano.
    def __init__(self, cnn3dInstance) :
        # Cnn
        self.recFieldCnn = cnn3dInstance.recFieldCnn
        self.batchSizeTrainValTest = [ cnn3dInstance.batchSize, cnn3dInstance.batchSizeValidation, cnn3dInstance.batchSizeTesting ]
        self.finalTargetLayer_outputShapeTrainValTest = [cnn3dInstance.finalTargetLayer.outputShapeTrain,
                                                        cnn3dInstance.finalTargetLayer.outputShapeVal,
                                                        cnn3dInstance.finalTargetLayer.outputShapeTest ]
        # Pathways related
        self._numPathwaysThatRequireInput = cnn3dInstance.getNumPathwaysThatRequireInput()
        self.numSubsPaths = cnn3dInstance.numSubsPaths
        
        self.pathways = []
        for pathway_i in range(len(cnn3dInstance.pathways)) :
            self.pathways.append( PathwayWrapperForSampling(cnn3dInstance.pathways[pathway_i]) )
        
    def getNumPathwaysThatRequireInput(self) :
        return self._numPathwaysThatRequireInput
    
    