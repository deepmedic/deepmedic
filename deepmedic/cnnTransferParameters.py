# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange

from deepmedic.pathwayTypes import PathwayTypes as pt

def transferParametersBetweenModels(myLogger, cnnTarget, cnnSource, listOfLayersToTransfer):
    """
    Transfer parameters from a model to another.
    
    cnnTarget : An instance of deepmedic.cnn3d.Cnn3d. This is the model that will receive the parameters of the pre-trained model.
    cnnSource: Similar to the above. The parameters of this model will be transfered to the above.
    listOfLayersToTransfer: A list of integers. The integers are the depth of the layers of cnnTarget that will be adopted from the pre-trained model.
        First layer is 1. Classification layer of the original 11-layers deep deepmedic is 11.
        The same layers from each parallel-pathway are transfered.
        If [] is given, no layer is transfered.
        If None is given, default functionality follows. Which transfers all layers except the classification layer.  
        Example: In the original deepmedic, with 8 layers at each parallel path followed by 3 FC layers, [1,2,3,4,9,10] will transfer parameters of the 4 first layers of EACH parallel pathway, and 2 hidden FC layers (depth 9 and 10).
    
    Important Note on functionality:
    The two models can be of different depth. But each pathway of cnnSource must have at least as many layers as the amount requested to transfer via listOfLayersToTransfer.
    The two models can have different number of feature maps (FM) at each layer.
    If cnnSource has more FMs than cnnTarget, only the earliest indexed are transfered from each layer.
    If cnnTarget has more FMs than cnnSource, the FMs of the latter will be transfered to the FMs of the former with the smallest indexes and the rest will remain unchanged.
    Number of parallel pathways between the models can differ. If cnnTarget has more pathways than cnnSource, then the lower scale pathways receive weights from the lowest scale pathway availble.
    """    
    depthDeepestLayerTarget = len(cnnTarget.pathways[0].getLayers()) + len(cnnTarget.getFcPathway().getLayers()) # Classif layer. NOT Softmax layer, which is not registered in the FC path.
    # NOTE: Softmax layer HAS learnt BIASES and I must transfer them separately if deepest pathway is asked to be transfered.
    
    for pathTarget_i in xrange( len(cnnTarget.pathways) ):
        pathTarget = cnnTarget.pathways[pathTarget_i]
        typePathTarget = pathTarget.pType()
        layersPathTarget = pathTarget.getLayers()
        
        for layerTarget_i in xrange( len(layersPathTarget) ):
            layerTarget = layersPathTarget[layerTarget_i]
            depthLayerTarget = layerTarget_i + 1 if typePathTarget != pt.FC else layerTarget_i + 1 + len( cnnTarget.pathways[0].getLayers() ) # +1 cause indexes. index 0 is depth 1.
            
            # Check if this layer of Target should receive parameters from Source.
            boolTransferLayer = False
            if listOfLayersToTransfer is None and depthLayerTarget != depthDeepestLayerTarget: # For list == None, we do the default transfer. Transfer all except the deepest classif Layer.
                boolTransferLayer = True
            if listOfLayersToTransfer is not None and depthLayerTarget in listOfLayersToTransfer: 
                boolTransferLayer = True
                
            if boolTransferLayer:
                myLogger.print3("[Pathway_" + str(pathTarget.getStringType()) + "][Conv.Layer_" + str(layerTarget_i) + " (index)], depth [" + str(depthLayerTarget) + "] (Target): Receiving parameters...")
                # transfer stuff.
                # Get the correct Source path.
                if typePathTarget != pt.FC:
                    if (len(cnnSource.pathways) - 1) >= (pathTarget_i + 1) : # if cnnSource has at least as many parallel pathways (-1 to exclude FC) as the number of the current pathwayTarget (+1 because it's index).
                        pathSource = cnnSource.pathways[pathTarget_i]
                    else:
                        pathSource = cnnSource.pathways[-2] # -1 is the FC pathway. -2 is the last parallel.
                        myLogger.print3("\t Source model has less parallel paths than Target. "+\
                                        "Parameters of Target are received from last parallel path of Source [Pathway_" + str(pathSource.getStringType()) + "]")
                else:
                    pathSource = cnnSource.getFcPathway()
                
                # Get the correct Source layer.
                if len(pathSource.getLayers()) < layerTarget_i + 1:
                    myLogger.print3("ERROR: This [Pathway_" + str(pathTarget.getStringType()) + "] of the [Source] model was found to have less layers than required!\n"+\
                                    "\t Number of layers in [Source] pathway: [" + str(len(pathSource.getLayers())) + "].\n"+\
                                    "\t Number of layers in [Target] pathway: [" + str(len(pathTarget.getLayers())) + "].\n"+\
                                    "\t Tried to transfer parameters to [Target] layer with *index* in this pathway: [" + str(layerTarget_i) + "]. (specified depth [" + str(depthLayerTarget) + "]).\n"+\
                                    "\t Note: First layer of pathway has *index* [0].\n"+\
                                    "\t Note#2: To transfer parameters from a Source model with less layers than the Target, specify the depth of layers to transfer using the command line option [-layers].\n"+\
                                    "\t Try [-h] for help or see documentation.\n"+\
                                    "Exiting!"); exit(1)
                
                layerSource_i = layerTarget_i
                layerSource = pathSource.getLayers()[layerSource_i]
                myLogger.print3("\t ...receiving parameters from [Pathway_" + str(pathSource.getStringType()) + "][Conv.Layer_" + str(layerSource_i) + " (index)] (Source).")
                
                # Found Source layer. Excellent. Now just transfer the parameters from Source to Target
                transferParametersBetweenLayers(myLogger=myLogger, layerTarget = layerTarget, layerSource = layerSource )
                
                if depthLayerTarget == depthDeepestLayerTarget: # It's the last Classification layer that was transfered. Also transfer the biases of the Softmax layer, which is not in FC path.
                    myLogger.print3("\t Last Classification layer was transfered. Thus for completeness, transfer the biases applied by the Softmax pseudo-layers. ")
                    softMaxLayerTarget = cnnTarget.finalTargetLayer
                    softMaxLayerSource = cnnSource.finalTargetLayer
                    # This should only transfer biases.
                    transferParametersBetweenLayers(myLogger=myLogger, layerTarget = softMaxLayerTarget, layerSource = softMaxLayerSource )
                    
    return cnnTarget


def transferParametersBetweenLayers(myLogger, layerTarget, layerSource ):
    # VIOLATES _HIDDEN ENCAPSULATION! TEMPORARY TILL I FIX THE API (TILL AFTER DA).
    minNumFmsInLayers = min( layerTarget.getNumberOfFeatureMaps(), layerSource.getNumberOfFeatureMaps() )
    minNumOfChansInInput = min( layerTarget.inputShapeTrain[1], layerSource.inputShapeTrain[1] )
    
    if layerTarget._W is not None and layerSource._W is not None:
        myLogger.print3("\t Transferring weights [W].")
        targetValue = layerTarget._W.get_value()
        sourceValue = layerSource._W.get_value()
        targetValue[:minNumFmsInLayers, :minNumOfChansInInput, :,:,:] = sourceValue[:minNumFmsInLayers, :minNumOfChansInInput, :,:,:]
        layerTarget._W.set_value( targetValue )
    if layerTarget._b is not None and layerSource._b is not None:
        myLogger.print3("\t Transferring biases [b].")
        targetValue = layerTarget._b.get_value()
        sourceValue = layerSource._b.get_value()
        targetValue[:minNumOfChansInInput] = sourceValue[:minNumOfChansInInput]
        layerTarget._b.set_value( targetValue )
    if layerTarget._gBn is not None and layerSource._gBn is not None:
        myLogger.print3("\t Transferring g of Batch Norm [gBn].")
        targetValue = layerTarget._gBn.get_value()
        sourceValue = layerSource._gBn.get_value()
        targetValue[:minNumOfChansInInput] = sourceValue[:minNumOfChansInInput]
        layerTarget._gBn.set_value( targetValue )
    if layerTarget._aPrelu is not None and layerSource._aPrelu is not None:
        myLogger.print3("\t Transferring a of PReLu [aPrelu].")
        targetValue = layerTarget._aPrelu.get_value()
        sourceValue = layerSource._aPrelu.get_value()
        targetValue[:minNumOfChansInInput] = sourceValue[:minNumOfChansInInput]
        layerTarget._aPrelu.set_value( targetValue )
        
    # For the rolling average used in inference by Batch-Norm.
    minLengthRollAvForBn = min( layerTarget._rollingAverageForBatchNormalizationOverThatManyBatches, layerSource._rollingAverageForBatchNormalizationOverThatManyBatches )
    if layerTarget._muBnsArrayForRollingAverage is not None and layerSource._muBnsArrayForRollingAverage is not None:
        myLogger.print3("\t Transferring rolling average of MU of Batch Norm [muBnsArrayForRollingAverage].")
        targetValue = layerTarget._muBnsArrayForRollingAverage.get_value()
        sourceValue = layerSource._muBnsArrayForRollingAverage.get_value()
        targetValue[:minLengthRollAvForBn, :minNumOfChansInInput] = sourceValue[:minLengthRollAvForBn, :minNumOfChansInInput]
        layerTarget._muBnsArrayForRollingAverage.set_value( targetValue )
    if layerTarget._varBnsArrayForRollingAverage is not None and layerSource._varBnsArrayForRollingAverage is not None:
        myLogger.print3("\t Transferring rolling average of Variance of Batch Norm [varBnsArrayForRollingAverage].")
        targetValue = layerTarget._varBnsArrayForRollingAverage.get_value()
        sourceValue = layerSource._varBnsArrayForRollingAverage.get_value()
        targetValue[:minLengthRollAvForBn, :minNumOfChansInInput] = sourceValue[:minLengthRollAvForBn, :minNumOfChansInInput]
        layerTarget._varBnsArrayForRollingAverage.set_value( targetValue )


