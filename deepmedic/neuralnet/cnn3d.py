# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import xrange
import numpy as np
import random
from math import ceil

import theano
import theano.tensor as T

from deepmedic.neuralnet.pathwayTypes import PathwayTypes as pt
from deepmedic.neuralnet.pathways import NormalPathway, SubsampledPathway, FcPathway
from deepmedic.neuralnet.cnnLayerTypes import SoftmaxLayer

from deepmedic.neuralnet.cnnHelpers import calculateReceptiveFieldDimensionsFromKernelsDimListPerLayerForFullyConvCnnWithStrides1

#-----helper functions that I use in here---

def padImageWithMirroring(inputImage, voxelsPerDimToPad) :
    # inputImage shape: [batchSize, #channels#, r, c, z]
    # inputImageDimensions : [ batchSize, #channels, dim r, dim c, dim z ] of inputImage
    # voxelsPerDimToPad shape: [ num o voxels in r-dim to add, ...c-dim, ...z-dim ]
    # If voxelsPerDimToPad is odd, 1 more voxel is added to the right side.
    # r-axis
    assert np.all(voxelsPerDimToPad) >= 0
    padLeft = int(voxelsPerDimToPad[0] // 2); padRight = int((voxelsPerDimToPad[0] + 1) // 2);
    paddedImage = T.concatenate([inputImage[:, :, int(voxelsPerDimToPad[0] // 2) - 1::-1 , :, :], inputImage], axis=2) if padLeft > 0 else inputImage
    paddedImage = T.concatenate([paddedImage, paddedImage[ :, :, -1:-1 - int((voxelsPerDimToPad[0] + 1) // 2):-1, :, :]], axis=2) if padRight > 0 else paddedImage
    # c-axis
    padLeft = int(voxelsPerDimToPad[1] // 2); padRight = int((voxelsPerDimToPad[1] + 1) // 2);
    paddedImage = T.concatenate([paddedImage[:, :, :, padLeft - 1::-1 , :], paddedImage], axis=3) if padLeft > 0 else paddedImage
    paddedImage = T.concatenate([paddedImage, paddedImage[:, :, :, -1:-1 - padRight:-1, :]], axis=3) if padRight > 0 else paddedImage
    # z-axis
    padLeft = int(voxelsPerDimToPad[2] // 2); padRight = int((voxelsPerDimToPad[2] + 1) // 2)
    paddedImage = T.concatenate([paddedImage[:, :, :, :, padLeft - 1::-1 ], paddedImage], axis=4) if padLeft > 0 else paddedImage
    paddedImage = T.concatenate([paddedImage, paddedImage[:, :, :, :, -1:-1 - padRight:-1]], axis=4) if padRight > 0 else paddedImage
    
    return paddedImage


##################################################
##################################################
################ THE CNN CLASS ###################
##################################################
##################################################

class Cnn3d(object):
    def __init__(self):
        
        self.cnnModelName = None
        
        self.pathways = [] # There should be only 1 normal and only one FC pathway. Eg, see self.getFcPathway()
        self.numSubsPaths = 0
        
        self.finalTargetLayer = ""
        
        self.numberOfOutputClasses = None
        
        #=== Compiled Functions for API ====
        self.cnnTrainModel = ""
        self.cnnValidateModel = ""
        self.cnnTestModel = ""
        self.cnnVisualiseFmFunction = ""
        
        #=====================================
        self.recFieldCnn = ""
        
        self.borrowFlag = ""
        
        self.batchSize = ""
        self.batchSizeValidation = ""
        self.batchSizeTesting = ""
        
        # self.patchesToTrainPerImagePart = ""
        self.dataTypeX = ""
        self.nkerns = ""  # number of feature maps.
        self.nkernsSubsampled = ""
        
        # Fully Connected Layers
        self.kernelDimensionsFirstFcLayer = ""
        
        # Automatically lower CNN's learning rate by looking at validation accuracy:
        self.topMeanValidationAccuracyAchievedInEpoch = [-1, -1]
        self.lastEpochAtTheEndOfWhichLrWasLowered = 0  # refers to CnnTrained epochs, not the epochs in the do_training loop.
        
        # Residual Learning
        self.indicesOfLayersToConnectResidualsInOutput = ""
        
        # Lower rank convolutional layers
        self.indicesOfLowerRankLayersPerPathway = ""
        self.ranksOfLowerRankLayersForEachPathway = ""
        
        
        # ======= Shared Variables with X and Y data for training/validation/testing ======
        self._initializedSharedVarsTrain = False
        self.sharedInpXTrain = ""
        self.sharedInpXPerSubsListTrain = []
        self.sharedLabelsYTrain = ""
        self._initializedSharedVarsVal = False
        self.sharedInpXVal = ""
        self.sharedInpXPerSubsListVal = []
        self.sharedLabelsYVal = ""
        self._initializedSharedVarsTest = False
        self.sharedInpXTest = ""
        self.sharedInpXPerSubsListTest = []
        
        
        #============= ATTRIBUTES SPECIFIC TO THE TRAINING STATE ============
        self.numberOfEpochsTrained = 0
        
        self._trainingStateAttributesInitialized = False
        
        self.indicesOfLayersPerPathwayTypeToFreeze = None
        self.costFunctionLetter = ""  # "L", "D" or "J"
        #====== Learning rate and momentum ==========
        self.initialLearningRate = ""  # used by exponential schedule
        self.learning_rate = theano.shared(np.cast["float32"](0.01))  # initial value, changed in make_cnn_model().compileTrainingFunction()
        self.classicMomentum0OrNesterov1 = None
        # SGD + Classic momentum: (to save the momentum)
        self.initialMomentum = ""  # used by exponential schedule
        self.momentum = theano.shared(np.cast["float32"](0.))
        self.momentumTypeNONNormalized0orNormalized1 = None
        self.velocities_forMom = []  # list of shared_variables. Each of the individual Dws is a sharedVar. This whole thing isnt.
        #=== Optimizer specific =====
        self.sgd0orAdam1orRmsProp2 = None
        # ADAM:
        self.b1_adam = None
        self.b2_adam = None
        self.epsilonForAdam = None
        self.i_adam = theano.shared(np.cast["float32"](0.))  # Current iteration of adam
        self.m_listForAllParamsAdam = []  # list of mean of grads for all parameters, for ADAM optimizer.
        self.v_listForAllParamsAdam = []  # list of variances of grads for all parameters, for ADAM optimizer.
        # RMSProp
        self.rho_rmsProp = None
        self.epsilonForRmsProp = None
        self.accuGradSquare_listForAllParamsRmsProp = []  # the rolling average accumulator of the variance of the grad (grad^2)
        # Regularisation
        self.L1_reg_constant = None
        self.L2_reg_constant = None
        
        
        #======= tensors, input to the CNN. Needed to be saved for later compilation after loading =======
        # Symbolic variables, which stand for the input. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
        self.inputTensorsXToCnnInitialized = False
        self.inputTensorNormTrain = None; self.inputTensorNormVal = None; self.inputTensorNormTest = None;
        self.listInputTensorPerSubsTrain = []; self.listInputTensorPerSubsVal = []; self.listInputTensorPerSubsTest = [];
        
    def getNumSubsPathways(self):
        count = 0
        for pathway in self.pathways :
            if pathway.pType() ==  pt.SUBS :
                count += 1
        return count
    
    def getNumPathwaysThatRequireInput(self):
        count = 0
        for pathway in self.pathways :
            if pathway.pType() != pt.FC :
                count += 1
        return count
    
    def getFcPathway(self):
        for pathway in self.pathways :
            if pathway.pType() == pt.FC :
                return pathway
        return None
    
    def increaseNumberOfEpochsTrained(self):
        self.numberOfEpochsTrained += 1
        
    def change_learning_rate_of_a_cnn(self, newValueForLearningRate, myLogger=None) :
        stringToPrint = "UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) + ") Changing the Cnn's Learning Rate to: " + str(newValueForLearningRate)
        if myLogger != None :
            myLogger.print3(stringToPrint)
        else :
            print(stringToPrint)
        self.learning_rate.set_value(newValueForLearningRate)
        self.lastEpochAtTheEndOfWhichLrWasLowered = self.numberOfEpochsTrained
        
    def divide_learning_rate_of_a_cnn_by(self, divideLrBy, myLogger=None) :
        oldLR = self.learning_rate.get_value()
        newValueForLearningRate = oldLR * 1.0 / divideLrBy
        self.change_learning_rate_of_a_cnn(newValueForLearningRate, myLogger)
        
    def change_momentum_of_a_cnn(self, newValueForMomentum, myLogger=None):
        stringToPrint = "UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) + ") Changing the Cnn's Momentum to: " + str(newValueForMomentum)
        if myLogger != None :
            myLogger.print3(stringToPrint)
        else :
            print(stringToPrint)
        self.momentum.set_value(newValueForMomentum)
        
    def multiply_momentum_of_a_cnn_by(self, multiplyMomentumBy, myLogger=None) :
        oldMom = self.momentum.get_value()
        newValueForMomentum = oldMom * multiplyMomentumBy
        self.change_momentum_of_a_cnn(newValueForMomentum, myLogger)
        
    def changeB1AndB2ParametersOfAdam(self, b1ParamForAdam, b2ParamForAdam, myLogger) :
        myLogger.print3("UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) + ") Changing the Cnn's B1 and B2 parameters for ADAM optimization to: B1 = " + str(b1ParamForAdam) + " || B2 = " + str(b2ParamForAdam))
        self.b1_adam = b1ParamForAdam
        self.b2_adam = b2ParamForAdam
        
    def changeRhoParameterOfRmsProp(self, rhoParamForRmsProp, myLogger) :
        myLogger.print3("UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) + ") Changing the Cnn's Rho parameter for RMSProp optimization to: Rho = " + str(rhoParamForRmsProp))
        self.rho_rmsProp = rhoParamForRmsProp
        
    def checkMeanValidationAccOfLastEpochAndUpdateCnnsTopAccAchievedIfNeeded(self,
                                                                        myLogger,
                                                                        meanValidationAccuracyOfLastEpoch,
                                                                        minIncreaseInValidationAccuracyConsideredForLrSchedule) :
        # Called at the end of an epoch, right before increasing self.numberOfEpochsTrained
        highestAchievedValidationAccuracyOfCnn = self.topMeanValidationAccuracyAchievedInEpoch[0]
        if meanValidationAccuracyOfLastEpoch > highestAchievedValidationAccuracyOfCnn + minIncreaseInValidationAccuracyConsideredForLrSchedule :
            self.topMeanValidationAccuracyAchievedInEpoch[0] = meanValidationAccuracyOfLastEpoch
            self.topMeanValidationAccuracyAchievedInEpoch[1] = self.numberOfEpochsTrained
            myLogger.print3("UPDATE: In this last epoch (cnnTrained) #" + str(self.topMeanValidationAccuracyAchievedInEpoch[1]) + " the CNN achieved a new highest mean validation accuracy of: " + str(self.topMeanValidationAccuracyAchievedInEpoch[0]))
            
    def _initializeSharedVarsForInputsTrain(self) :
        # ======= Initialize sharedVariables ==========
        self._initializedSharedVarsTrain = True
        # Create the needed shared variables. Number of dimensions should be correct (5 for x, 4 for y). But size is placeholder. Changes when shared.set_value during training.
        self.sharedInpXTrain = theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag)
        for subsPath_i in xrange(self.numSubsPaths) :
            self.sharedInpXPerSubsListTrain.append(theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag))
        # When storing data on the GPU it has to be stored as floats (floatX). Later this variable is cast as "int", to be used correctly in computations.
        self.sharedLabelsYTrain = theano.shared(np.zeros([1, 1, 1, 1], dtype="float32") , borrow=self.borrowFlag)
        
    def _initializeSharedVarsForInputsVal(self) :
        self._initializedSharedVarsVal = True
        self.sharedInpXVal = theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag)
        for subsPath_i in xrange(self.numSubsPaths) :
            self.sharedInpXPerSubsListVal.append(theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag))
        self.sharedLabelsYVal = theano.shared(np.zeros([1, 1, 1, 1], dtype="float32") , borrow=self.borrowFlag)
        
    def _initializeSharedVarsForInputsTest(self) :
        self._initializedSharedVarsTest = True
        self.sharedInpXTest = theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag)
        for subsPath_i in xrange(self.numSubsPaths) :
            self.sharedInpXPerSubsListTest.append(theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag))
            
    def freeGpuTrainingData(self) :
        if self._initializedSharedVarsTrain :  # False if this is called (eg save model) before train/val/test function is compiled.
            self.sharedInpXTrain.set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))  # = []
            for subsPath_i in xrange(self.numSubsPaths) :
                self.sharedInpXPerSubsListTrain[subsPath_i].set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))
            self.sharedLabelsYTrain.set_value(np.zeros([1, 1, 1, 1], dtype="float32"))  # = []
            
    def freeGpuValidationData(self) :
        if self._initializedSharedVarsVal :
            self.sharedInpXVal.set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))  # = []
            for subsPath_i in xrange(self.numSubsPaths) :
                self.sharedInpXPerSubsListVal[subsPath_i].set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))
            self.sharedLabelsYVal.set_value(np.zeros([1, 1, 1, 1], dtype="float32"))  # = []
            
    def freeGpuTestingData(self) :
        if self._initializedSharedVarsTest :
            self.sharedInpXTest.set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))  # = []
            for subsPath_i in xrange(self.numSubsPaths) :
                self.sharedInpXPerSubsListTest[subsPath_i].set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))
                
    def checkTrainingStateAttributesInitialized(self):
        return self._trainingStateAttributesInitialized
    
    # for inference with batch-normalization. Every training batch, this is called to update an internal matrix of each layer, with the last mus and vars, so that I can compute the rolling average for inference.
    def updateTheMatricesOfTheLayersWithTheLastMusAndVarsForTheMovingAverageOfTheBatchNormInference(self) :
        self._updateMatricesOfBnRollingAverageForInference()
        
    def _updateMatricesOfBnRollingAverageForInference(self):
        for pathway in self.pathways :
            for layer in pathway.getLayers() :
                layer.updateTheMatricesWithTheLastMusAndVarsForTheRollingAverageOfBNInference()  # Will do nothing if no BN.
                
    #========================================OPTIMIZERS========================================
    """
    From https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617 :
    ClassicMomentum:
    (1) v_t = mu * v_t-1 - lr * gradient_f(params_t)
    (2) params_t = params_t-1 + v_t
    (3) params_t = params_t-1 + mu * v_t-1 - lr * gradient_f(params_t-1)
    
    Nesterov momentum:
    (4) v_t = mu * v_t-1 - lr * gradient_f(params_t-1 + mu * v_t-1)
    (5) params_t = params_t-1 + v_t
    
    alternate formulation for Nesterov momentum:
    (6) v_t = mu * v_t-1 - lr * gradient_f(params_t-1)
    (7) params_t = params_t-1 + mu * v_t - lr * gradient_f(params_t-1)
    (8) params_t = params_t-1 + mu**2 * v_t-1 - (1+mu) * lr * gradient_f(params_t-1)
    
    Can also find help for optimizers in Lasagne: https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
    """
    
    def _initializeSharedVariablesOfOptimizer(self, myLogger) :
        # ======= Get List Of Trained Parameters to be fit by gradient descent=======
        paramsToOptDuringTraining = self._getTrainableParameters(myLogger)
        if self.sgd0orAdam1orRmsProp2 == 0 :
            self._initializeSharedVariablesOfSgd(paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 1 :
            self._initializeSharedVariablesOfAdam(paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 2 :
            self._initializeSharedVariablesOfRmsProp(paramsToOptDuringTraining)
        else :
            return False
        return True
        
    def _initializeSharedVariablesOfSgd(self, paramsToOptDuringTraining) :
        self.velocities_forMom = []
        for param in paramsToOptDuringTraining :
            v = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            self.velocities_forMom.append(v)
            
    def getUpdatesAccordingToSgd(self, cost, paramsToOptDuringTraining) :
        # create a list of gradients for all model parameters
        grads = T.grad(cost, paramsToOptDuringTraining)
        
        updates = []
        # The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self.momentum * self.momentumTypeNONNormalized0orNormalized1
        
        for param, grad, v in zip(paramsToOptDuringTraining, grads, self.velocities_forMom) :
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum * self.learning_rate * grad
            newVelocity = self.momentum * v - stepToGradientDirection
            
            if self.classicMomentum0OrNesterov1 == 0 :
                updateToParam = newVelocity
            else :  # Nesterov
                updateToParam = self.momentum * newVelocity - stepToGradientDirection
                
            updates.append((v, newVelocity))  # I can do (1-mom)*learnRate*grad.
            updates.append((param, param + updateToParam))
            
        return updates
    
    def _initializeSharedVariablesOfRmsProp(self, paramsToOptDuringTraining) :
        self.accuGradSquare_listForAllParamsRmsProp = []
        self.velocities_forMom = []
        for param in paramsToOptDuringTraining :
            accu = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)  # accumulates the mean of the grad's square.
            self.accuGradSquare_listForAllParamsRmsProp.append(accu)
            v = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)  # velocity
            self.velocities_forMom.append(v)
            
    def getUpdatesAccordingToRmsProp(self, cost, params) :
        # epsilon=1e-4 in paper. I got NaN in cost function when I ran it with this value. Worked ok with epsilon=1e-6.
        
        # Code taken and updated (it was V2 of paper, updated to V8) from https://gist.github.com/Newmu/acb738767acb4788bac3
        grads = T.grad(cost, params)
        updates = []
        # The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self.momentum * self.momentumTypeNONNormalized0orNormalized1
        
        for param, grad, accu, v in zip(params, grads, self.accuGradSquare_listForAllParamsRmsProp, self.velocities_forMom):
            accu_new = self.rho_rmsProp * accu + (1 - self.rho_rmsProp) * T.sqr(grad)
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum * (self.learning_rate * grad / T.sqrt(accu_new + self.epsilonForRmsProp))
            newVelocity = self.momentum * v - stepToGradientDirection
            
            if self.classicMomentum0OrNesterov1 == 0 :
                updateToParam = newVelocity
            else :  # Nesterov
                updateToParam = self.momentum * newVelocity - stepToGradientDirection
                
            updates.append((accu, accu_new))
            updates.append((v, newVelocity))  # I can do (1-mom)*learnRate*grad.
            updates.append((param, param + updateToParam))
            
        return updates
    
    def _initializeSharedVariablesOfAdam(self, paramsToOptDuringTraining) :
        self.i_adam = theano.shared(np.cast["float32"](0.))  # Current iteration
        self.m_listForAllParamsAdam = []  # list of mean of grads for all parameters, for ADAM optimizer.
        self.v_listForAllParamsAdam = []  # list of variances of grads for all parameters, for ADAM optimizer.
        for param in paramsToOptDuringTraining :
            m = theano.shared(param.get_value() * 0.)
            self.m_listForAllParamsAdam.append(m)
            v = theano.shared(param.get_value() * 0.)
            self.v_listForAllParamsAdam.append(v)
            
    def getUpdatesAccordingToAdam(self, cost, params) :
        # Epsilon on paper was 10**(-8).
        # Code is on par with version V8 of Kingma's paper.
        grads = T.grad(cost, params)
        
        updates = []
        
        i = self.i_adam
        i_t = i + 1.
        fix1 = 1. - (self.b1_adam)**i_t
        fix2 = 1. - (self.b2_adam)**i_t
        lr_t = self.learning_rate * (T.sqrt(fix2) / fix1)
        for param, grad, m, v in zip(params, grads, self.m_listForAllParamsAdam, self.v_listForAllParamsAdam):
            m_t = (self.b1_adam * m) + ((1. - self.b1_adam) * grad)
            v_t = (self.b2_adam * v) + ((1. - self.b2_adam) * T.sqr(grad))  # Double check this with the paper.
            grad_t = m_t / (T.sqrt(v_t) + self.epsilonForAdam)
            param_t = param - (lr_t * grad_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, param_t))
        updates.append((i, i_t))
        return updates
    
    def _getUpdatesOfTrainableParameters(self, myLogger, cost) :
        # ======= Get List Of Trained Parameters to be fit by gradient descent=======
        paramsToOptDuringTraining = self._getTrainableParameters(myLogger)
        if self.sgd0orAdam1orRmsProp2 == 0 :
            myLogger.print3("Optimizer used: [SGD]. Momentum used: Classic0 or Nesterov1 : " + str(self.classicMomentum0OrNesterov1))
            updates = self.getUpdatesAccordingToSgd(cost, paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 1 :
            myLogger.print3("Optimizer used: [ADAM]. No momentum implemented for Adam.")
            updates = self.getUpdatesAccordingToAdam(cost, paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 2 :
            myLogger.print3("Optimizer used: [RMSProp]. Momentum used: Classic0 or Nesterov1 : " + str(self.classicMomentum0OrNesterov1))
            updates = self.getUpdatesAccordingToRmsProp(cost, paramsToOptDuringTraining)
        return updates
    
    def _getTrainableParameters(self, myLogger):
        # A getter. Don't alter anything here!
        paramsToOptDuringTraining = []  # Ws and Bs
        for pathway in self.pathways :
            for layer_i in xrange(0, len(pathway.getLayers())) :
                if layer_i not in self.indicesOfLayersPerPathwayTypeToFreeze[ pathway.pType() ] :
                    paramsToOptDuringTraining = paramsToOptDuringTraining + pathway.getLayer(layer_i).getTrainableParams()
                else : # Layer will be held fixed. Notice that Batch Norm parameters are still learnt.
                    myLogger.print3("WARN: [Pathway_" + str(pathway.getStringType()) + "] The weights of [Layer-"+str(layer_i)+"] will NOT be trained as specified (index, first layer is 0).")
        return paramsToOptDuringTraining
    
    def _getL1RegCost(self) :
        L1 = 0
        for pathway in self.pathways :
            for layer in pathway.getLayers() :    
                L1 += layer.getL1RegCost()
        return L1
    
    def _getL2RegCost(self) :
        L2_sqr = 0
        for pathway in self.pathways :
            for layer in pathway.getLayers() :    
                L2_sqr += layer.getL2RegCost()
        return L2_sqr
    
    # This function should be called at least once prior to compiling train function for the first time. 
    # If I need to "resume" training, this should not be called.
    # However, if I need to use a pretrained model, and train it in a second stage, I should recall this, with the new stage's parameters, and then recompile trainFunction.
    def initializeTrainingState(self,
                                myLogger,
                                indicesOfLayersPerPathwayTypeToFreeze,
                                costFunctionLetter,
                                learning_rate,
                                sgd0orAdam1orRmsProp2,
                                classicMomentum0OrNesterov1,
                                momentum,
                                momentumTypeNONNormalized0orNormalized1,
                                b1ParamForAdam,
                                b2ParamForAdam,
                                epsilonForAdam,
                                rhoParamForRmsProp,
                                epsilonForRmsProp,
                                L1_reg_constant,
                                L2_reg_constant
                                ) :
        myLogger.print3("...Initializing attributes for the optimization...")
        self.numberOfEpochsTrained = 0
        
        # Layers to train (rest are left untouched, eg for pretrained models.
        self.indicesOfLayersPerPathwayTypeToFreeze = indicesOfLayersPerPathwayTypeToFreeze
        
        # Cost function
        if costFunctionLetter != "previous" :
            self.costFunctionLetter = costFunctionLetter
            
        # Regularization
        self.L1_reg_constant = L1_reg_constant
        self.L2_reg_constant = L2_reg_constant
        
        # Learning rate and momentum
        self.initialLearningRate = learning_rate # This is important for the learning rate schedule to work.
        self.change_learning_rate_of_a_cnn(learning_rate, myLogger)
        self.classicMomentum0OrNesterov1 = classicMomentum0OrNesterov1
        self.initialMomentum = momentum
        self.momentumTypeNONNormalized0orNormalized1 = momentumTypeNONNormalized0orNormalized1
        self.change_momentum_of_a_cnn(momentum, myLogger)
        
        # Optimizer
        self.sgd0orAdam1orRmsProp2 = sgd0orAdam1orRmsProp2
        if sgd0orAdam1orRmsProp2 == 1 :
            self.changeB1AndB2ParametersOfAdam(b1ParamForAdam, b2ParamForAdam, myLogger)
            self.epsilonForAdam = epsilonForAdam
        elif sgd0orAdam1orRmsProp2 == 2 :
            self.changeRhoParameterOfRmsProp(rhoParamForRmsProp, myLogger)
            self.epsilonForRmsProp = epsilonForRmsProp
            
        # Important point. Initializing the shareds that hold the velocities etc states of the optimizers.
        self._initializeSharedVariablesOfOptimizer(myLogger)
        
        self._trainingStateAttributesInitialized = True
        
    def _getUpdatesForBnRollingAverage(self) :
        # These are not the variables of the normalization of the FMs' distributions that are optimized during training. These are only the Mu and Stds that are used during inference,
        # ... and here we update the sharedVariable which is used "from the outside during do_training()" to update the rolling-average-matrix for inference. Do for all layers.
        updatesForBnRollingAverage = []
        for pathway in self.pathways :
            for layer in pathway.getLayers() :
                updatesForBnRollingAverage.extend(layer.getUpdatesForBnRollingAverage())  #CAREFUL: WARN, PROBLEM, THEANO BUG! If a layer has only 1FM, the .newMu_B ends up being of type (true,) instead of vector!!! Error!!!
        return updatesForBnRollingAverage
    
    # NOTE: compileTrainFunction() changes the self.initialLearningRate. Which is used for the exponential schedule!
    def compileTrainFunction(self, myLogger) :
        # At the next stage of the refactoring:
        # 1. Take an additional variable that says whether to "initialize" new training, or to "resume" training
        # 2. Build model here. Which internally LOADS the weights, array made by newModel. Dont initialize a model here. If you want to pretrain, have a -pretrainedModel function to create a new model.
        # 3. initializeTrainingState() if the input variable (1) says so (eg to do another training stage). Otherwise, dont call it, to resume training.
        myLogger.print3("...Building the training function...")
        
        if not self.checkTrainingStateAttributesInitialized() :
            myLogger.print3("ERROR: Prior to compiling the training function, training state attributes need to be initialized via a call of [Cnn3d.setTrainingStateAttributes(...)]. Exiting!"); exit(1)
            
        self._initializeSharedVarsForInputsTrain()
        
        # symbolic variables needed:
        index = T.lscalar()
        x = self.inputTensorNormTrain
        listXPerSubs = self.listInputTensorPerSubsTrain
        
        y = T.itensor4('y')  # Input of the theano-compiled-function. Dimensions of y labels: [batchSize, r, c, z]
        # When storing data on the GPU it has to be stored as floats (floatX). Thus the sharedVariable is FloatX/32. Here this variable is cast as "int", to be used correctly in computations.
        intCastSharedLabelsYTrain = T.cast(self.sharedLabelsYTrain, 'int32')
        inputVectorWeightsOfClassesInCostFunction = T.fvector()  # These two were added to counter class imbalance by changing the weights in the cost function
        weightPerClass = T.fvector()  # a vector with 1 element per class.
        
        #==========================COST FUNCTION=======================
        # The cost Function to use.
        if self.costFunctionLetter == "L" :
            costFromLastLayer = self.finalTargetLayer.negativeLogLikelihood(y, weightPerClass)
        else :
            myLogger.print3("ERROR: Problem in make_cnn_model(). The parameter self.costFunctionLetter did not have an acceptable value( L,D,J ). Exiting."); exit(1)
            
        cost = (costFromLastLayer
                + self.L1_reg_constant * self._getL1RegCost()
                + self.L2_reg_constant * self._getL2RegCost())
        
        #============================OPTIMIZATION=============================
        updates = self._getUpdatesOfTrainableParameters(myLogger, cost)
        
        #================BATCH NORMALIZATION ROLLING AVERAGE UPDATES======================
        updates = updates + self._getUpdatesForBnRollingAverage()
        
        #========================COMPILATION OF FUNCTIONS =================
        givensSet = { x: self.sharedInpXTrain[index * self.batchSize: (index + 1) * self.batchSize] }
        for subPath_i in xrange(self.numSubsPaths) : # if there are subsampled paths...
            xSub = listXPerSubs[subPath_i]
            sharedInpXSubTrain = self.sharedInpXPerSubsListTrain[subPath_i]
            givensSet.update({ xSub: sharedInpXSubTrain[index * self.batchSize: (index + 1) * self.batchSize] })
        givensSet.update({  y: intCastSharedLabelsYTrain[index * self.batchSize: (index + 1) * self.batchSize],
                            weightPerClass: inputVectorWeightsOfClassesInCostFunction })
        
        myLogger.print3("...Compiling the function for training... (This may take a few minutes...)")
        self.cnnTrainModel = theano.function(
                                [index, inputVectorWeightsOfClassesInCostFunction],
                                [cost] + self.finalTargetLayer.getRpRnTpTnForTrain0OrVal1(y, 0),
                                updates=updates,
                                givens=givensSet
                                )
        myLogger.print3("The training function was compiled.")
        
    def compileValidationFunction(self, myLogger) :
        myLogger.print3("...Building the validation function...")
        
        self._initializeSharedVarsForInputsVal()
        
        # symbolic variables needed:
        index = T.lscalar()
        x = self.inputTensorNormVal
        listXPerSubs = self.listInputTensorPerSubsVal
        y = T.itensor4('y')  # Input of the theano-compiled-function. Dimensions of y labels: [batchSize, r, c, z]
        # When storing data on the GPU it has to be stored as floats (floatX). Thus the sharedVariable is FloatX/32. Here this variable is cast as "int", to be used correctly in computations.
        intCastSharedLabelsYVal = T.cast(self.sharedLabelsYVal, 'int32')
        
        givensSet = { x: self.sharedInpXVal[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation] }
        for subPath_i in xrange(self.numSubsPaths) : # if there are subsampled paths...
            xSub = listXPerSubs[subPath_i]
            sharedInpXSubVal = self.sharedInpXPerSubsListVal[subPath_i]
            givensSet.update({ xSub: sharedInpXSubVal[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation] })
        givensSet.update({ y: intCastSharedLabelsYVal[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation] })
        
        myLogger.print3("...Compiling the function for validation... (This may take a few minutes...)")
        self.cnnValidateModel = theano.function(
                                    [index],
                                    self.finalTargetLayer.getRpRnTpTnForTrain0OrVal1(y, 1),
                                    givens=givensSet
                                    )
        myLogger.print3("The validation function was compiled.")
        
        
    def compileTestAndVisualisationFunction(self, myLogger) :
        myLogger.print3("...Building the function for testing and visualisation of FMs...")
        
        self._initializeSharedVarsForInputsTest()
        
        # symbolic variables needed:
        index = T.lscalar()
        x = self.inputTensorNormTest
        listXPerSubs = self.listInputTensorPerSubsTest
        
        listToReturnWithAllTheFmActivationsAndPredictionsAppended = []
        for pathway in self.pathways :
            for layer in pathway.getLayers() :  # each layer that this pathway/fc has.
                listToReturnWithAllTheFmActivationsAndPredictionsAppended.append( layer.fmsActivations([0, 9999]) )
                
        listToReturnWithAllTheFmActivationsAndPredictionsAppended.append(self.finalTargetLayer.predictionProbabilities())
        
        givensSet = { x: self.sharedInpXTest[index * self.batchSizeTesting: (index + 1) * self.batchSizeTesting] }
        for subPath_i in xrange(self.numSubsPaths) : # if there are subsampled paths...
            xSub = listXPerSubs[subPath_i]
            sharedInpXSubTest = self.sharedInpXPerSubsListTest[subPath_i]
            givensSet.update({ xSub: sharedInpXSubTest[index * self.batchSizeTesting: (index + 1) * self.batchSizeTesting] })
            
        myLogger.print3("...Compiling the function for testing and visualisation of FMs... (This may take a few minutes...)")
        self.cnnTestAndVisualiseAllFmsFunction = theano.function(
                                                        [index],
                                                        listToReturnWithAllTheFmActivationsAndPredictionsAppended,
                                                        givens=givensSet
                                                        )
        myLogger.print3("The function for testing and visualisation of FMs was compiled.")
        
    def _getInputTensorsXToCnn(self):
        if not self.inputTensorsXToCnnInitialized :
            # Symbolic variables, which stand for the input to the CNN. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
            tensor5 = T.TensorType(dtype='float32', broadcastable=(False, False, False, False, False))
            self.inputTensorNormTrain = tensor5()  # Actually, for these 3, a single tensor5() could be used, as long as I reshape it separately for each afterwards. The actual value is loaded by the compiled functions.
            self.inputTensorNormVal = tensor5()  # myTensor.reshape(inputImageShapeValidation)
            self.inputTensorNormTest = tensor5()
            # For the multiple subsampled pathways.
            for subsPath_i in xrange(self.numSubsPaths) :
                self.listInputTensorPerSubsTrain.append(tensor5())  # Actually, for these 3, a single tensor5() could be used.
                self.listInputTensorPerSubsVal.append(tensor5())  # myTensor.reshape(inputImageSubsampledShapeValidation)
                self.listInputTensorPerSubsTest.append(tensor5())
            self.inputTensorsXToCnnInitialized = True
            
        return (self.inputTensorNormTrain, self.inputTensorNormVal, self.inputTensorNormTest,
                self.listInputTensorPerSubsTrain, self.listInputTensorPerSubsVal, self.listInputTensorPerSubsTest)
        
        
    def _getClassificationLayer(self):
        return SoftmaxLayer()
        
    def make_cnn_model( self,
                        myLogger,
                        cnnModelName,
                        #=== Model Parameters ===
                        numberOfOutputClasses,
                        numberOfImageChannelsPath1,
                        numberOfImageChannelsPath2,
                        
                        #=== Normal Pathway ===
                        nkerns,
                        kernelDimensions,
                        #=== Subsampled Pathway ===
                        # THESE NEXT TWO, ALONG WITH THE ONES FOR FC, COULD BE PUT IN ONE STRUCTURE WITH NORMAL, EG LIKE kerns = [ [kernsNorm], [kernsSub], [kernsFc]]
                        nkernsSubsampled, # Used to control if secondary pathways: [] if no secondary pathways. Now its the "factors"
                        kernelDimensionsSubsampled,
                        subsampleFactorsPerSubPath, # Controls how many pathways: [] if no secondary pathways. Else, List of lists. One sublist per secondary pathway. Each sublist has 3 ints, the rcz subsampling factors.
                        #=== FC Layers ===
                        fcLayersFMs,
                        kernelDimensionsFirstFcLayer,
                        softmaxTemperature,
                        
                        #=== Other Architectural params ===
                        activationFunctionToUseRelu0orPrelu1,
                        #---Residual Connections----
                        indicesOfLayersToConnectResidualsInOutput,
                        #--Lower Rank Layer Per Pathway---
                        indicesOfLowerRankLayersPerPathway,
                        ranksOfLowerRankLayersForEachPathway,
                        #---Pooling---
                        maxPoolingParamsStructure,
                        #--- Skip Connections --- #Deprecated, not used/supported
                        convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes,
                        
                        #===Size of Image Segments ===
                        imagePartDimensionsTraining ,
                        imagePartDimensionsValidation,
                        imagePartDimensionsTesting,
                        
                        #=== Batch Sizes ===
                        batch_size,
                        batch_size_validation,
                        batch_size_testing,
                        
                        #=== Others ===
                        # Dropout
                        dropoutRatesForAllPathways,  # list of sublists, one for each pathway. Each either empty or full with the dropout rates of all the layers in the path.
                        # Initialization
                        initializationTechniqueClassic0orDelvingInto1,
                        # Batch Normalization
                        applyBnToInputOfPathways,  # one Boolean flag per pathway type. Placeholder for the FC pathway.
                        rollingAverageForBatchNormalizationOverThatManyBatches,
                        
                        #=== various ====
                        borrowFlag,
                        dataTypeX='float32',
                        ):
        """
        maxPoolingParamsStructure: The padding of the function further below adds zeros. Zeros are not good, especially if I use PreLus. So I made mine, that pads by mirroring.
        Be careful that, without this, if I have ds=2, str=1 and ignoreBorder=False, it still reduces the dimension of the image by 1. That's why I need this. To keep the dimensions stable.
        It mirrors the last elements of each dimension as many times as it is given as arg.
        """
        self.cnnModelName = cnnModelName
        
        # ============= Model Parameters Passed as arguments ================
        self.numberOfOutputClasses = numberOfOutputClasses
        self.numberOfImageChannelsPath1 = numberOfImageChannelsPath1
        self.numberOfImageChannelsPath2 = numberOfImageChannelsPath2
        # === Architecture ===
        self.nkerns = nkerns  # Useless?
        self.nkernsSubsampled = nkernsSubsampled  # Useless?
        self.numSubsPaths = len(subsampleFactorsPerSubPath) # do I want this as attribute? Or function is ok?
        
        # fcLayersFMs???
        self.kernelDimensionsFirstFcLayer = kernelDimensionsFirstFcLayer
        
        # == Other Architectural Params ==
        self.indicesOfLayersToConnectResidualsInOutput = indicesOfLayersToConnectResidualsInOutput
        self.indicesOfLowerRankLayersPerPathway = indicesOfLowerRankLayersPerPathway
        # pooling?

        # == Batch Sizes ==
        self.batchSize = batch_size
        self.batchSizeValidation = batch_size_validation
        self.batchSizeTesting = batch_size_testing
        # == Others ==
        self.dropoutRatesForAllPathways = dropoutRatesForAllPathways
        self.initializationTechniqueClassic0orDelvingInto1 = initializationTechniqueClassic0orDelvingInto1
        # == various ==
        self.borrowFlag = borrowFlag
        self.dataTypeX = dataTypeX
        
        # ======== Calculated Attributes =========
        #This recField CNN should in future be calculated with all non-secondary pathways, ie normal+fc. Use another variable for pathway.recField.
        self.recFieldCnn = calculateReceptiveFieldDimensionsFromKernelsDimListPerLayerForFullyConvCnnWithStrides1(kernelDimensions)
        
        #==============================
        rng = np.random.RandomState(23455)
        
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        myLogger.print3("...Building the CNN model...")
        
        # Symbolic variables, which stand for the input. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
        (inputTensorNormTrain, inputTensorNormVal, inputTensorNormTest,
        listInputTensorPerSubsTrain, listInputTensorPerSubsVal, listInputTensorPerSubsTest) = self._getInputTensorsXToCnn()
        
        #=======================Make the NORMAL PATHWAY of the CNN=======================
        thisPathway = NormalPathway()
        self.pathways.append(thisPathway)
        thisPathwayType = thisPathway.pType()
        
        inputToPathwayTrain = inputTensorNormTrain
        inputToPathwayVal = inputTensorNormVal
        inputToPathwayTest = inputTensorNormTest
        inputToPathwayShapeTrain = [self.batchSize, numberOfImageChannelsPath1] + imagePartDimensionsTraining
        inputToPathwayShapeVal = [self.batchSizeValidation, numberOfImageChannelsPath1] + imagePartDimensionsValidation
        inputToPathwayShapeTest = [self.batchSizeTesting, numberOfImageChannelsPath1] + imagePartDimensionsTesting
        
        thisPathWayNKerns = nkerns
        thisPathWayKernelDimensions = kernelDimensions
        
        thisPathwayNumOfLayers = len(thisPathWayNKerns)
        thisPathwayUseBnPerLayer = [rollingAverageForBatchNormalizationOverThatManyBatches > 0] * thisPathwayNumOfLayers
        thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if rollingAverageForBatchNormalizationOverThatManyBatches > 0 else False  # For the 1st layer, ask specific flag.
        
        thisPathwayActivFuncPerLayer = [activationFunctionToUseRelu0orPrelu1] * thisPathwayNumOfLayers
        thisPathwayActivFuncPerLayer[0] = -1 if thisPathwayType != pt.FC else activationFunctionToUseRelu0orPrelu1  # To not apply activation on raw input. -1 is linear activation.
        
        thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                         inputToPathwayTrain,
                                                                         inputToPathwayVal,
                                                                         inputToPathwayTest,
                                                                         inputToPathwayShapeTrain,
                                                                         inputToPathwayShapeVal,
                                                                         inputToPathwayShapeTest,
                                                                         
                                                                         thisPathWayNKerns,
                                                                         thisPathWayKernelDimensions,
                                                                         
                                                                         initializationTechniqueClassic0orDelvingInto1,
                                                                         thisPathwayUseBnPerLayer,
                                                                         rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                         thisPathwayActivFuncPerLayer,
                                                                         dropoutRatesForAllPathways[thisPathwayType],
                                                                         
                                                                         maxPoolingParamsStructure[thisPathwayType],
                                                                         
                                                                         indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                         ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                         indicesOfLayersToConnectResidualsInOutput[thisPathwayType]
                                                                         )
        # Skip connections to end of pathway.
        thisPathway.makeMultiscaleConnectionsForLayerType(convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes[thisPathwayType])
        
        [dimsOfOutputFrom1stPathwayTrain, dimsOfOutputFrom1stPathwayVal, dimsOfOutputFrom1stPathwayTest] = thisPathway.getShapeOfOutput()
        
        #=======================Make the SUBSAMPLED PATHWAYs of the CNN=============================
        for subPath_i in xrange(self.numSubsPaths) :
            thisPathway = SubsampledPathway(subsampleFactorsPerSubPath[subPath_i])
            self.pathways.append(thisPathway) # There will be at least an entry as a secondary pathway. But it won't have any layers if it was not actually used.
            thisPathwayType = thisPathway.pType()
            
            inputToPathwayTrain = listInputTensorPerSubsTrain[subPath_i]
            inputToPathwayVal = listInputTensorPerSubsVal[subPath_i]
            inputToPathwayTest = listInputTensorPerSubsTest[subPath_i]
            
            thisPathWayNKerns = nkernsSubsampled[subPath_i]
            thisPathWayKernelDimensions = kernelDimensionsSubsampled
            
            thisPathwayNumOfLayers = len(thisPathWayNKerns)
            thisPathwayUseBnPerLayer = [rollingAverageForBatchNormalizationOverThatManyBatches > 0] * thisPathwayNumOfLayers
            thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if rollingAverageForBatchNormalizationOverThatManyBatches > 0 else False  # For the 1st layer, ask specific flag.
            
            thisPathwayActivFuncPerLayer = [activationFunctionToUseRelu0orPrelu1] * thisPathwayNumOfLayers
            thisPathwayActivFuncPerLayer[0] = -1 if thisPathwayType != pt.FC else activationFunctionToUseRelu0orPrelu1  # To not apply activation on raw input. -1 is linear activation.
            
            inputToPathwayShapeTrain = [self.batchSize, numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayTrain);
            inputToPathwayShapeVal = [self.batchSizeValidation, numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayVal)
            inputToPathwayShapeTest = [self.batchSizeTesting, numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayTest)
            
            thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                     inputToPathwayTrain,
                                                                     inputToPathwayVal,
                                                                     inputToPathwayTest,
                                                                     inputToPathwayShapeTrain,
                                                                     inputToPathwayShapeVal,
                                                                     inputToPathwayShapeTest,
                                                                     thisPathWayNKerns,
                                                                     thisPathWayKernelDimensions,
                                                                     
                                                                     initializationTechniqueClassic0orDelvingInto1,
                                                                     thisPathwayUseBnPerLayer,
                                                                     rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                     thisPathwayActivFuncPerLayer,
                                                                     dropoutRatesForAllPathways[thisPathwayType],
                                                                     
                                                                     maxPoolingParamsStructure[thisPathwayType],
                                                                     
                                                                     indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                     ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                     indicesOfLayersToConnectResidualsInOutput[thisPathwayType]
                                                                     )
            # Skip connections to end of pathway.
            thisPathway.makeMultiscaleConnectionsForLayerType(convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes[thisPathwayType])
            
            # this creates essentially the "upsampling layer"
            thisPathway.upsampleOutputToNormalRes(upsamplingScheme="repeat",
                                                  shapeToMatchInRczTrain=dimsOfOutputFrom1stPathwayTrain,
                                                  shapeToMatchInRczVal=dimsOfOutputFrom1stPathwayVal,
                                                  shapeToMatchInRczTest=dimsOfOutputFrom1stPathwayTest)
            
            
        #====================================CONCATENATE the output of the 2 cnn-pathways=============================
        inputToFirstFcLayerTrain = None; inputToFirstFcLayerVal = None; inputToFirstFcLayerTest = None; numberOfFmsOfInputToFirstFcLayer = 0
        for path_i in xrange(len(self.pathways)) :
            [outputNormResOfPathTrain, outputNormResOfPathVal, outputNormResOfPathTest] = self.pathways[path_i].getOutputAtNormalRes()
            [dimsOfOutputNormResOfPathTrain, dimsOfOutputNormResOfPathVal, dimsOfOutputNormResOfPathTest] = self.pathways[path_i].getShapeOfOutputAtNormalRes()
            
            inputToFirstFcLayerTrain =  T.concatenate([inputToFirstFcLayerTrain, outputNormResOfPathTrain], axis=1) if path_i != 0 else outputNormResOfPathTrain
            inputToFirstFcLayerVal = T.concatenate([inputToFirstFcLayerVal, outputNormResOfPathVal], axis=1) if path_i != 0 else outputNormResOfPathVal
            inputToFirstFcLayerTest = T.concatenate([inputToFirstFcLayerTest, outputNormResOfPathTest], axis=1) if path_i != 0 else outputNormResOfPathTest
            numberOfFmsOfInputToFirstFcLayer += dimsOfOutputNormResOfPathTrain[1]
            
        #======================= Make the Fully Connected Layers =======================
        thisPathway = FcPathway()
        self.pathways.append(thisPathway)
        thisPathwayType = thisPathway.pType()
        
        # This is the shape of the kernel in the first FC layer.
        # NOTE: If there is no hidden FC layer, this kernel is used in the Classification layer then.
        # Originally it was 1x1x1 only. The pathways themselves where taking care of the receptive field.
        # However I can now define it larger (eg 3x3x3), in case it helps combining the multiresolution features better/smoother.
        # The convolution is seamless, ie same shape output/input, by mirror padding the input.
        firstFcLayerAfterConcatenationKernelShape = self.kernelDimensionsFirstFcLayer
        voxelsToPadPerDim = [ kernelDim - 1 for kernelDim in firstFcLayerAfterConcatenationKernelShape ]
        myLogger.print3("DEBUG: Shape of the kernel of the first FC layer is : " + str(firstFcLayerAfterConcatenationKernelShape))
        myLogger.print3("DEBUG: Input to the FC Pathway will be padded by that many voxels per dimension: " + str(voxelsToPadPerDim))
        inputToPathwayTrain = padImageWithMirroring(inputToFirstFcLayerTrain, voxelsToPadPerDim)
        inputToPathwayVal = padImageWithMirroring(inputToFirstFcLayerVal, voxelsToPadPerDim)
        inputToPathwayTest = padImageWithMirroring(inputToFirstFcLayerTest, voxelsToPadPerDim)
        inputToPathwayShapeTrain = [self.batchSize, numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayTrain[2:5]
        inputToPathwayShapeVal = [self.batchSizeValidation, numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayVal[2:5]
        inputToPathwayShapeTest = [self.batchSizeTesting, numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayTest[2:5]
        for rcz_i in xrange(3) : 
            inputToPathwayShapeTrain[2+rcz_i] += voxelsToPadPerDim[rcz_i]
            inputToPathwayShapeVal[2+rcz_i] += voxelsToPadPerDim[rcz_i]
            inputToPathwayShapeTest[2+rcz_i] += voxelsToPadPerDim[rcz_i]
        
        thisPathWayNKerns = fcLayersFMs + [self.numberOfOutputClasses]
        thisPathWayKernelDimensions = [firstFcLayerAfterConcatenationKernelShape] + [[1, 1, 1]] * (len(thisPathWayNKerns) - 1)
        
        thisPathwayNumOfLayers = len(thisPathWayNKerns)
        thisPathwayUseBnPerLayer = [rollingAverageForBatchNormalizationOverThatManyBatches > 0] * thisPathwayNumOfLayers
        thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if rollingAverageForBatchNormalizationOverThatManyBatches > 0 else False  # For the 1st layer, ask specific flag.
        
        thisPathwayActivFuncPerLayer = [activationFunctionToUseRelu0orPrelu1] * thisPathwayNumOfLayers
        thisPathwayActivFuncPerLayer[0] = -1 if thisPathwayType != pt.FC else activationFunctionToUseRelu0orPrelu1  # To not apply activation on raw input. -1 is linear activation.
        
        thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                         inputToPathwayTrain,
                                                                         inputToPathwayVal,
                                                                         inputToPathwayTest,
                                                                         inputToPathwayShapeTrain,
                                                                         inputToPathwayShapeVal,
                                                                         inputToPathwayShapeTest,
                                                                         
                                                                         thisPathWayNKerns,
                                                                         thisPathWayKernelDimensions,
                                                                         
                                                                         initializationTechniqueClassic0orDelvingInto1,
                                                                         thisPathwayUseBnPerLayer,
                                                                         rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                         thisPathwayActivFuncPerLayer,
                                                                         dropoutRatesForAllPathways[thisPathwayType],
                                                                         
                                                                         maxPoolingParamsStructure[thisPathwayType],
                                                                         
                                                                         indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                         ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                         indicesOfLayersToConnectResidualsInOutput[thisPathwayType],
                                                                         )
        
        # =========== Make the final Target Layer (softmax, regression, whatever) ==========
        myLogger.print3("Adding the final Softmax Target layer...")
        
        self.finalTargetLayer = self._getClassificationLayer()
        self.finalTargetLayer.makeLayer(rng, self.getFcPathway().getLayer(-1), softmaxTemperature)
        
        myLogger.print3("Finished building the CNN's model.")
        
        