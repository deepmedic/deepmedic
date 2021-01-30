# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np

from deepmedic.logging.utils import strFl4fNA, strFl5fNA, strListFl4fNA, strListFl5fNA, getMeanOfListExclNA


class AccuracyMonitorForEpSegm(object):
    
    NA_PATTERN = "N/A"  # not applicable. Eg for accuracy when class not present.
    
    def __init__(self,
                 log,
                 training0orValidation1,
                 epoch,  # number Of epochs trained prior to this
                 numberOfClasses,
                 numberOfSubepochsPerEpoch,
                 tensorboard_logger=None):
        """
        tensorboard_logger: None if no tb logging, otherwise instance of deepmedic.logging.tensorboard_logger.TensorboardLogger
        """
        self.tensorboard_logger = tensorboard_logger

        self.log = log
        self.training0orValidation1 = training0orValidation1
        self.trainOrValString = "TRAINING" if self.training0orValidation1 == 0 else "VALIDATION"
        self.epoch = epoch
        self.numberOfClasses = numberOfClasses
        self.numberOfSubepochsPerEpoch = numberOfSubepochsPerEpoch
        
        self.numberOfSubepochsForWhichUpdated = 0
        
        # === Collecting accuracy metrics over the whole epoch: ===
        # --- mean Empirical Accuracy: number of Corr Classified voxels and all voxels, wrt to all the classes, in multiclass problems.---
        if self.training0orValidation1 == 0 :
            self.meanCostOfEachSubep = [] # mean value of the cost function (training only)
        self.correctlyPredVoxelsInEachSubep = []
        self.numberOfAllSamplesOfEachSubep = []
        self.meanEmpiricalAccuracyOfEachSubep = [] # mean (multiclass) accuracy. correctlyPredVoxelsInEachSubep / numberOfAllSamplesOfEachSubep # NOTE: Cannot contain N0T-APPLICABLE
        
        # --- Per Class Accuracies and Real/True Pos/Neg (in a One-Vs-All fashion)
        # These do not have the class-0 background flipped to foreground!
        
        self.listPerSubepPerClassRpRnTpTn = [] # subepochs X Classes X 4. 4 = RP, RN, TP, TN
        self.listPerSubepPerClassMeanAccSensSpecDsc = [] # NOTE: May contain N0T-APPLICABLE=self.NA_PATTERN elements, eg when class not present!
        
        # For the merged-foreground, which is reported instead of class-0:
        self.listPerSubepForegrRpRnTpTn = []
        self.listPerSubepForegrMeanAccSensSpecDsc = [] # NOTE: May contain N0T-APPLICABLE=self.NA_PATTERN elements, eg when class not present!
        
    # ==== API ====
    def get_avg_accuracy_ep(self): # Multiclass mean accuracy. As in: Number-of-Voxels-Correctly-Classified / All-voxels
        return np.mean(self.meanEmpiricalAccuracyOfEachSubep)
    
    
    # Generic. Does not flip the class-0 background class.
    def update_metrics_after_subep(self, meanCostOfSubepoch, perClassRpRnTpTnInSubep):
        # perClassRpRnTpTnInSubep # Class X 4. The Real Pos, Real Neg, True Pos (pred), True Neg (pred).
        #---------- This first part takes care of the overall, multi-class mean accuracy: meanAccuracy = Num-Voxels-Predicted-Correct-Class / All-Voxels. 
        if self.training0orValidation1 == 0 :
            self.meanCostOfEachSubep.append(meanCostOfSubepoch)
            
        correctlyPredVoxelsInSubep = 0; # Predicted with the correct class.
        for class_i in range(self.numberOfClasses) :
            correctlyPredVoxelsInSubep += perClassRpRnTpTnInSubep[class_i][2] # Add up the true positives per class (which are one vs all).
        self.correctlyPredVoxelsInEachSubep.append(correctlyPredVoxelsInSubep)
        numberOfAllSamples = perClassRpRnTpTnInSubep[0,0] + perClassRpRnTpTnInSubep[0,1] #RealPos + RealNeg wrt any class (eg backgr)
        self.numberOfAllSamplesOfEachSubep.append(numberOfAllSamples)
        meanAccuracyOfSubepoch = self.NA_PATTERN if numberOfAllSamples == 0 else correctlyPredVoxelsInSubep*1.0/numberOfAllSamples
        self.meanEmpiricalAccuracyOfEachSubep.append(meanAccuracyOfSubepoch)
        
        #----------- Calculate accuracy over subepoch for each class_i, in a One-Vs-All fashion ------
        
        self.listPerSubepPerClassRpRnTpTn.append(perClassRpRnTpTnInSubep)
        
        listWithPerClassMeanAccSensSpecDscInSubep = [] # Classes x 4 . 4=Mean Acc, sens, spec, dsc.
        for class_i in range(self.numberOfClasses) :
            numOfRealPosInSubep = perClassRpRnTpTnInSubep[class_i, 0]
            numOfRealNegInSubep = perClassRpRnTpTnInSubep[class_i, 1]
            numOfTruePosInSubep = perClassRpRnTpTnInSubep[class_i, 2]
            numOfTrueNegInSubep = perClassRpRnTpTnInSubep[class_i, 3]
            
            numOfFalsePosInSubep = numOfRealNegInSubep - numOfTrueNegInSubep
            
            meanAccuracyClassVsAllOfSubep = (numOfTruePosInSubep+numOfTrueNegInSubep) / float(numOfRealPosInSubep+numOfRealNegInSubep)
            meanAccuracyOnPosOfSubep = self.NA_PATTERN if numOfRealPosInSubep == 0 else numOfTruePosInSubep*1.0/numOfRealPosInSubep
            meanPrecOfSubep = self.NA_PATTERN if (numOfTruePosInSubep+numOfFalsePosInSubep) == 0 else numOfTruePosInSubep*1.0/(numOfTruePosInSubep+numOfFalsePosInSubep)
            meanAccuracyOnNegOfSubep = self.NA_PATTERN if numOfRealNegInSubep == 0 else numOfTrueNegInSubep*1.0/numOfRealNegInSubep
            # Compute dice for the subepoch training/validation batches!
            numOfPredPosInSubep = numOfRealNegInSubep - numOfTrueNegInSubep + numOfTruePosInSubep
            meanDiceOfSubep = self.NA_PATTERN if numOfRealPosInSubep == 0 else (2.0*numOfTruePosInSubep)/(numOfPredPosInSubep + numOfRealPosInSubep)
            
            listWithPerClassMeanAccSensSpecDscInSubep.append( [meanAccuracyClassVsAllOfSubep, meanAccuracyOnPosOfSubep, meanPrecOfSubep, meanAccuracyOnNegOfSubep, meanDiceOfSubep] )
        self.listPerSubepPerClassMeanAccSensSpecDsc.append(listWithPerClassMeanAccSensSpecDscInSubep)
        
        # ===============UPDATE THE MERGED FOREGROUND CLASS. (used in multi-class problems instead of class-0 background).================
        # RealNeg/Pos of background is the RealPos/Neg of the foreground. TrueNeg/Pos of background is the TruePos/Neg of the foreground.
        foregrTp = perClassRpRnTpTnInSubep[0][3]; foregrTn = perClassRpRnTpTnInSubep[0][2]; foregrRp = perClassRpRnTpTnInSubep[0][1]; foregrRn = perClassRpRnTpTnInSubep[0][0]
        foregrFp = foregrRn - foregrTn
        self.listPerSubepForegrRpRnTpTn.append([ foregrRp, foregrRn, foregrTp, foregrTn ] )
        
        foregrMeanAccOfSubep = (foregrTp+foregrTn) / float(foregrRp+foregrRn)
        foregrMeanAccOnPosOfSubep = self.NA_PATTERN if foregrRp == 0 else foregrTp*1.0/foregrRp
        foregrMeanPrecOfSubep = self.NA_PATTERN if (foregrTp+foregrFp) == 0 else foregrTp*1.0 / (foregrTp+foregrFp)
        foregrMeanAccOnNegOfSubep = self.NA_PATTERN if foregrRn == 0 else foregrTn*1.0/foregrRn
        foregrPredPosInSubep = foregrRn - foregrTn + foregrTp
        foregrMeanDiceOfSubep = self.NA_PATTERN if foregrRp == 0 else (2.0*foregrTp)/(foregrPredPosInSubep + foregrRp)
        self.listPerSubepForegrMeanAccSensSpecDsc.append( [ foregrMeanAccOfSubep, foregrMeanAccOnPosOfSubep, foregrMeanPrecOfSubep, foregrMeanAccOnNegOfSubep, foregrMeanDiceOfSubep] )
        
        # Done!
        self.numberOfSubepochsForWhichUpdated += 1

    def log_acc_subep_to_txt(self):
        currSubep = self.numberOfSubepochsForWhichUpdated - 1
        logStr = self.trainOrValString + ": Epoch #" + str(self.epoch) + ", Subepoch #" + str(currSubep)
        self.log.print3("")
        self.log.print3("+++++++++++++++++++++++ Reporting Accuracy over whole subepoch +++++++++++++++++++++++")
        self.log.print3(logStr + ", Overall:\t mean accuracy:   \t" + strFl4fNA(self.meanEmpiricalAccuracyOfEachSubep[currSubep], self.NA_PATTERN) + \
                        "\t=> Correctly-Classified-Voxels/All-Predicted-Voxels = " + str(self.correctlyPredVoxelsInEachSubep[currSubep]) + "/" + str(self.numberOfAllSamplesOfEachSubep[currSubep]) )
        if self.training0orValidation1 == 0:  # During training, also report the mean value of the Cost Function:
            self.log.print3(logStr + ", Overall:\t mean cost:      \t" + strFl5fNA(self.meanCostOfEachSubep[currSubep], self.NA_PATTERN))
            
        # Report accuracy over subepoch for each class_i:
        for class_i in range(self.numberOfClasses):
            class_string = "Class-"+str(class_i)
            extraDescription = "[Whole Foreground (Pos) Vs Background (Neg)]" if class_i == 0 else "[This Class (Pos) Vs All Others (Neg)]"
            
            self.log.print3( "+++++++++++++++ Reporting Accuracy over whole subepoch for " + class_string + " ++++++++ " + extraDescription + " ++++++++++++++++" )
            
            [meanAccClassOfSubep,
            meanAccOnPosOfSubep,
            meanPrecOfSubep,
            meanAccOnNegOfSubep,
            meanDiceOfSubep ] = self.listPerSubepPerClassMeanAccSensSpecDsc[currSubep][class_i] if class_i != 0 else \
                                    self.listPerSubepForegrMeanAccSensSpecDsc[currSubep] # If class-0, report foreground.
            [numOfRpInSubep,
            numOfRnInSubep,
            numOfTpInSubep,
            numOfTnInSubep] = self.listPerSubepPerClassRpRnTpTn[currSubep][class_i] if class_i != 0 else \
                                    self.listPerSubepForegrRpRnTpTn[currSubep] # If class-0, report foreground.
            
            numOfFpInSubep = numOfRnInSubep - numOfTnInSubep
            
            logStrClass = logStr + ", " + class_string + ":"
            self.log.print3(logStrClass+"\t mean accuracy:   \t" +
                            strFl4fNA(meanAccClassOfSubep, self.NA_PATTERN) +
                            "\t=> (TruePos+TrueNeg)/All-Predicted-Voxels = " +
                            str(numOfTpInSubep+numOfTnInSubep) + "/" +
                            str(numOfRpInSubep+numOfRnInSubep))
            self.log.print3(logStrClass+"\t mean sensitivity:\t" + strFl4fNA(meanAccOnPosOfSubep, self.NA_PATTERN)+"\t=> TruePos/RealPos = "+str(numOfTpInSubep)+"/"+str(numOfRpInSubep))
            self.log.print3(logStrClass+"\t mean precision:\t" + strFl4fNA(meanPrecOfSubep, self.NA_PATTERN)+"\t=> TruePos/(TruePos+FalsePos) = "+str(numOfTpInSubep)+"/"+str(numOfTpInSubep+numOfFpInSubep))
            self.log.print3(logStrClass+"\t mean specificity:\t" + strFl4fNA(meanAccOnNegOfSubep, self.NA_PATTERN)+"\t=> TrueNeg/RealNeg = "+str(numOfTnInSubep)+"/"+str(numOfRnInSubep))
            self.log.print3(logStrClass+"\t mean Dice:       \t" + strFl4fNA(meanDiceOfSubep, self.NA_PATTERN))

    def log_to_tensorboard(self, metrics_dict, class_string, step_num):
        if self.tensorboard_logger is not None:
            for metric, value in metrics_dict.items():
                if value == self.NA_PATTERN:
                    value = np.nan
                self.tensorboard_logger.add_summary(value, metric + '/' + class_string, step_num)

    def log_acc_subep_to_tensorboard(self):
        currSubep = self.numberOfSubepochsForWhichUpdated - 1

        self.log.print3('=============== LOGGING TO TENSORBOARD ===============')
        # check if user included tensorboard logging in the config
        if self.tensorboard_logger is None:
            self.log.print3('Tensorboard logging not activated. Skipping...')
            self.log.print3('======================================================')
            return
        
        self.log.print3('Logging ' + self.trainOrValString + ' metrics')
        self.log.print3('Epoch: ' + str(self.epoch) +
                        ' | Subepoch ' + str(currSubep))
        step_num = currSubep + (self.epoch * (self.numberOfSubepochsPerEpoch))
        self.log.print3('Step number (index of subepoch since start): ' + str(step_num))

        # During training, also report the mean value of the Cost Function:
        if self.training0orValidation1 == 0 and self.tensorboard_logger is not None:
            self.log.print3('--- Logging average metrics for all classes ---')

            # create metrics dictionary
            metrics_dict = {'samples: accuracy': self.meanEmpiricalAccuracyOfEachSubep[currSubep],
                            'samples: cost': self.meanCostOfEachSubep[currSubep]}
            class_string = 'Class-all'
            self.log_to_tensorboard(metrics_dict, class_string, step_num)

            self.log.print3('Logged metrics: ' + str(list(metrics_dict.keys())))

        # Report accuracy over subepoch for each class_i:
        self.log.print3('--- Logging per class metrics ---')
        for class_i in range(self.numberOfClasses):
            class_string = "Class-" + str(class_i)

            [meanAccClassOfSubep,
             meanAccOnPosOfSubep,
             meanPrecOfSubep,
             meanAccOnNegOfSubep,
             meanDiceOfSubep] = self.listPerSubepPerClassMeanAccSensSpecDsc[currSubep][class_i] if class_i != 0 else \
                self.listPerSubepForegrMeanAccSensSpecDsc[currSubep]  # If class-0, report foreground.

            # create metrics dictionary
            metrics_dict = {'samples: accuracy': meanAccClassOfSubep,
                            'samples: sensitivity': meanAccOnPosOfSubep,
                            'samples: precision': meanPrecOfSubep,
                            'samples: specificity': meanAccOnNegOfSubep,
                            'samples: Dice': meanDiceOfSubep}
            self.log_to_tensorboard(metrics_dict, class_string, step_num)

        self.log.print3('Logged metrics: ' + str(list(metrics_dict.keys())))
        self.log.print3('======================================================')

                
    def report_metrics_whole_vols(self, mean_metrics):
        # mean_metrics: 
        # metrics_dict_list: list that holds one element per class, [ elem-class0, elem-class1, .... ]
        #                    Each element is a dictionary of metrics for the class.
        #                    E.g. elem-class0 = {'dice1': value, 'dice2': value, dice3': value}
        self.log.print3('=============== LOGGING TO TENSORBOARD ===============')
        if self.tensorboard_logger is None:
            self.log.print3('Tensorboard logging not activated. Skipping...')
            self.log.print3('======================================================')
            return
        
        self.log.print3('Logging validation metrics from segmentation of whole validation volumes.')
        self.log.print3('Epoch: ' + str(self.epoch))
        step_num = self.numberOfSubepochsPerEpoch - 1 + (self.epoch * self.numberOfSubepochsPerEpoch)
        self.log.print3('Step number (index of subepoch since start): ' + str(step_num))

        # Report mean metrics for each class_i:
        for class_i in range(self.numberOfClasses):
            # the keys for the below are defined in testing.py routine.
            metrics_dict = {'whole scans: Dice1 (Prediction VS Truth)': mean_metrics['dice1'][class_i],
                            'whole scans: Dice2 (Prediction within ROI mask VS Truth)': mean_metrics['dice2'][class_i],
                            'whole scans: Dice3 (Prediction VS Truth, both within ROI mask)': mean_metrics['dice3'][class_i]
                            }
            class_string = 'Class-' + str(class_i)
            self.log_to_tensorboard(metrics_dict, class_string, step_num)

        self.log.print3('Logged metrics: ' + str(list(metrics_dict.keys())))
        self.log.print3('======================================================')
        
        
    def report_metrics_samples_ep(self) :
        logStr = self.trainOrValString + ": Epoch #" + str(self.epoch)
        
        # Report the multi-class accuracy first.
        self.log.print3( "( >>>>>>>>>>>>>>>>>>>> Reporting Accuracy over whole epoch <<<<<<<<<<<<<<<<<<<<<<<<<<<<" )
        meanEmpiricalAccOfEp = getMeanOfListExclNA(self.meanEmpiricalAccuracyOfEachSubep, self.NA_PATTERN)
        self.log.print3( logStr + ", Overall:\t mean accuracy of epoch:\t" + strFl4fNA(meanEmpiricalAccOfEp, self.NA_PATTERN)+"\t=> Correctly-Classified-Voxels/All-Predicted-Voxels")
        if self.training0orValidation1 == 0 : # During training, also report the mean value of the Cost Function:
            meanCostOfEp = getMeanOfListExclNA(self.meanCostOfEachSubep, self.NA_PATTERN)
            self.log.print3( logStr + ", Overall:\t mean cost of epoch:    \t" + strFl5fNA(meanCostOfEp, self.NA_PATTERN) )
            
        self.log.print3( logStr + ", Overall:\t mean accuracy of each subepoch:\t"+ strListFl4fNA(self.meanEmpiricalAccuracyOfEachSubep, self.NA_PATTERN) )
        if self.training0orValidation1 == 0 :
            self.log.print3( logStr + ", Overall:\t mean cost of each subepoch:    \t" + strListFl5fNA(self.meanCostOfEachSubep, self.NA_PATTERN) )
            
        # Report for each class.
        for class_i in range(self.numberOfClasses) :
            class_string = "Class-"+str(class_i)
            extraDescription = "[Whole Foreground (Pos) Vs Background (Neg)]" if class_i == 0 else "[This Class (Pos) Vs All Others (Neg)]"
            
            self.log.print3( ">>>>>>>>>>>> Reporting Accuracy over whole epoch for " + class_string + " >>>>>>>>> " + extraDescription + " <<<<<<<<<<<<<" )
            
            if class_i != 0 :
                meanAccPerSubep = [ self.listPerSubepPerClassMeanAccSensSpecDsc[subep_i][class_i][0] for subep_i in range(len(self.listPerSubepPerClassMeanAccSensSpecDsc)) ]
                meanSensPerSubep = [ self.listPerSubepPerClassMeanAccSensSpecDsc[subep_i][class_i][1] for subep_i in range(len(self.listPerSubepPerClassMeanAccSensSpecDsc)) ]
                meanPrecPerSubep = [ self.listPerSubepPerClassMeanAccSensSpecDsc[subep_i][class_i][2] for subep_i in range(len(self.listPerSubepPerClassMeanAccSensSpecDsc)) ]
                meanSpecPerSubep = [ self.listPerSubepPerClassMeanAccSensSpecDsc[subep_i][class_i][3] for subep_i in range(len(self.listPerSubepPerClassMeanAccSensSpecDsc)) ]
                meanDscPerSubep = [ self.listPerSubepPerClassMeanAccSensSpecDsc[subep_i][class_i][4] for subep_i in range(len(self.listPerSubepPerClassMeanAccSensSpecDsc)) ]
            else : # Foreground Vs Background
                meanAccPerSubep = [ self.listPerSubepForegrMeanAccSensSpecDsc[subep_i][0] for subep_i in range(len(self.listPerSubepForegrMeanAccSensSpecDsc)) ]
                meanSensPerSubep = [ self.listPerSubepForegrMeanAccSensSpecDsc[subep_i][1] for subep_i in range(len(self.listPerSubepForegrMeanAccSensSpecDsc)) ]
                meanPrecPerSubep = [ self.listPerSubepForegrMeanAccSensSpecDsc[subep_i][2] for subep_i in range(len(self.listPerSubepForegrMeanAccSensSpecDsc)) ]
                meanSpecPerSubep = [ self.listPerSubepForegrMeanAccSensSpecDsc[subep_i][3] for subep_i in range(len(self.listPerSubepForegrMeanAccSensSpecDsc)) ]
                meanDscPerSubep = [ self.listPerSubepForegrMeanAccSensSpecDsc[subep_i][4] for subep_i in range(len(self.listPerSubepForegrMeanAccSensSpecDsc)) ]
                
            meanAccOfEp = getMeanOfListExclNA(meanAccPerSubep, self.NA_PATTERN)
            meanSensOfEp = getMeanOfListExclNA(meanSensPerSubep, self.NA_PATTERN)
            meanPrecOfEp = getMeanOfListExclNA(meanPrecPerSubep, self.NA_PATTERN)
            meanSpecOfEp = getMeanOfListExclNA(meanSpecPerSubep, self.NA_PATTERN)
            meanDscOfEp = getMeanOfListExclNA(meanDscPerSubep, self.NA_PATTERN)
            
            logStrClass = logStr + ", " + class_string + ":"
            self.log.print3(logStrClass + "\t mean accuracy of epoch:\t"+ strFl4fNA(meanAccOfEp, self.NA_PATTERN) +"\t=> (TruePos+TrueNeg)/All-Predicted-Voxels")
            self.log.print3(logStrClass + "\t mean sensitivity of epoch:\t"+ strFl4fNA(meanSensOfEp, self.NA_PATTERN) +"\t=> TruePos/RealPos")
            self.log.print3(logStrClass + "\t mean precision of epoch:\t"+ strFl4fNA(meanPrecOfEp, self.NA_PATTERN) +"\t=> TruePos/(TruePos+FalsePos)")
            self.log.print3(logStrClass + "\t mean specificity of epoch:\t"+ strFl4fNA(meanSpecOfEp, self.NA_PATTERN) +"\t=> TrueNeg/RealNeg")
            self.log.print3(logStrClass + "\t mean Dice of epoch:    \t"+ strFl4fNA(meanDscOfEp, self.NA_PATTERN) )
            
            #Visualised in my scripts:
            self.log.print3(logStrClass + "\t mean accuracy of each subepoch:\t"+ strListFl4fNA(meanAccPerSubep, self.NA_PATTERN) )
            self.log.print3(logStrClass + "\t mean sensitivity of each subepoch:\t" + strListFl4fNA(meanSensPerSubep, self.NA_PATTERN) )
            self.log.print3(logStrClass + "\t mean precision of each subepoch:\t" + strListFl4fNA(meanPrecPerSubep, self.NA_PATTERN) )
            self.log.print3(logStrClass + "\t mean specificity of each subepoch:\t" + strListFl4fNA(meanSpecPerSubep, self.NA_PATTERN) )
            self.log.print3(logStrClass + "\t mean Dice of each subepoch:    \t" + strListFl4fNA(meanDscPerSubep, self.NA_PATTERN) )
            
        self.log.print3( ">>>>>>>>>>>>>>>>>>>>>>>>> End Of Accuracy Report at the end of Epoch <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" )
        self.log.print3( ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" )
