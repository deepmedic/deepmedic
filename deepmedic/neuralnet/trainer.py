# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import tensorflow as tf

import deepmedic.neuralnet.optimizers as optimizers_dm
import deepmedic.neuralnet.cost_functions as cfs

# Calls:
# __init__
# setup_costs <--- can be multiple costs. But combined into one total_cost, called from get_param_updates.
# create_optimizer <---can be multiple optimizers for different parts of the net. And all combined/called in get_param_updates.
# get_param_updates_wrt_total_cost <--- Because of returning updates, 1 Trainer = 1 backprop(Sgd iter). Can be many costs, all summed, 1 backprop.
# run_updates_end_of_ep (in routine.training)
# Note: If I d like to change the design, making 1 Trainer per Cost, I would need to compute fuse all costs externally (eg cnn3d),...
# ... and do backprop for grads from the fused cost. And then give grads to separate Optimizers if desired. 

class Trainer(object):
    # I SHOULD NOT MAKE ANOTHER TRAINER-MANAGER. The deepmedicTrain() function is essentially the coordinator, intermigled with parser.
    # This acts as a learning-rate scheduler and as a manager of the optimizer.
    # In case of multiple optimizers, either this must become a manager of multiple optimizers+schedules, ...
    # ... or I ll need one trainer per optimizer and a new trainer-manager. 
    def __init__(self,
                 log,
                 indicesOfLayersPerPathwayTypeToFreeze,
                 losses_and_weights,
                 L1_reg_weight,
                 L2_reg_weight,
                 # Cost schedules
                 reweight_classes_in_cost,
                 network_to_train):
        
        log.print3("Building Trainer.")
        
        self._net = network_to_train # Used to grab trainable parameter, and most of all, to formulate the costs in set_costs.
        self._indicesOfLayersPerPathwayTypeToFreeze = indicesOfLayersPerPathwayTypeToFreeze # Layers to train (eg for pretrained models.
        # Regularisation
        self._L1_reg_weight = L1_reg_weight
        self._L2_reg_weight = L2_reg_weight
        # Costs
        self._losses_and_weights = losses_and_weights  # "L", "D" or "J"
        self._total_cost = None # This is set-up by calling self.setup_costs(...)
        # Params for costs
        self._reweight_classes_in_cost = reweight_classes_in_cost
        
        
        ################# OPTIMIZER AND SCHEDULES ###############
        
        ######### training state ##########
        self._num_epochs_trained_tfv = tf.Variable(0, dtype="int64", trainable=False, name="num_epochs_trained") # int32 tf.vars cannot be (explicitly) loaded to gpu.
        # State Ops
        # TODO: All the ops should be constructed and kept into one place (eg dict), and then can be easily called from outside...
        # ... INSTEAD of called a public function such as set_lr or change_lr.
        self._tf_plchld_float32 = tf.compat.v1.placeholder( dtype="float32", name="tf_plchld_float32") # convenience feed for tf.assign
        self._tf_plchld_int32 = tf.compat.v1.placeholder( dtype="int32", name="tf_plchld_int32") # convenience feed for tf.assign
        self._op_increase_num_epochs_trained = tf.compat.v1.assign( self._num_epochs_trained_tfv, self._num_epochs_trained_tfv + 1)
        
        
        ########### Optimizer ###########
        # Optimizers
        self._optimizer = None # Trainer could be coordinating multiple optimizers, over multiple costs?
        
        ######## LR schedule specific ######
        # These are separated from the above, "Trainer" section, for future further modularization...
        # ... in case I decice it is wiser to have one trainer, coordinating multiple optimizers, with one LR schedule each.
        # I could also have one trainer per optimizer, and one trainer-manager.
        # In that case, the above "Trainer" section could be moved to the "trainer-manager" class.
        self._lr_sched_params = None # To control if schedule-specific API is available (eg auto)
        self._curr_lr = None # Tensor (or tfv if auto). Used for printing.
        self._curr_mom = None # Tensor. Used for printing.
        # State
        self._init_lr_tfv = None  # used by exponential schedule
        # Mom is only for SGD/RmsProp
        self._init_mom_tfv = None  # used by exponential schedule
        
        # ====== [Auto] - learning rate schedule ====
        # These should only be defined if auto-schedule is chosen.
        # State
        self._learning_rate_tfv = None # tf.var
        self._momentum_tfv = None
        self._top_mean_val_acc_tfv = None
        self._epoch_with_top_mean_val_acc_tvf = None
        self._last_epoch_lr_got_lowered_tvf = None
        self._op_assign_new_lr = None
        self._op_assign_new_mom = None
        self._op_assign_top_mean_val_acc_tfv = None
        self._op_assign_epoch_with_top_mean_val_acc_tvf = None
        self._op_assign_last_epoch_lr_lowered = None
        
        
    ############## All the logic wrt cost / regularizers should be done here ##############
    def compute_costs(self, log, p_y_given_x): # Needs to be run with initialized self._num_epochs_trained_tfv
        if not self._total_cost is None:
            log.print3("ERROR: Problem in Trainer. It was called to setup the total cost, but it was not None."+\
                       "\n\t This should not happen. Setup should be called only once.\n Exiting!")
            exit(1)
        
        # Cost functions
        cost = 0
        y_gt = self._net._output_gt_tensor_feeds['train']['y_gt']
        if "xentr" in self._losses_and_weights and self._losses_and_weights["xentr"] is not None:
            log.print3("COST: Using cross entropy with weight: " +str(self._losses_and_weights["xentr"]))
            w_per_cl_vec = self._compute_w_per_class_vector_for_xentr(self._net.num_classes, y_gt)
            cost += self._losses_and_weights["xentr"] * cfs.x_entr(p_y_given_x, y_gt, w_per_cl_vec)
        if "iou" in self._losses_and_weights and self._losses_and_weights["iou"] is not None:
            log.print3("COST: Using iou loss with weight: " +str(self._losses_and_weights["iou"]))
            cost += self._losses_and_weights["iou"] * cfs.iou(p_y_given_x, y_gt)
        if "dsc" in self._losses_and_weights and self._losses_and_weights["dsc"] is not None:
            log.print3("COST: Using dsc loss with weight: " +str(self._losses_and_weights["dsc"]))
            cost += self._losses_and_weights["dsc"] * cfs.dsc(p_y_given_x, y_gt)
            
        cost_L1_reg = self._L1_reg_weight * cfs.cost_L1(self._net.params_for_L1_L2_reg())
        cost_L2_reg = self._L2_reg_weight * cfs.cost_L2(self._net.params_for_L1_L2_reg())
        cost = cost + cost_L1_reg + cost_L2_reg
        
        self._total_cost = cost
        
    ############## Optimizer and schedules follows ##############
    # This is independent of the call to setup_costs (can be called before). Can be modularized.
    def create_optimizer(   self,
                            log,
                            sgd0orAdam1orRmsProp2,
                            lr_sched_params, # Could be given to init and saved there.
                            learning_rate_init,
                            momentum_init,
                            classicMomentum0OrNesterov1,
                            momentumTypeNONNormalized0orNormalized1,
                            b1ParamForAdam,
                            b2ParamForAdam,
                            epsilonForAdam,
                            rhoParamForRmsProp,
                            epsilonForRmsProp
                            ) :
        log.print3("...Initializing state of the optimizer...")
        
        self._lr_sched_params = lr_sched_params
        
        # Learning rate and momentum
        self._init_lr_tfv = tf.Variable(learning_rate_init, dtype="float32", trainable=False, name="init_lr") # This is important for the learning rate schedule to work.
        self._curr_lr =  self._get_lr_from_schedule()
        
        # SGD and RMS only.
        self._init_mom_tfv = tf.Variable(momentum_init, dtype="float32", trainable=False, name="init_mom")
        self._curr_mom = self._get_mom_from_schedule()
        
        # Optimizer
        params_to_opt = self._net.get_trainable_params(log, self._indicesOfLayersPerPathwayTypeToFreeze)
        if sgd0orAdam1orRmsProp2 == 0:
            self._optimizer = optimizers_dm.SgdOptimizer( params_to_opt,
                                                          self._curr_lr,
                                                          self._curr_mom,
                                                          momentumTypeNONNormalized0orNormalized1,
                                                          classicMomentum0OrNesterov1 )
        elif sgd0orAdam1orRmsProp2 == 1:
            self._optimizer = optimizers_dm.AdamOptimizer( params_to_opt,
                                                           self._curr_lr,
                                                           b1ParamForAdam,
                                                           b2ParamForAdam,
                                                           epsilonForAdam )
        elif sgd0orAdam1orRmsProp2 == 2:
            self._optimizer = optimizers_dm.RmsPropOptimizer( params_to_opt,
                                                              self._curr_lr,
                                                              self._curr_mom,
                                                              momentumTypeNONNormalized0orNormalized1,
                                                              classicMomentum0OrNesterov1,
                                                              rhoParamForRmsProp,
                                                              epsilonForRmsProp  )
        

        
    def get_total_cost(self):
        # Run only after: self.setup_costs(...)
        return self._total_cost
    
    # Called from within cnn3d.setup_ops_n_feeds_to_train()
    def get_param_updates_wrt_total_cost(self):
        # Excludes BN rolling average updates.
        updates = self._optimizer.get_update_ops_given_cost( self.get_total_cost() ) # A list of assign ops. For cnn AND optimizer's params.
        return updates
        
    def get_num_epochs_trained_tfv(self):
        return self._num_epochs_trained_tfv
    
    # Unused currently.
    def get_incr_num_epochs_trained_op(self):
        return self._op_increase_num_epochs_trained
    
    
    # ==== For cost schedules =====
    
    def _compute_w_per_class_vector_for_xentr(self, num_classes, y_gt, eps = 1e-6):
        # Re-weights samples in the cost function on a per-class basis.
        # E.g. to exclude a class, or counter class imbalance.
        # From first to given epoch, start from weighting classes equally to natural frequency, decreasing weighting linearly.
        # Return value: a function of epochs_trained_tfv
        
        if self._reweight_classes_in_cost is None or self._reweight_classes_in_cost["type"] is None: # No re-weighting.
            w_per_cl_vec = tf.ones( shape=[num_classes], dtype='float32' )
            
        else: # A type of reweighting has been specified
            
            if self._reweight_classes_in_cost["type"] == "freq":
                # Frequency re-weighting
                num_lbls_in_ygt = tf.cast( tf.reduce_prod(tf.shape(y_gt)), dtype="float32" )
                num_lbls_in_ygt_per_c = tf.math.bincount( arr = y_gt, minlength=num_classes, maxlength=num_classes, dtype="float32" ) # without the min/max, length of vector can change.
                y1 = (1./(num_lbls_in_ygt_per_c + eps)) * (num_lbls_in_ygt / num_classes)
                
            elif self._reweight_classes_in_cost["type"] == "per_c":
                # self._reweight_classes_in_cost["prms"] should be a list, with one float per class
                assert len(self._reweight_classes_in_cost["prms"]) == num_classes
                y1 = tf.constant(self._reweight_classes_in_cost["prms"], dtype="float32")
            
            # Linear schedule:
            lin_schedule_min_max_epoch = self._reweight_classes_in_cost["schedule"]
            assert lin_schedule_min_max_epoch[0] < lin_schedule_min_max_epoch[1]
            
            # yx - y1 = (x - x1) * (y2 - y1)/(x2 - x1)
            # yx = the multiplier I currently want, y1 = the multiplier at the beginning, y2 = the multiplier at the end
            # x = current epoch, x1 = epoch where linear decrease starts, x2 = epoch where linear decrease ends
            y2 = 1. # Where weight should be after end of schedule.
            
            x1 = tf.cast(lin_schedule_min_max_epoch[0], dtype="float32")
            x2 = tf.cast(lin_schedule_min_max_epoch[1], dtype="float32")
            x = tf.cast(self._num_epochs_trained_tfv, dtype="float32")
            # To handle the piecewise linear behaviour of x being before x1 and after x2 giving the same y as if =x1 or =x2 :
            x = tf.maximum(x1, x)
            x = tf.minimum(x, x2)
            yx = (x - x1) * (y2 - y1)/(x2 - x1) + y1
            w_per_cl_vec = yx

        return w_per_cl_vec
    
        
    def _get_mom_from_schedule(self):
        if self._lr_sched_params['type'] == 'expon':
            # Increased linearly.
            first_it_for_sch = self._lr_sched_params['expon']['epochs_wait_before_decr']
            final_it_for_sch = self._lr_sched_params['expon']['final_ep_for_sch'] # * subepochs_per_ep
            assert first_it_for_sch < final_it_for_sch
            curr_it =  tf.cast(self._num_epochs_trained_tfv, dtype='float32') # * subepochs_per_ep + curr_subepoch
            
            x_min = 0.
            x2 = final_it_for_sch - first_it_for_sch
            x = tf.maximum( tf.constant(0, dtype="float32"), curr_it - first_it_for_sch )
            x = tf.minimum( x, x2 )
            y_min = self._init_mom_tfv
            y_max = self._lr_sched_params['expon']['mom_to_reach_at_last_ep']
            
            curr_mom = (x - x_min)/(x2-x_min)  * (y_max - y_min) + y_max
        else :
            curr_mom = self._init_mom_tfv
            
        return curr_mom
    
    
    def _get_lr_from_schedule(self):
        TINY  = 1e-8
        
        if self._lr_sched_params['type'] == 'stable' :
            curr_lr = self._init_lr_tfv
            
        elif self._lr_sched_params['type'] == 'poly' :
            first_it_for_sch = self._lr_sched_params['poly']['epochs_wait_before_decr']
            final_it_for_sch = self._lr_sched_params['poly']['final_ep_for_sch'] # * subepochs_per_ep
            assert first_it_for_sch < final_it_for_sch
            curr_it = tf.cast(self._num_epochs_trained_tfv, dtype='float32') # * subepochs_per_ep + curr_subepoch
            
            #curr_lr = init_lr * ( 1 - x/x2) ^ power. Power = 0.9 in parsenet, which we validated to behave ok.
            x2 = final_it_for_sch - first_it_for_sch
            x = tf.maximum( tf.constant(0, dtype="float32"), curr_it - first_it_for_sch ) # to make schedule happen within the window (first, final) epoch, stable outside.
            x = tf.minimum( x, x2 ) # in case the current iteration is after max, so that I keep schedule stable afterwards. 
            y1 = self._init_lr_tfv
            y2 = 0.9
            curr_lr = y1 * tf.pow( 1.0 - x/x2, y2 )
            
        elif self._lr_sched_params['type'] == 'expon' :
            first_it_for_sch = self._lr_sched_params['expon']['epochs_wait_before_decr']
            final_it_for_sch = self._lr_sched_params['expon']['final_ep_for_sch'] # * subepochs_per_ep
            assert first_it_for_sch < final_it_for_sch
            curr_it = tf.cast(self._num_epochs_trained_tfv, dtype='float32')
            
            # y = y1 * gamma^x. gamma = (y2 / y1)^(1/x2)
            x2 = final_it_for_sch - first_it_for_sch
            x = tf.maximum( tf.constant(0, dtype="float32"), curr_it-first_it_for_sch )
            x = tf.minimum( x, x2 )
            y1 = self._init_lr_tfv
            y2 = self._lr_sched_params['expon']['lr_to_reach_at_last_ep']
            gamma = tf.pow( (y2+TINY)/y1, 1.0/x2 )
            curr_lr = y1 * tf.pow( gamma, x )
            
        elif self._lr_sched_params['type'] == 'predef' :
            #Predefined Schedule.
            div_lr_by = self._lr_sched_params['predef']['div_lr_by']
            epochs_boundaries = [ tf.cast(e, tf.int32) for e in self._lr_sched_params['predef']['epochs'] ]
            lr_values = [ ( self._init_lr_tfv / pow(div_lr_by, i) ) for i in range( 1+len(epochs_boundaries) ) ]
            curr_lr = tf.compat.v1.train.piecewise_constant(self._num_epochs_trained_tfv, boundaries = epochs_boundaries, values = lr_values)
        
        elif self._lr_sched_params['type'] == 'auto' :
            self._learning_rate_tfv = tf.Variable( self._init_lr_tfv, dtype="float32", trainable=False, name="curr_lr_tfv")
            self._top_mean_val_acc_tfv = tf.Variable(0, dtype="float32", trainable=False, name="top_mean_val_acc")
            self._epoch_with_top_mean_val_acc_tvf = tf.Variable(0, dtype=self._num_epochs_trained_tfv.dtype.as_numpy_dtype, trainable=False, name="ep_top_mean_val_acc")
            self._last_epoch_lr_got_lowered_tvf = tf.Variable(0, dtype="float32", trainable=False, name="last_ep_lr_lowered")
            
            self._op_assign_new_lr = tf.compat.v1.assign(self._learning_rate_tfv, self._tf_plchld_float32)
            self._op_assign_top_mean_val_acc_tfv = tf.compat.v1.assign(self._top_mean_val_acc_tfv, self._tf_plchld_float32)
            self._op_assign_epoch_with_top_mean_val_acc_tvf = tf.compat.v1.assign(self._epoch_with_top_mean_val_acc_tvf, self._tf_plchld_int32)
            self._op_assign_last_epoch_lr_lowered = tf.compat.v1.assign(self._last_epoch_lr_got_lowered_tvf, self._tf_plchld_float32)
            
            # The LR will be changed during the routine.training, by a call to function self.run_lr_sched_updates( sessionTf )
            curr_lr = self._learning_rate_tfv
                    
        return curr_lr
    
    
    ############### API REGARDING THE LR SCHEDULE ###########
    
    def run_updates_end_of_ep(self, log, sessionTf, mean_val_acc_of_ep):
        # In case I will need to do more than lr_sched_updates.
        self._run_lr_sched_updates(log, sessionTf, mean_val_acc_of_ep)
        
        # Done with everything in epoch. Increase number of trained epochs.
        num_eps_trained = sessionTf.run( self._op_increase_num_epochs_trained )
        
        log.print3("Trainer: Current learning rate: "+str( sessionTf.run(self._curr_lr) ))
        log.print3("Trainer: Current momentum: "+str( sessionTf.run(self._curr_mom) ))
        log.print3("Trainer: Number of epochs the model has been trained: "+str( num_eps_trained ))
        
        
        
    def _run_lr_sched_updates(self, log, sessionTf, mean_val_acc_of_ep): # This should be the only API.
        # The majority of schedules are implemented as tf operations on the graph.
        # Here, only implement the ones that need to change dynamically with things happening during training. Eg Auto.
        if self._lr_sched_params['type'] == 'auto':
            self._update_top_acc_if_needed(log, sessionTf, mean_val_acc_of_ep)
            self._run_auto_sched_updates(log, sessionTf)
            
            
    def _check_valid_func_call_for_lr_sched(self, log, list_allowed_scheds):
        if self._lr_sched_params['type'] not in list_allowed_scheds:
            log.print3( "ERROR: Asked to manually change learning rate. This is only expected if LR-schedule is auto."+\
                             "\n\t Current schedule is [" +str(self._lr_sched_type)+ "]. Exiting!")
            exit1(1)
    
    
    def _change_lr_to(self, log, sessionTf, new_lr) :
        self._check_valid_func_call_for_lr_sched(log, ['auto']) # currently should never be called with a different schedule.
        log.print3( "UPDATE: Changing the network's learning rate to: " + str(new_lr) )
        sessionTf.run( fetches = self._op_assign_new_lr, feed_dict = { self._tf_plchld_float32: new_lr } )
        last_epoch_lr_lowered = self._num_epochs_trained_tfv.eval(session=sessionTf)
        sessionTf.run( fetches = self._op_assign_last_epoch_lr_lowered, feed_dict = { self._tf_plchld_float32: last_epoch_lr_lowered } )
        
        
    def _divide_lr_by(self, log, sessionTf, div_lr_by) :
        self._check_valid_func_call_for_lr_sched(log, ['auto']) # currently should never be called with a different schedule.
        old_lr = sessionTf.run( fetches=self._learning_rate_tfv )
        new_lr = old_lr / div_lr_by
        self._change_lr_to(log, sessionTf, new_lr)
        
        
    ## AUTO SCHEDULE specific ##    
    def _run_auto_sched_updates(self, log, sessionTf):
        self._check_valid_func_call_for_lr_sched(log, ['auto'])
        num_epochs_trained = self._num_epochs_trained_tfv.eval(session=sessionTf)
        epoch_with_top_mean_val_acc = self._epoch_with_top_mean_val_acc_tvf.eval(session=sessionTf)
        last_epoch_lr_got_lowered = self._last_epoch_lr_got_lowered_tvf.eval(session=sessionTf)
        epochs_wait_before_decr = self._lr_sched_params['auto']['epochs_wait_before_decr']
        
        if (num_epochs_trained >= epoch_with_top_mean_val_acc + epochs_wait_before_decr) and \
                (num_epochs_trained >= last_epoch_lr_got_lowered + epochs_wait_before_decr) :
            
            log.print3("DEBUG: Going to lower Learning Rate because of [AUTO] schedule." +\
                            "\n\t The network has been trained for: " + str(num_epochs_trained) + " epochs." +\
                            "\n\t Epoch with highest achieved validation accuracy: " + str(epoch_with_top_mean_val_acc) +\
                            "\n\t Epoch that learning rate was lowered last time: " + str(last_epoch_lr_got_lowered) +\
                            "\n\t Waited that many epochs for accuracy to increase: " +str(epochs_wait_before_decr) + " epochs." +\
                            "\n\t Going to lower learning rate...")
            
            self._divide_lr_by(log, sessionTf, self._lr_sched_params['auto']['div_lr_by'])
    
    
    def _update_top_acc_if_needed(self, log, sessionTf, mean_val_acc_of_ep) :
        # Called at the end of an epoch, right before increasing self._num_epochs_trained_tfv
        assert (mean_val_acc_of_ep is not None) and (mean_val_acc_of_ep >= 0) # flags in case validation is not performed.
        top_mean_val_acc = self._top_mean_val_acc_tfv.eval(session=sessionTf)
        num_epochs_trained = self._num_epochs_trained_tfv.eval(session=sessionTf)
        
        if mean_val_acc_of_ep > top_mean_val_acc + self._lr_sched_params['auto']['min_incr_of_val_acc_considered'] :
            log.print3("UPDATE: In this epoch the CNN achieved a new highest mean validation accuracy of: " + str(mean_val_acc_of_ep))
            sessionTf.run( fetches=self._op_assign_top_mean_val_acc_tfv, feed_dict={ self._tf_plchld_float32: mean_val_acc_of_ep } )
            sessionTf.run( fetches=self._op_assign_epoch_with_top_mean_val_acc_tvf, feed_dict={ self._tf_plchld_int32: num_epochs_trained } )
            
    

        

    
    
    

    
        
        
        

        
        
        
