# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from six.moves import input
import os

import tensorflow as tf
from deepmedic.frontEnd.session import Session
from deepmedic.frontEnd.configParsing.utils import abs_from_rel_path
from deepmedic.frontEnd.configParsing.testSessionParams import TestSessionParameters
from deepmedic.frontEnd.sessHelpers import make_folders_for_test_session, handle_exception_tf_restore
from deepmedic.neuralnet.cnn3d import Cnn3d
from deepmedic.routines.testing import inference_on_whole_volumes


class TestSession(Session):
    
    def __init__(self, cfg):
        self._out_folder_preds = None
        self._out_folder_fms = None
        
        Session.__init__(self, cfg)
        
    def _make_session_name(self):
        session_name = TestSessionParameters.get_session_name(self._cfg[self._cfg.SESSION_NAME])
        return session_name
    
    def make_output_folders(self):
        self._main_out_folder_abs = abs_from_rel_path(self._cfg[self._cfg.FOLDER_OUTP], self.get_abs_path_to_cfg())
        [self._out_folder_logs,
         self._out_folder_preds,
         self._out_folder_fms] = make_folders_for_test_session(self._main_out_folder_abs, self._session_name)
         
         
    def compile_session_params_from_cfg(self, *args):
        (model_params,) = args
        
        self._params = TestSessionParameters(self._log,
                                             self._main_out_folder_abs,
                                             self._out_folder_preds,
                                             self._out_folder_fms,
                                             model_params.get_n_classes(),
                                             self._cfg)
        
        self._log.print3("")
        self._log.print3("============     NEW TESTING SESSION    ===============")
        self._params.print_params()    
        self._log.print3("=======================================================\n")
        
        return self._params
        
        
    def _ask_user_if_test_with_random(self):
        user_input = None
        try:
            user_input = input("WARN:\t Testing was requested, but without specifying a pretrained, saved model to load.\n"+\
                                   "\t A saved model can be specified via the command line or the test-config file.\n" +\
                                   "\t Please see documentation or run ./deepMedicRun -h for help on how to load a model.\n"+\
                                   "\t Do you wish to continue and test inference with a randomly initialized model? [y/n] : ")
            while user_input not in ['y','n']: 
                user_input = input("Please specify 'y' or 'n': ")
        except:
            print("\nERROR:\tTesting was requested, but without specifying a pretrained, saved model to load."+\
                  "\n\tTried to ask for user input whether to continue testing with a randomly initialized model, but failed."+\
                  "\n\tReason unknown (nohup? remote?)."+\
                  "\n\tPlease see documentation or run ./deepMedicRun -h for help on how to load a model."+\
                  "\n\tExiting."); exit(1)
        if user_input == 'y':
            pass
        else:
            print("Exiting as requested."); exit(0)
    
    
    def run_session(self, *args):
        (sess_device,
         model_params,) = args
        
        graphTf = tf.Graph()
        
        with graphTf.as_default():
            with graphTf.device(sess_device): # Throws an error if GPU is specified but not available.
                self._log.print3("=========== Making the CNN graph... ===============")
                cnn3d = Cnn3d()
                with tf.compat.v1.variable_scope("net"):
                    cnn3d.make_cnn_model(*model_params.get_args_for_arch())  # Creates network's graph (no optimizer)
                    inp_plchldrs, inp_shapes_per_path = cnn3d.create_inp_plchldrs(model_params.get_inp_dims_hr_path('test'), 'test')
                    p_y_given_x = cnn3d.apply(inp_plchldrs, 'infer', 'test', verbose=True, log=self._log)
                    
            self._log.print3("=========== Compiling the Testing Function ============")
            self._log.print3("=======================================================\n")
            
            cnn3d.setup_ops_n_feeds_to_test(self._log, inp_plchldrs, p_y_given_x, self._params.inds_fms_per_pathtype_per_layer_to_save)
            # Create the saver
            coll_vars_net = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="net")
            saver_net = tf.compat.v1.train.Saver(var_list=coll_vars_net)  # saver_net would suffice
            # TF2: dict_vars_net = {'net_var'+str(i): v for i, v in enumerate(coll_vars_net)}
            # TF2: ckpt_net = tf.train.Checkpoint(**dict_vars_net)
            
        with tf.compat.v1.Session(graph=graphTf, config=tf.compat.v1.ConfigProto(log_device_placement=False, device_count={'CPU':999, 'GPU':99})) as sessionTf:
            file_to_load_params_from = self._params.get_path_to_load_model_from()
            if file_to_load_params_from is not None: # Load params
                self._log.print3("=========== Loading parameters from specified saved model ===============")
                chkpt_fname = tf.train.latest_checkpoint(file_to_load_params_from) if os.path.isdir(file_to_load_params_from) else file_to_load_params_from
                self._log.print3("Loading parameters from:" + str(chkpt_fname))
                try:
                    saver_net.restore(sessionTf, chkpt_fname)
                    # TF2: ckpt_net.restore(chkpt_fname)
                    self._log.print3("Parameters were loaded.")
                except Exception as e: handle_exception_tf_restore(self._log, e)
                
            else:
                self._ask_user_if_test_with_random()  # Asks user whether to continue with randomly initialized model.
                self._log.print3("")
                self._log.print3("=========== Initializing network variables  ===============")
                tf.compat.v1.variables_initializer(var_list=coll_vars_net).run()
                self._log.print3("Model variables were initialized.")
                
                
            self._log.print3("")
            self._log.print3("======================================================")
            self._log.print3("=========== Testing with the CNN model ===============")
            self._log.print3("======================================================")
            
            res_code = inference_on_whole_volumes(*([sessionTf, cnn3d] +
                                                    self._params.get_args_for_testing() +
                                                    [inp_shapes_per_path]))
        
        self._log.print3("")
        self._log.print3("======================================================")
        self._log.print3("=========== Testing session finished =================")
        self._log.print3("======================================================")
