# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np

import tensorflow as tf
from deepmedic.neuralnet.pathways import Pathways
from deepmedic.neuralnet.blocks import SoftmaxBlock
from deepmedic.config.model import ModelConfig
from deepmedic.logging.loggers import Logger

##################################################
##################################################
################ THE CNN CLASS ###################
##################################################
##################################################


class Cnn3d:
    def __init__(self, config: ModelConfig, log: Logger = None):
        if log is None:
            log = Logger()
        self._rng = np.random.RandomState(seed=None)
        self.log = log
        self.config = config

        self.num_subs_paths = len(self.config.subsampled_pathway_configs)

        self.num_classes = self.config.n_classes

        # There should be only 1 normal and only one FC pathway. Eg, see self.getFcPathway()
        self.pathways = Pathways(
            n_input_channels=self.config.n_input_channels,
            n_classes=self.config.n_classes,
            activation_function=self.config.activation_function,
            n_batches_for_bn_mov_avg=self.config.n_batches_for_bn_mov_avg,
            normal_pathway_config=self.config.normal_pathway_config,
            subsampled_pathway_configs=self.config.subsampled_pathway_configs,
            fc_pathway_config=self.config.fc_layers_config,
            conv_w_init_type=self.config.conv_w_init_type,
            log=self.log,
            rng=self._rng,
        )
        self.final_target_layer = SoftmaxBlock(
            n_fms=self.pathways.fc_pathway().get_n_fms_out(),
            temperature=self.config.fc_layers_config.softmax_temperature
        )

        # ======= Output tensors Y_GT ========
        # For each targetLayer, I should be placing a y_gt placeholder/feed.
        self._output_gt_tensor_feeds = {"train": {}, "val": {}}

        # These entries are setup in the setup_train/val/test functions here ############
        self._ops_main = {"train": {}, "val": {}, "test": {}}
        self._feeds_main = {"train": {}, "val": {}, "test": {}}

    def get_num_subs_pathways(self):
        return len(self.pathways.subsampled_pathways())

    def get_num_pathways_that_require_input(self):
        return 1 + self.get_num_subs_pathways()

    def update_arrays_of_bn_moving_avg(self, sessionTf):
        """
        for inference with batch-normalization.
        Every training batch, this is called to update an internal matrix of each layer, with the last mus and vars,
        so that I can compute the rolling average for inference.
        """
        self.pathways.update_arrays_of_bn_moving_avg(sessionTf)

    def _get_update_ops_for_bn_moving_avg(self):
        """
        These are not the variables of the normalization of the FMs' distributions that are optimized during training.
        These are only the Mu and Stds that are used during inference,
        ... and here we update the sharedVariable which is used "from the outside during do_training()"
        to update the rolling-average-matrix for inference. Do for all layers.
        """

        updates_for_bn_rolling_average = []
        for pathway in self.pathways:
            for block in pathway.get_blocks():
                updates_for_bn_rolling_average.extend(block.get_update_ops_for_bn_moving_avg())
        return updates_for_bn_rolling_average

    def get_trainable_params(self, log, indices_of_layers_per_pathway_type_to_freeze):
        # Called from Trainer.
        params_to_opt_during_training = []  # Ws and Bs
        for pathway in self.pathways:
            for block_i in range(len(pathway.get_blocks())):
                if block_i not in indices_of_layers_per_pathway_type_to_freeze[pathway.pType()]:
                    params_to_opt_during_training = (
                        params_to_opt_during_training + pathway.get_block(block_i).trainable_params()
                    )
                else:  # Layer will be held fixed. Notice that Batch Norm parameters are still learnt.
                    log.print3(
                        "WARN: [Pathway_{ptype}] The weights of [Layer-{layer}] will NOT be trained as specified "
                        "(index, first layer is 0).".format(ptype=str(pathway.get_str_type()), layer=str(block_i))
                    )
        return params_to_opt_during_training

    def params_for_L1_L2_reg(self):
        total_params = []
        for pathway in self.pathways:
            for block in pathway.get_blocks():
                total_params += block.params_for_L1_L2_reg()
        return total_params

    def get_main_ops(self, str_train_val_test):
        # str_train_val_test: "train", "val" or "test"
        return self._ops_main[str_train_val_test]

    def get_main_feeds(self, str_train_val_test):
        return self._feeds_main[str_train_val_test]

    def setup_ops_n_feeds_to_train(
        self, inp_plchldrs, p_y_given_x, total_cost, updates_of_params_wrt_total_cost, log: Logger = None
    ):
        if log is None:
            log = self.log
        log.print3("...Building the training function...")

        y_gt = self._output_gt_tensor_feeds["train"]["y_gt"]

        # ================BATCH NORMALIZATION ROLLING AVERAGE UPDATES======================
        updates = updates_of_params_wrt_total_cost + self._get_update_ops_for_bn_moving_avg()
        updates_grouped_op = tf.group(*updates)  # this op returns no output when run.

        # ======================== Collecting ops and feeds =================
        log.print3("...Collecting ops and feeds for training...")

        self._ops_main["train"]["cost"] = total_cost
        self._ops_main["train"]["list_rp_rn_tp_tn"] = self.final_target_layer.get_rp_rn_tp_tn(p_y_given_x, y_gt)
        self._ops_main["train"]["updates_grouped_op"] = updates_grouped_op

        self._feeds_main["train"]["x"] = inp_plchldrs["x"]
        for subpath_i in range(self.num_subs_paths):  # if there are subsampled paths...
            self._feeds_main["train"]["x_sub_" + str(subpath_i)] = inp_plchldrs["x_sub_" + str(subpath_i)]
        self._feeds_main["train"]["y_gt"] = y_gt

        log.print3("Done.")

    def setup_ops_n_feeds_to_val(self, inp_plchldrs, p_y_given_x, log: Logger = None):
        if log is None:
            log = self.log
        log.print3("...Building the validation function...")

        y_gt = self._output_gt_tensor_feeds["val"]["y_gt"]

        log.print3("...Collecting ops and feeds for validation...")

        self._ops_main["val"]["list_rp_rn_tp_tn"] = self.final_target_layer.get_rp_rn_tp_tn(p_y_given_x, y_gt)

        self._feeds_main["val"]["x"] = inp_plchldrs["x"]
        for subpath_i in range(self.num_subs_paths):  # if there are subsampled paths...
            self._feeds_main["val"]["x_sub_" + str(subpath_i)] = inp_plchldrs["x_sub_" + str(subpath_i)]
        self._feeds_main["val"]["y_gt"] = y_gt

        log.print3("Done.")

    def setup_ops_n_feeds_to_test(
        self, inp_plchldrs, p_y_given_x, indices_fms_per_pathtype_per_layer_to_save=None, log: Logger = None
    ):
        if log is None:
            log = self.log
        log.print3("...Building the function for testing and visualisation of FMs...")

        list_to_return_with_all_the_fm_activations_per_layer = []
        if indices_fms_per_pathtype_per_layer_to_save is not None:
            for pathway in self.pathways:
                indices_of_fms_to_visualise_per_layer_of_certain_pathway = indices_fms_per_pathtype_per_layer_to_save[
                    pathway.pType()
                ]
                if indices_of_fms_to_visualise_per_layer_of_certain_pathway != []:
                    blocks = pathway.get_blocks()
                    for block_i in range(len(blocks)):  # each layer that this pathway/fc has.
                        indices_of_fms_to_extract_from_this_layer = indices_of_fms_to_visualise_per_layer_of_certain_pathway[
                            block_i
                        ]
                        if (
                            len(indices_of_fms_to_extract_from_this_layer) > 0
                        ):  # if no FMs are to be taken, this should be []
                            list_to_return_with_all_the_fm_activations_per_layer.append(
                                blocks[block_i].fm_activations(indices_of_fms_to_extract_from_this_layer)
                            )

        log.print3("...Collecting ops and feeds for testing...")

        self._ops_main["test"]["list_of_fms_per_layer"] = list_to_return_with_all_the_fm_activations_per_layer
        self._ops_main["test"]["pred_probs"] = p_y_given_x

        self._feeds_main["test"]["x"] = inp_plchldrs["x"]
        for subpath_i in range(self.num_subs_paths):  # if there are subsampled paths...
            self._feeds_main["test"]["x_sub_" + str(subpath_i)] = inp_plchldrs["x_sub_" + str(subpath_i)]

        log.print3("Done.")

    def create_input_placeholders(self, train_val_test: str):  # TODO: Remove for eager
        assert train_val_test in ["train", "val", "test"]
        if train_val_test == "train":
            inp_dims = self.config.segment_dim_train
        elif train_val_test == "val":
            inp_dims = self.config.segment_dim_val
        elif train_val_test == "test":
            inp_dims = self.config.segment_dim_inference
        else:
            raise ValueError('train_val_test must be in ["train", "val", "test"]')
        inp_shapes_per_path = self.pathways.calc_inp_dims_of_paths_from_hr_inp(inp_dims)
        return self._setup_input_placeholders(train_val_test, inp_shapes_per_path), inp_shapes_per_path

    def _setup_input_placeholders(self, train_val_test, inp_shapes_per_path):  # TODO: REMOVE for eager
        assert train_val_test in ["train", "val", "test"]
        input_placeholders = {}
        input_placeholders["x"] = tf.compat.v1.placeholder(
            dtype="float32",
            shape=[None, self.pathways.normal_pathway().get_n_fms_in()] + inp_shapes_per_path[0],
            name="inp_x_" + train_val_test,
        )
        for subpath_i in range(self.num_subs_paths):  # if there are subsampled paths...
            input_placeholders["x_sub_" + str(subpath_i)] = tf.compat.v1.placeholder(
                dtype="float32",
                shape=[None, self.pathways.normal_pathway().get_n_fms_in()] + inp_shapes_per_path[subpath_i + 1],
                name="inp_x_sub_" + str(subpath_i) + "_" + train_val_test,
            )
        return input_placeholders

    def build(self, log: Logger = None):
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        if log is None:
            log = self.log

        log.print3("...Building the CNN model...")
        self.pathways.build()
        # =========== Make the final Target Layer (softmax, regression, whatever) ==========
        log.print3("Adding the final Softmax layer...")
        self.final_target_layer.build()
        self.pathways.fc_pathway().get_block(-1).connect_target_block(self.final_target_layer)

        # =============== BUILDING FINISHED - BELOW IS TEMPORARY ========================
        self._output_gt_tensor_feeds["train"]["y_gt"] = tf.compat.v1.placeholder(
            dtype="int32", shape=[None, None, None, None], name="y_train"
        )
        self._output_gt_tensor_feeds["val"]["y_gt"] = tf.compat.v1.placeholder(
            dtype="int32", shape=[None, None, None, None], name="y_val"
        )

        log.print3("Finished building the CNN's model.")

    def apply(self, inputs_per_pathw, mode, train_val_test, verbose=True, log=None):
        # Currently applies it on the placeholders. TODO: On actual input.
        # train_val_test: TEMPORARY. ONLY TO RETURN FMS. REMOVE IN END OF REFACTORING.
        # assert len(inputs_per_pathw) == len(self.pathways) - 1
        if log is None:
            log = self.log
        # ===== Apply High-Res path =========
        input = inputs_per_pathw["x"]
        out = self.pathways.normal_pathway().apply(input, mode, train_val_test, verbose, log)
        dims_outp_pathway_hr = out.shape
        fms_from_paths_to_concat = [out]

        # === Subsampled pathways =========
        for subpath_i, sub_pathway in enumerate(self.pathways.subsampled_pathways()):
            input = inputs_per_pathw["x_sub_" + str(subpath_i)]
            out_lr = sub_pathway.apply(input, mode, train_val_test, verbose, log)
            # this creates essentially the "upsampling layer"
            out = sub_pathway.upsample_to_high_res(out_lr, shape_to_match=dims_outp_pathway_hr, upsampl_type="repeat")
            fms_from_paths_to_concat.append(out)

        # ===== Concatenate and final convs ========
        conc_inp_fms = tf.concat(fms_from_paths_to_concat, axis=1)
        logits_no_bias = self.pathways.fc_pathway().apply(conc_inp_fms, mode, train_val_test, verbose, log)
        # Softmax
        p_y_given_x = self.final_target_layer.apply(logits_no_bias, mode)

        return p_y_given_x

    def calc_outp_dims_given_inp(self, inp_dims_hr_path):
        outp_dims_hr_path = self.pathways.normal_pathway().calc_outp_dims_given_inp(inp_dims_hr_path)
        return self.pathways.fc_pathway().calc_outp_dims_given_inp(outp_dims_hr_path)

    def calc_unpredicted_margin(self, inp_dims_hr_path):
        # unpred_margin: [[before-x, after-x], [before-y, after-y], [before-z, after-z]]
        outp_dims = self.calc_outp_dims_given_inp(inp_dims_hr_path)
        n_unpred_vox = [inp_dims_hr_path[d] - outp_dims[d] for d in range(3)]
        unpred_margin = [[n_unpred_vox[d] // 2, n_unpred_vox[d] - n_unpred_vox[d] // 2] for d in range(3)]
        return unpred_margin
