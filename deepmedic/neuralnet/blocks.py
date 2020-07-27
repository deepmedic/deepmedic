# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
from typing import List
import tensorflow as tf
import numpy as np
import deepmedic.neuralnet.layers as dm_layers


"""
#################################################################
#                         Block Types                           #
#################################################################
Blocks are sequences of layers.
Blocks dont create new trainable parameters.
Anything that adds new trainable parameters is a layer, added to the block/sequence.
Inheritance:
Block -> ConvBlock -> LowRankConvBlock
               L-----> ConvBlockWithSoftmax
       L-----> SoftmaxBlock (could be changed to layer)
"""


class Block:
    def __init__(self, n_fms_in: int, n_fms_out: int):
        # === Basic architecture parameters ===
        self._n_fms_in = n_fms_in
        self._n_fms_out = n_fms_out

        # === All layers that the block applies ===
        self._layers = []
        # Keep track to update moving avg. Only when rollingAverageForBn>0 AND use_bn,
        # with the latter used for the 1st layers of pathways (on image).
        self._bn_l = None

        # === Output of the block ===
        self.output = {"train": None, "val": None, "test": None} # TODO: Remove for eigen.

        # ==== Target Block Connected to that layer (softmax, regression, auxiliary loss etc), if any ======
        self._target_blocks = []

    def rec_field(self, rf_at_inp: List = None, stride_rf_at_inp: List = None):
        """
        Returns: Rf of neurons at the output of the final layer of the block, with respect to input.
                 Stride of rf at the block's output wrt input (how much it shifts at inp if we shift 1 neuron at out)
        """
        if rf_at_inp is None:
            rf_at_inp = [1, 1, 1]
        if stride_rf_at_inp is None:
            stride_rf_at_inp = [1, 1, 1]
        rf_prev_layer = rf_at_inp
        stride_rf_prev_layer = stride_rf_at_inp
        for layer in self._layers:
            rf_prev_layer, stride_rf_prev_layer = layer.rec_field(rf_prev_layer, stride_rf_prev_layer)
        return rf_prev_layer, stride_rf_prev_layer

    def calc_outp_dims_given_inp(self, inp_dims):
        outp_dims_prev_layer = inp_dims
        for layer in self._layers:
            outp_dims_prev_layer = layer.calc_outp_dims_given_inp(outp_dims_prev_layer)
        return outp_dims_prev_layer

    def calc_inp_dims_given_outp(self, outp_dims):
        inp_dims_deeper_layer = outp_dims
        for layer in self._layers:
            inp_dims_deeper_layer = layer.calc_inp_dims_given_outp(inp_dims_deeper_layer)
        return inp_dims_deeper_layer

    # Getters
    def get_n_fms_in(self):
        return self._n_fms_in

    def get_n_fms_out(self):
        return self._n_fms_out

    def fm_activations(self, indices_of_fms):
        # indices_of_fms_in_layer_to_visualise_from_to_exclusive
        return self.output["test"][:, indices_of_fms[0]: indices_of_fms[1], :, :, :]

    def apply(self, input, mode):
        # Main functionality
        # mode: 'train' or 'infer'
        signal = input
        for layer in self._layers:
            signal = layer.apply(signal, mode)
        return signal

    def connect_target_block(self, new_target_block_instance):
        # new_target_block_instance : eg softmax layer. Future: Regression layer, or other auxiliary classifiers.
        self._target_blocks += [new_target_block_instance]

    def trainable_params(self):
        total_params = []
        for layer in self._layers:
            total_params += layer.trainable_params()  # concat lists
        for block in self._target_blocks:
            total_params += block.trainable_params()  # concat lists
        return total_params

    def params_for_L1_L2_reg(self):
        total_params = []
        for layer in self._layers:
            total_params += layer.params_for_L1_L2_reg()
        for block in self._target_blocks:
            total_params += block.params_for_L1_L2_reg()
        return total_params

    def update_arrays_of_bn_moving_avg(self, sessionTf):
        # This function should be erased when I reimplement the Rolling average.
        if self._bn_l is not None :
            self._bn_l.update_arrays_of_bn_moving_avg(sessionTf)

    def get_update_ops_for_bn_moving_avg(self) :
        return self._bn_l.get_update_ops_for_bn_moving_avg() if self._bn_l is not None else []


class ConvBlock(Block):
    def __init__(
        self,
        n_fms_in: int,
        n_fms_out: int,
        conv_kernel_dims: List,
        pool_prms: List,
        conv_pad_mode: str,
        use_bn: bool,
        moving_avg_length: int,
        rng: np.random.RandomState,
        conv_w_init_type: List,
        activ_func: str = "relu",
        dropout_rate: float = 0.0
    ):
        super().__init__(
            n_fms_in=n_fms_in,
            n_fms_out=n_fms_out
        )
        self._conv_kernel_dims = conv_kernel_dims
        self._pool_prms = pool_prms
        self._conv_pad_mode = conv_pad_mode
        self._use_bn = use_bn
        self._moving_avg_length = moving_avg_length
        self._activ_func = activ_func
        self._dropout_rate = dropout_rate
        self._rng = rng
        self._conv_w_init_type = conv_w_init_type

    def build(self):
        # The main function that builds this.
        """
            param rng: numpy.random.RandomState used to initialize weights
            param inputToLayer: tensor5 of shape inputToLayerShape
            param filterShape: (number of filters, num input feature maps,
                                filter height, filter width, filter depth)
            param inputToLayerShape: (batch size, num input feature maps,
                                image height, image width, filter depth)
            use_bn: True of False. Used to not allow doing BN on first layers straight on image,
                    even if rollingAvForBnOverThayManyBatches > 0.
        """

        #  Order of what is applied, ala He et al "Identity Mappings in Deep Residual Networks" 2016
        #  Input -> [ BatchNorm OR biases applied] -> NonLinearity -> DropOut -> Pooling --> Conv ]

        # ------------------ Batch Normalization ------------------
        if self._use_bn and self._moving_avg_length > 0:
            self._bn_l = dm_layers.BatchNormLayer(self._moving_avg_length, n_channels=self._n_fms_in)
            self._layers.append(self._bn_l)
        else:
            # Not using batch normalization
            # make the bias terms and apply them. Like the old days before BN's own learnt bias terms.
            bias_l = dm_layers.BiasLayer(self._n_fms_in)
            self._layers.append(bias_l)

        # ------------ Apply Activation/ non-linearity -----------
        act_l = dm_layers.get_act_layer(self._activ_func, self._n_fms_in)
        self._layers.append(act_l)

        # ------------- Dropout --------------
        dropout_l = dm_layers.DropoutLayer(self._dropout_rate, self._rng)
        self._layers.append(dropout_l)

        # -----------  Pooling ----------------------------------
        if len(self._pool_prms) > 0 : #Max pooling is actually happening here...
            # if len == 0, pool_prms == [], no max pooling before this conv.
            pooling_l = dm_layers.PoolingLayer(self._pool_prms[0], self._pool_prms[1], self._pool_prms[2], self._pool_prms[3])
            self._layers.append(pooling_l)

        # --------- Convolution ---------------------------------
        conv_l = self._create_conv_layer(self._conv_w_init_type, self._rng)
        self._layers.append(conv_l)

    def _create_conv_layer(self, init_method, rng):
        return dm_layers.ConvolutionalLayer(
            self._n_fms_in, self._n_fms_out, self._conv_kernel_dims, init_method, self._conv_pad_mode, rng
        )


class LowRankConvBlock(ConvBlock):
    """
    Ala Yani Ioannou et al, Training CNNs with Low-Rank Filters For Efficient Image Classification, ICLR 2016.
    Allowed Ranks: Rank=1 or 2.
    """
    def __init__(
        self,
        n_fms_in: int,
        n_fms_out: int,
        conv_kernel_dims: List,
        pool_prms: List,
        conv_pad_mode: str,
        use_bn: bool,
        moving_avg_length: int,
        rng: np.random.RandomState,
        conv_w_init_type: List,
        activ_func: str = "relu",
        dropout_rate: float = 0.0,
        rank: int = 2
    ):
        super().__init__(
            n_fms_in=n_fms_in,
            n_fms_out=n_fms_out,
            conv_kernel_dims=conv_kernel_dims,
            pool_prms=pool_prms,
            conv_pad_mode=conv_pad_mode,
            use_bn=use_bn,
            moving_avg_length=moving_avg_length,
            activ_func=activ_func,
            dropout_rate=dropout_rate,
            rng=rng,
            conv_w_init_type=conv_w_init_type
        )
        self._rank = rank  # 1 or 2 dimensions

    def _create_conv_layer(self, init_method, rng):
        # Overload the ConvBlock's function. Called from build. The only different behaviour.
        return dm_layers.LowRankConvolutionalLayer(
            self._n_fms_in, self._n_fms_out, self._conv_kernel_dims, init_method, self._conv_pad_mode, rng
        )


class SoftmaxBlock(Block):
    """ Softmax for classification. Note, this is simply the softmax function, after adding bias. Not a ConvBlock """
    def __init__(self, n_fms: int, temperature: float):
        super().__init__(n_fms_in=n_fms, n_fms_out=n_fms)
        self._temperature = temperature
        self._bias_l = None

    def build(self):
        # t: temperature. Scalar
        self._bias_l = dm_layers.BiasLayer(self._n_fms_in)
        self._layers.append(self._bias_l)

    def apply(self, input, mode):
        """
        At this last classification layer, the conv output needs to have bias added before the softmax.
        NOTE: So, two biases are associated with this layer. self.b which is added in the ouput of the
        previous layer's output of conv, and this self._bClassLayer that is added only to this final
        output before the softmax.
        """

        logits = self._bias_l.apply(input, mode)
        p_y_given_x = tf.nn.softmax(logits/self._temperature, axis=1)
        return p_y_given_x

    def get_rp_rn_tp_tn(self, p_y_given_x, y_gt):
        """
        The returned list has (numberOfClasses)x4 integers: >numberOfRealPositives, numberOfRealNegatives,
        numberOfTruePredictedPositives, numberOfTruePredictedNegatives< for each class (incl background).
        Order in the list is the natural order of the classes
        (ie class-0 RP,RN,TPP,TPN, class-1 RP,RN,TPP,TPN, class-2 RP,RN,TPP,TPN ...)
            param y_gt: y_gt = T.itensor4('y'). Dimensions [batchSize, r, c, z]
            mode: 'train' or 'infer'
        """

        y_pred = tf.argmax(p_y_given_x, axis=1)

        list_num_rp_rn_tp_tn_per_class = []

        for class_i in range(0, self._n_fms_out):
            # Number of Real Positive, Real Negatives, True Predicted Positives and True Predicted Negatives
            # are reported PER CLASS (first for WHOLE).
            is_rp = tf.equal(y_gt, class_i)
            is_rn = tf.logical_not(is_rp)

            is_predicted_pos = tf.equal(y_pred, class_i)
            is_predicted_neg = tf.logical_not(is_predicted_pos)
            is_tp = tf.logical_and(is_rp,is_predicted_pos)
            is_tn = tf.logical_and(is_rn,is_predicted_neg)

            list_num_rp_rn_tp_tn_per_class.append(tf.reduce_sum(tf.cast(is_rp, dtype="int32")))
            list_num_rp_rn_tp_tn_per_class.append(tf.reduce_sum(tf.cast(is_rn, dtype="int32")))
            list_num_rp_rn_tp_tn_per_class.append(tf.reduce_sum(tf.cast(is_tp, dtype="int32")))
            list_num_rp_rn_tp_tn_per_class.append(tf.reduce_sum(tf.cast(is_tn, dtype="int32")))

        return list_num_rp_rn_tp_tn_per_class

    # Not used.
    def mean_error(self, y_pred, y_gt):
        """
        Returns float = number of errors / number of examples of the minibatch ; [0., 1.]
        y_gt = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        """
        # check if y is of the correct datatype
        if y_gt.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            tneq = tf.logical_not( tf.equal(y_pred, y_gt) )
            mean_error = tf.reduce_mean(tneq)
            return mean_error #The percentage of the predictions that is not the correct class.
        else:
            raise NotImplementedError("Not implemented behaviour for y_gt.dtype different than int.")
