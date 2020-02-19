# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
import copy
from math import ceil

import tensorflow as tf

from deepmedic.neuralnet.pathwayTypes import PathwayTypes
from deepmedic.neuralnet.blocks import ConvBlock, LowRankConvBlock
import deepmedic.neuralnet.ops as ops


#################################################################
#                        Classes of Pathways                    #
#################################################################

class Pathway(object):
    # This is a virtual class.

    def __init__(self, pName=None) :
        self._pName = pName
        self._pType = None # Pathway Type.

        # === Input to the pathway ===
        self._n_fms_in = None
        # === Basic architecture parameters ===
        self._blocks = []
        self._subs_factor = [1,1,1]
        self._inds_of_blocks_for_res_conns_at_out = None

        # === Output of the block ===
        self._n_fms_out = None

    # Getters
    def get_n_fms_in(self):
        return self._n_fms_in
    def get_n_fms_out(self):
        return self._n_fms_out

    def apply(self, input, mode, train_val_test, verbose=False, log=None):
        # mode: 'train' / 'infer'
        # train_val_test: TEMPORARY. ONLY TO RETURN FMS. REMOVE IN END OF REFACTORING.
        if verbose:
            log.print3("Pathway ["+str(self.getStringType())+"], Mode: [" + mode + "], Input's Shape: " + str(input.shape))

        input_to_prev_layer = None
        input_to_next_layer = input

        for idx, block in enumerate(self._blocks):
            if verbose:
                log.print3("\tBlock [" + str(idx) + "], Mode: [" + mode + "], Input's Shape: " + str(input_to_next_layer.shape))

            out = block.apply(input_to_next_layer, mode)
            block.output[train_val_test] = out # HACK TEMPORARY. ONLY USED FOR RETURNING FMS.
            if idx not in self._inds_of_blocks_for_res_conns_at_out: #not a residual connecting here
                input_to_prev_layer = input_to_next_layer
                input_to_next_layer = out
            else : #make residual connection
                assert idx > 0 # The very first block (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                log.print3("\tResidual-Connection made between: output of [Block_"+str(idx)+"] & input of previous block.")
                out_res = ops.make_residual_connection(input_to_prev_layer, out)
                input_to_prev_layer = input_to_next_layer
                input_to_next_layer = out_res

        if verbose:
            log.print3("Pathway ["+str(self.getStringType())+"], Mode: [" + mode + "], Output's Shape: " + str(input_to_next_layer.shape))
            #log.print3("Pathway [" + str(self.getStringType()) + "] done.")

        return input_to_next_layer

    def rec_field(self, rf_at_inp=[1,1,1], stride_rf_at_inp=[1,1,1]):
        # Returns: Rf of neurons at the output of the final layer of the block, with respect to input.
        #          Stride of rf at the block's output wrt input (how much it shifts at inp if we shift 1 neuron at out)
        rf_prev_block = rf_at_inp
        stride_rf_prev_block = stride_rf_at_inp
        for block in self._blocks:
            rf_prev_block, stride_rf_prev_block = block.rec_field(rf_prev_block, stride_rf_prev_block)
        return rf_prev_block, stride_rf_prev_block

    def calc_outp_dims_given_inp(self, inp_dims):
        outp_dims_prev_block = inp_dims
        for block in self._blocks:
            outp_dims_prev_block = block.calc_outp_dims_given_inp(outp_dims_prev_block)
        return outp_dims_prev_block

    def calc_inp_dims_given_outp(self, outp_dims):
        inp_dims_deeper_block = outp_dims
        for block in self._blocks:
            inp_dims_deeper_block = block.calc_inp_dims_given_outp(inp_dims_deeper_block)
        return inp_dims_deeper_block

    def build(self,
              log,
              rng,
              n_input_channels,
              num_kerns_per_layer,
              conv_kernel_dims_per_layer,
              conv_w_init_method,
              conv_pad_mode_per_layer,
              use_bn_per_layer, # As a flag for case that I want to apply BN on input image. I want to apply to input of FC.
              moving_avg_length,
              activ_func_per_layer,
              dropout_rate_per_layer=[],
              pool_prms_for_path = [],
              inds_of_lower_rank_convs=[],
              ranks_of_lower_rank_convs = [],
              inds_of_blocks_for_res_conns_at_out=[]
              ):
        log.print3("[Pathway_" + str(self.getStringType()) + "] is being built...")

        self._n_fms_in = n_input_channels
        self._inds_of_blocks_for_res_conns_at_out = inds_of_blocks_for_res_conns_at_out

        n_fms_input_to_prev_layer = None
        n_fms_input_to_next_layer = n_input_channels
        n_blocks = len(num_kerns_per_layer)
        for layer_i in range(0, n_blocks) :

            if layer_i in inds_of_lower_rank_convs :
                block = LowRankConvBlock(ranks_of_lower_rank_convs[ inds_of_lower_rank_convs.index(layer_i) ])
            else : # normal conv block
                block = ConvBlock()

            log.print3("\tBlock [" + str(layer_i) + "], FMs-In: " + str(n_fms_input_to_next_layer) +\
                                               ", FMs-Out: " + str(num_kerns_per_layer[layer_i]) +\
                                               ", Conv Filter dimensions: " + str(conv_kernel_dims_per_layer[layer_i]))
            block.build(rng,
                        n_fms_in=n_fms_input_to_next_layer,
                        n_fms_out=num_kerns_per_layer[layer_i],
                        conv_kernel_dims=conv_kernel_dims_per_layer[layer_i],
                        pool_prms=pool_prms_for_path[layer_i],
                        conv_w_init_method=conv_w_init_method,
                        conv_pad_mode=conv_pad_mode_per_layer[layer_i] if len(conv_pad_mode_per_layer) > 0 else None,
                        use_bn=use_bn_per_layer[layer_i],
                        moving_avg_length=moving_avg_length,
                        activ_func=activ_func_per_layer[layer_i],
                        dropout_rate=dropout_rate_per_layer[layer_i] if len(dropout_rate_per_layer) > 0 else 0
                        )
            self._blocks.append(block)

            n_fms_input_to_prev_layer = n_fms_input_to_next_layer
            n_fms_input_to_next_layer = num_kerns_per_layer[layer_i]

        self._n_fms_out = n_fms_input_to_next_layer


    # Getters
    def pName(self):
        return self._pName
    def pType(self):
        return self._pType
    def get_blocks(self):
        return self._blocks
    def get_block(self, index):
        return self._blocks[index]
    def subs_factor(self):
        return self._subs_factor

    # Other API :
    def getStringType(self) : raise NotImplementedMethod() # Abstract implementation. Children classes should implement this.


class NormalPathway(Pathway):
    def __init__(self, pName=None):
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.NORM
    # Override parent's abstract classes.
    def getStringType(self):
        return "NORMAL"


class SubsampledPathway(Pathway):
    def __init__(self, subsamplingFactor, pName=None):
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.SUBS
        self._subs_factor = subsamplingFactor

    def upsample_to_high_res(self, input, shape_to_match, upsampl_type="repeat"):
        # shape_to_match: list of dimensions x,y,z to match, eg by cropping after upsampling. [dimx, dimy, dimz]
        return ops.upsample_5D_tens_and_crop(input, self.subs_factor(), upsampl_type, shape_to_match)

    # OVERRIDING parent's classes.
    def getStringType(self):
        return "SUBSAMPLED" + str(self.subs_factor())

    def calc_inp_dims_given_outp_after_upsample(self, outp_dims_in_hr):
        outp_dims_in_lr = [int(ceil(int(outp_dims_in_hr[d])/ self.subs_factor()[d])) for d in range(3)]
        inp_dims_req = self.calc_inp_dims_given_outp(outp_dims_in_lr)
        return inp_dims_req

class FcPathway(Pathway):
    def __init__(self, pName=None):
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.FC
    # Override parent's abstract classes.
    def getStringType(self):
        return "FC"
