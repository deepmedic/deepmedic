# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
from math import ceil
from typing import Union, List

from deepmedic.neuralnet.pathwayTypes import PathwayTypes
from deepmedic.neuralnet.blocks import ConvBlock, LowRankConvBlock
import deepmedic.neuralnet.ops as ops
from deepmedic.config.model import PathWayConfig, SubsampledPathwayConfig, FCLayersConfig, ModelConfig
from deepmedic.logging.loggers import Logger

#################################################################
#                        Classes of Pathways                    #
#################################################################


class Pathway:
    _pType = None

    def __init__(
        self,
        n_input_channels: int,
        activation_function: str,
        n_batches_for_bn_mov_avg: int,
        config: Union[PathWayConfig, SubsampledPathwayConfig, FCLayersConfig],
        rng: np.random.RandomState,
        conv_w_init_type: List,
        log: Logger = None,
    ):
        if log is None:
            log = Logger()
        self.log = log
        self.config = config

        # === Input to the pathway ===
        self._n_fms_in = n_input_channels
        # === Basic architecture parameters ===
        self._blocks = []
        self._subs_factor = [1, 1, 1]
        self._res_conn = self.config.res_conn

        # === Output of the block ===
        self._n_fms_out = self.config.n_FMs_per_layer[-1]

        self._activ_func_per_layer = [activation_function] * len(self.config.n_FMs_per_layer)
        # To not apply activation on raw input. -1 is linear activation.
        self._activ_func_per_layer[0] = "linear" if self.pType() != PathwayTypes.FC else activation_function

        self._use_bn_per_layer = [n_batches_for_bn_mov_avg > 0] * len(self.config.n_FMs_per_layer)
        # For the 1st layer, ask specific flag.
        self._use_bn_per_layer[0] = self.config.apply_bn if n_batches_for_bn_mov_avg > 0 else False
        self._n_batches_for_bn_mov_avg = n_batches_for_bn_mov_avg

        self._rng = rng
        self._conv_w_init_type = conv_w_init_type

    # Getters
    def get_n_fms_in(self):
        return self._n_fms_in

    def get_n_fms_out(self):
        return self._n_fms_out

    def apply(self, input, mode, train_val_test, verbose=False, log=None):
        """
            mode: 'train' / 'infer'
            train_val_test: TEMPORARY. ONLY TO RETURN FMS. REMOVE IN END OF REFACTORING.
        """

        if log is None:
            log = self.log
        if verbose:
            log.print3(
                "Pathway [{ptype}], Mode: [{mode}], Input's Shape: {shape}".format(
                    ptype=str(self.get_str_type()), mode=mode, shape=str(input.shape)
                )
            )

        input_to_prev_layer = None
        input_to_next_layer = input

        for idx, block in enumerate(self._blocks):
            if verbose:
                log.print3(
                    "\tBlock [{idx}], Mode: [{mode}], Input's Shape: {shape}".format(
                        idx=idx, mode=mode, shape=str(input.shape)
                    )
                )
            out = block.apply(input_to_next_layer, mode)
            block.output[train_val_test] = out  # HACK TEMPORARY. ONLY USED FOR RETURNING FMS.
            if idx not in self._res_conn:  # not a residual connecting here
                input_to_prev_layer = input_to_next_layer
                input_to_next_layer = out
            else:  # make residual connection
                # The very first block (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                assert idx > 0
                log.print3(
                    "\tResidual-Connection made between: output of [Block_{idx}] & input of previous block.".format(
                        idx=idx
                    )
                )
                out_res = ops.make_residual_connection(input_to_prev_layer, out)
                input_to_prev_layer = input_to_next_layer
                input_to_next_layer = out_res

        if verbose:
            log.print3(
                "Pathway [{ptype}], Mode: [{mode}], Output's Shape: {shape}".format(
                    ptype=str(self.get_str_type()), mode=mode, shape=str(input_to_next_layer.shape)
                )
            )
            # log.print3("Pathway [" + str(self.getStringType()) + "] done.")

        return input_to_next_layer

    def rec_field(self, rf_at_inp: List = None, stride_rf_at_inp: List = None):
        """
            Returns: Rf of neurons at the output of the final layer of the block, with respect to input.
                     Stride of rf at the block's output wrt input
                     (how much it shifts at inp if we shift 1 neuron at out)
        """

        if rf_at_inp is None:
            rf_at_inp = [1, 1, 1]
        if stride_rf_at_inp is None:
            stride_rf_at_inp = [1, 1, 1]
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

    def build(self, log: Logger = None):
        if log is None:
            log = self.log

        log.print3("[Pathway_{ptype}] is being built...".format(ptype=(self.get_str_type())))
        n_fms_input_to_next_layer = self._n_fms_in
        n_blocks = len(self.config.n_FMs_per_layer)
        for layer_i in range(0, n_blocks):
            pad_mode = self.config.pad_mode_per_layer[layer_i] if len(self.config.pad_mode_per_layer) > 0 else None

            if layer_i in self.config.lower_rank:
                block = LowRankConvBlock(
                    n_fms_in=n_fms_input_to_next_layer,
                    n_fms_out=self.config.n_FMs_per_layer[layer_i],
                    conv_kernel_dims=self.config.kernel_dims_per_layer[layer_i],
                    pool_prms=self.config.mp_params[layer_i],
                    conv_pad_mode=pad_mode,
                    use_bn=self._use_bn_per_layer[layer_i],
                    moving_avg_length=self._n_batches_for_bn_mov_avg,
                    activ_func=self._activ_func_per_layer[layer_i],
                    dropout_rate=self.config.dropout[layer_i] if len(self.config.dropout) > 0 else 0,
                    rank=self.config.rank_of_lower_rank,
                    rng=self._rng,
                    conv_w_init_type=self._conv_w_init_type,
                )
            else:  # normal conv block
                block = ConvBlock(
                    n_fms_in=n_fms_input_to_next_layer,
                    n_fms_out=self.config.n_FMs_per_layer[layer_i],
                    conv_kernel_dims=self.config.kernel_dims_per_layer[layer_i],
                    pool_prms=self.config.mp_params[layer_i],
                    conv_pad_mode=pad_mode,
                    use_bn=self._use_bn_per_layer[layer_i],
                    moving_avg_length=self._n_batches_for_bn_mov_avg,
                    activ_func=self._activ_func_per_layer[layer_i],
                    dropout_rate=self.config.dropout[layer_i] if len(self.config.dropout) > 0 else 0,
                    rng=self._rng,
                    conv_w_init_type=self._conv_w_init_type,
                )

            log.print3(
                "\tBlock [{idx}], FMs-In: {fm_in}, FMs_Out: {fm_out}, Conv Filter dimensions: {kern_dim}".format(
                    idx=layer_i,
                    fm_in=str(n_fms_input_to_next_layer),
                    fm_out=str(self.config.n_FMs_per_layer[layer_i]),
                    kern_dim=str(self.config.kernel_dims_per_layer[layer_i]),
                )
            )
            block.build()
            self._blocks.append(block)
            n_fms_input_to_next_layer = self.config.n_FMs_per_layer[layer_i]

    # Getters
    def pName(self):
        return None

    def pType(self):
        return self._pType

    def get_blocks(self):
        return self._blocks

    def get_block(self, index):
        return self._blocks[index]

    def subs_factor(self):
        return self._subs_factor

    def get_str_type(self):
        raise NotImplementedError("Abstract implementation. Children classes should implement this.")


class NormalPathway(Pathway):
    _pType = PathwayTypes.NORM

    def __init__(
        self,
        n_input_channels: int,
        activation_function: str,
        n_batches_for_bn_mov_avg: int,
        config: PathWayConfig,
        rng: np.random.RandomState,
        conv_w_init_type: List,
        log: Logger = None,
    ):
        super().__init__(
            n_input_channels=n_input_channels,
            activation_function=activation_function,
            n_batches_for_bn_mov_avg=n_batches_for_bn_mov_avg,
            config=config,
            log=log,
            rng=rng,
            conv_w_init_type=conv_w_init_type,
        )

    # Override parent's abstract classes.
    def get_str_type(self):
        return "NORMAL"


class SubsampledPathway(Pathway):
    _pType = PathwayTypes.SUBS

    def __init__(
        self,
        n_input_channels: int,
        activation_function: str,
        n_batches_for_bn_mov_avg: int,
        config: SubsampledPathwayConfig,
        rng: np.random.RandomState,
        conv_w_init_type: List,
        log: Logger = None,
    ):
        super().__init__(
            n_input_channels=n_input_channels,
            activation_function=activation_function,
            n_batches_for_bn_mov_avg=n_batches_for_bn_mov_avg,
            config=config,
            log=log,
            rng=rng,
            conv_w_init_type=conv_w_init_type,
        )
        self._subs_factor = self.config.subsample_factor

    def upsample_to_high_res(self, input, shape_to_match, upsampl_type="repeat"):
        """
            shape_to_match: list of dimensions x,y,z to match, eg by cropping after upsampling. [dimx, dimy, dimz]
        """
        return ops.upsample_5D_tens_and_crop(input, self.subs_factor(), upsampl_type, shape_to_match)

    def get_str_type(self):
        # OVERRIDING parent's classes.
        return "SUBSAMPLED" + str(self.subs_factor())

    def calc_inp_dims_given_outp_after_upsample(self, outp_dims_in_hr):
        outp_dims_in_lr = [int(ceil(int(outp_dims_in_hr[d]) / self.subs_factor()[d])) for d in range(3)]
        inp_dims_req = self.calc_inp_dims_given_outp(outp_dims_in_lr)
        return inp_dims_req


class FcPathway(Pathway):
    _pType = PathwayTypes.FC

    def __init__(
        self,
        n_input_channels: int,
        activation_function: str,
        n_batches_for_bn_mov_avg: int,
        config: FCLayersConfig,
        rng: np.random.RandomState,
        conv_w_init_type: List,
        log: Logger = None,
    ):
        super().__init__(
            n_input_channels=n_input_channels,
            activation_function=activation_function,
            n_batches_for_bn_mov_avg=n_batches_for_bn_mov_avg,
            config=config,
            log=log,
            rng=rng,
            conv_w_init_type=conv_w_init_type,
        )

    def get_str_type(self):
        return "FC"


class PathwayIterator:
    def __init__(self, pathways: List[Union[NormalPathway, SubsampledPathway, FcPathway]]):
        self._pathways = iter(pathways)

    def __next__(self) -> Union[NormalPathway, SubsampledPathway, FcPathway]:
        return next(self._pathways)


class Pathways:
    def __init__(
        self,
        n_input_channels: int,
        n_classes: int,
        activation_function: str,
        n_batches_for_bn_mov_avg: int,
        normal_pathway_config: PathWayConfig,
        subsampled_pathway_configs: List[SubsampledPathwayConfig],
        fc_pathway_config: FCLayersConfig,
        conv_w_init_type: List,
        rng: np.random.RandomState,
        log: Logger = None
    ):
        pathways = []
        pathway = NormalPathway(
            n_input_channels=n_input_channels,
            activation_function=activation_function,
            n_batches_for_bn_mov_avg=n_batches_for_bn_mov_avg,
            config=normal_pathway_config,
            log=log,
            rng=rng,
            conv_w_init_type=conv_w_init_type,
        )
        pathways.append(pathway)
        for subpath_config in subsampled_pathway_configs:
            pathway = SubsampledPathway(
                n_input_channels=n_input_channels,
                activation_function=activation_function,
                n_batches_for_bn_mov_avg=n_batches_for_bn_mov_avg,
                config=subpath_config,
                log=log,
                rng=rng,
                conv_w_init_type=conv_w_init_type,
            )
            # There will be at least an entry as a secondary pathway.
            # But it won't have any layers if it was not actually used.
            pathways.append(pathway)
        n_fms_inp_to_fc_path = 0
        for path_i in range(len(pathways)):
            n_fms_inp_to_fc_path += pathways[path_i].get_n_fms_out()

        # ======================= Make the Fully Connected Layers =======================
        fc_pathway_config.n_FMs_per_layer += [n_classes]
        pathway = FcPathway(
            n_input_channels=n_fms_inp_to_fc_path,
            activation_function=activation_function,
            n_batches_for_bn_mov_avg=n_batches_for_bn_mov_avg,
            config=fc_pathway_config,
            log=log,
            rng=rng,
            conv_w_init_type=conv_w_init_type,
        )
        pathways.append(pathway)
        self._pathways = pathways

    def build(self):
        for pathway in self:
            pathway.build()

    def __iter__(self):
        return PathwayIterator(self._pathways)

    def append(self, item: Pathway):
        self._pathways.append(item)

    def __len__(self):
        return len(self._pathways)

    def __getitem__(self, idx: int):
        return self._pathways[idx]

    def get_fc_pathway(self) -> Union[FcPathway, None]:
        for pathway in self:
            if pathway.pType() == PathwayTypes.FC:
                return pathway
        return None

    def get_num_subs_pathways(self) -> int:
        count = 0
        for pathway in self:
            if pathway.pType() == PathwayTypes.SUBS:
                count += 1
        return count

    def get_num_pathways_that_require_input(self) -> int:
        count = 0
        for pathway in self:
            if pathway.pType() != PathwayTypes.FC:
                count += 1
        return count

    def update_arrays_of_bn_moving_avg(self, sessionTf):
        """
        for inference with batch-normalization.
        Every training batch, this is called to update an internal matrix of each layer, with the last mus and vars,
        so that I can compute the rolling average for inference.
        """

        for pathway in self:
            for block in pathway.get_blocks():
                block.update_arrays_of_bn_moving_avg(sessionTf)  # Will do nothing if no BN.
