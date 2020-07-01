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
        self._subs_factor = [1, 1, 1]
        self._res_conn = self.config.res_conn

        # === Output of the block ===
        self._n_fms_out = self.config.n_FMs_per_layer[-1]

        self._activation_function = activation_function
        self._n_batches_for_bn_mov_avg = n_batches_for_bn_mov_avg

        self._rng = rng
        self._conv_w_init_type = conv_w_init_type
        self._blocks = self.__init_blocks()

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

        for idx, block in enumerate(self.get_blocks()):
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
        for block in self.get_blocks():
            rf_prev_block, stride_rf_prev_block = block.rec_field(rf_prev_block, stride_rf_prev_block)
        return rf_prev_block, stride_rf_prev_block

    def calc_outp_dims_given_inp(self, inp_dims):
        outp_dims_prev_block = inp_dims
        for block in self.get_blocks():
            outp_dims_prev_block = block.calc_outp_dims_given_inp(outp_dims_prev_block)
        return outp_dims_prev_block

    def calc_inp_dims_given_outp(self, outp_dims):
        inp_dims_deeper_block = outp_dims
        for block in self.get_blocks():
            inp_dims_deeper_block = block.calc_inp_dims_given_outp(inp_dims_deeper_block)
        return inp_dims_deeper_block

    def __init_blocks(self) -> List[Union[ConvBlock, LowRankConvBlock]]:
        blocks = []
        n_fms_input_to_next_layer = self._n_fms_in
        n_blocks = len(self.config.n_FMs_per_layer)
        use_bn = self._n_batches_for_bn_mov_avg > 0
        for layer_i in range(0, n_blocks):
            pad_mode = self.config.pad_mode_per_layer[layer_i] if len(self.config.pad_mode_per_layer) > 0 else None
            # To not apply activation on raw input. -1 is linear activation.
            if layer_i == 0 and self.pType() != PathwayTypes.FC:
                activation_function = "linear"
            else:
                activation_function = self._activation_function
            # For the 1st layer, ask specific flag.
            if layer_i == 0 and use_bn:
                use_bn = self.config.apply_bn
            if layer_i in self.config.lower_rank:
                block = LowRankConvBlock(
                    n_fms_in=n_fms_input_to_next_layer,
                    n_fms_out=self.config.n_FMs_per_layer[layer_i],
                    conv_kernel_dims=self.config.kernel_dims_per_layer[layer_i],
                    pool_prms=self.config.mp_params[layer_i],
                    conv_pad_mode=pad_mode,
                    use_bn=use_bn,
                    moving_avg_length=self._n_batches_for_bn_mov_avg,
                    activ_func=activation_function,
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
                    use_bn=use_bn,
                    moving_avg_length=self._n_batches_for_bn_mov_avg,
                    activ_func=activation_function,
                    dropout_rate=self.config.dropout[layer_i] if len(self.config.dropout) > 0 else 0,
                    rng=self._rng,
                    conv_w_init_type=self._conv_w_init_type,
                )
            blocks.append(block)
            n_fms_input_to_next_layer = self.config.n_FMs_per_layer[layer_i]
        return blocks

    def build(self, log: Logger = None):
        if log is None:
            log = self.log
        log.print3("[Pathway_{ptype}] is being built...".format(ptype=(self.get_str_type())))
        n_fms_input_to_next_layer = self._n_fms_in
        for layer_i, block in enumerate(self.get_blocks()):
            log.print3(
                "\tBlock [{idx}], FMs-In: {fm_in}, FMs_Out: {fm_out}, Conv Filter dimensions: {kern_dim}".format(
                    idx=layer_i,
                    fm_in=str(n_fms_input_to_next_layer),
                    fm_out=str(self.config.n_FMs_per_layer[layer_i]),
                    kern_dim=str(self.config.kernel_dims_per_layer[layer_i]),
                )
            )
            block.build()
            n_fms_input_to_next_layer = self.config.n_FMs_per_layer[layer_i]

    # Getters
    def pName(self):
        return None

    def pType(self):
        return self._pType

    def get_blocks(self) -> List[Union[ConvBlock, LowRankConvBlock]]:
        return self._blocks

    def get_block(self, index) -> Union[ConvBlock, LowRankConvBlock]:
        return self._blocks[index]

    def append_block(self, block: Union[ConvBlock, LowRankConvBlock]):
        self._blocks.append(block)

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
        self._normal_pathway = NormalPathway(
            n_input_channels=n_input_channels,
            activation_function=activation_function,
            n_batches_for_bn_mov_avg=n_batches_for_bn_mov_avg,
            config=normal_pathway_config,
            log=log,
            rng=rng,
            conv_w_init_type=conv_w_init_type,
        )
        subsampled_pathways = []
        if subsampled_pathway_configs is not None:
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
                subsampled_pathways.append(pathway)
        self._subsampled_pathways = subsampled_pathways
        # ======================= Make the Fully Connected Layers =======================
        n_fms_inp_to_fc_path = self._normal_pathway.get_n_fms_out()
        for pathway in subsampled_pathways:
            n_fms_inp_to_fc_path += pathway.get_n_fms_out()
        fc_pathway_config.n_FMs_per_layer += [n_classes]
        self._fc_pathway = FcPathway(
            n_input_channels=n_fms_inp_to_fc_path,
            activation_function=activation_function,
            n_batches_for_bn_mov_avg=n_batches_for_bn_mov_avg,
            config=fc_pathway_config,
            log=log,
            rng=rng,
            conv_w_init_type=conv_w_init_type,
        )

    def build(self):
        for pathway in self:
            pathway.build()

    def to_list(self) -> List[Union[NormalPathway, SubsampledPathway, FcPathway]]:
        pathways = [self._normal_pathway]
        pathways.extend(self._subsampled_pathways)
        pathways.append(self._fc_pathway)
        return pathways

    def __iter__(self):
        return PathwayIterator(self.to_list())

    def __len__(self):
        return len(self.to_list())

    def __getitem__(self, idx: int):
        return self.to_list()[idx]

    def normal_pathway(self) -> NormalPathway:
        return self._normal_pathway

    def fc_pathway(self) -> FcPathway:
        return self._fc_pathway

    def subsampled_pathways(self) -> List[SubsampledPathway]:
        return self._subsampled_pathways

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

    def calc_inp_dims_of_paths_from_hr_inp(self, inp_hr_dims):
        out_shape_of_hr_path = self._normal_pathway.calc_outp_dims_given_inp(inp_hr_dims)
        inp_shape_per_path = [inp_hr_dims]
        for subpath in self.subsampled_pathways():
            inp_shape_lr = subpath.calc_inp_dims_given_outp_after_upsample(out_shape_of_hr_path)
            inp_shape_per_path.append(inp_shape_lr)
        inp_shape_per_path.append(out_shape_of_hr_path)
        # [ [path0-in-dim-x, path0-in-dim-y, path0-in-dim-z],
        #   [path1-in-dim-x, path1-in-dim-y, path1-in-dim-z],
        #    ...
        #   [pathFc-in-dim-x, pathFc-in-dim-y, pathFc-in-dim-z] ]
        return inp_shape_per_path