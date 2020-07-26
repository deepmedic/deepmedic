from typing import List

from deepmedic.logging import loggers
from deepmedic.config import BaseConfig
from deepmedic.config.utils import calc_rec_field_of_path_assuming_strides_1


class ConvBlockConfig(BaseConfig):
    __slots__ = [
        "n_fm_in",
        "n_fm_out",
        "kernel_dims",
        "pool_params",
        "pad_mode",
        "apply_bn",
        "bn_moving_avg_length",
        "activation_function",
        "dropout",
        "w_init_type",
    ]

    def __init__(
        self,
        n_fm_in: int,
        n_fm_out: int,
        kernel_dims: List[int],
        apply_bn: bool,
        pad_mode: str = None,
        pool_params: List = None,
        dropout: float = None,
        bn_moving_avg_length: int = None,
        activation_function: str = None,
        w_init_type: List = None
    ):
        self.n_fm_in = n_fm_in
        self.n_fm_out = n_fm_out
        self.kernel_dims = kernel_dims
        self.apply_bn = apply_bn

        self.activation_function = self._get_str(activation_function, default="prelu")
        self.pool_params = self._get_list(pool_params, default=[])
        self.dropout = self._get_float(dropout, default=None)
        self.bn_moving_avg_length = self._get_int(bn_moving_avg_length, default=60)
        self.pad_mode = self._get_str(pad_mode, default="VALID")
        self.w_init_type = self._get_list(w_init_type, default=["fanIn", 2])


class LowerRankConvBlockConfig(ConvBlockConfig):
    __slots__ = [
        "n_fm_in",
        "n_fm_out",
        "kernel_dims",
        "pool_params",
        "pad_mode",
        "apply_bn",
        "bn_moving_avg_length",
        "activation_function",
        "dropout",
        "w_init_type",
        "rank",
    ]

    def __init__(
        self,
        n_fm_in: int,
        n_fm_out: int,
        kernel_dims: List[int],
        apply_bn: bool,
        pad_mode: str = None,
        pool_params: List = None,
        dropout: float = None,
        bn_moving_avg_length: int = None,
        activation_function: str = None,
        w_init_type: List = None,
        rank: int = None
    ):
        super().__init__(
            n_fm_in=n_fm_in,
            n_fm_out=n_fm_out,
            kernel_dims=kernel_dims,
            apply_bn=apply_bn,
            pad_mode=pad_mode,
            pool_params=pool_params,
            dropout=dropout,
            bn_moving_avg_length=bn_moving_avg_length,
            activation_function=activation_function,
            w_init_type=w_init_type,
        )
        self.rank = self._get_int(rank, default=2)


class PathWayConfig(BaseConfig):
    __slots__ = [
        "n_FMs_per_layer",
        "kernel_dims_per_layer",
        "pad_mode_per_layer",
        "dropout",
        "apply_bn",
        "mp_params",
        "res_conn",
        "lower_rank",
        "rank_of_lower_rank",
    ]

    def __init__(
        self,
        n_FMs_per_layer: List[int],
        kernel_dims_per_layer: List[List[int]],
        pad_mode_per_layer: List[str] = None,
        dropout: List[float] = None,
        apply_bn: bool = None,
        mp_params: List = None,
        res_conn: List[int] = None,
        lower_rank: List[int] = None,
    ):
        self.n_FMs_per_layer = n_FMs_per_layer
        self.kernel_dims_per_layer = kernel_dims_per_layer
        self.pad_mode_per_layer = self._get_list_of_str(
            pad_mode_per_layer, default=["VALID"] * len(self.n_FMs_per_layer)
        )
        self.dropout = self._get_list_of_float(dropout, default=self.default_dropout())
        self.apply_bn = self._get_bool(apply_bn, default=False)
        self.mp_params = self._get_list(mp_params, default=[[] for _ in range(len(self.n_FMs_per_layer))])
        self.res_conn = self._get_list_of_int(res_conn, default=[])
        self.lower_rank = self._get_list_of_int(lower_rank, default=[])
        self.rank_of_lower_rank = 2

    def default_dropout(self):
        return []


class SubsampledPathwayConfig(PathWayConfig):
    __slots__ = [
        "n_FMs_per_layer",
        "kernel_dims_per_layer",
        "pad_mode_per_layer",
        "dropout",
        "apply_bn",
        "mp_params",
        "res_conn",
        "lower_rank",
        "rank_of_lower_rank",
        "subsample_factor",
    ]

    def __init__(
        self,
        n_FMs_per_layer: List[int],
        kernel_dims_per_layer: List[List[int]],
        pad_mode_per_layer: List[str] = None,
        subsample_factor: List[int] = None,
        dropout: List[float] = None,
        apply_bn: bool = None,
        mp_params: List = None,
        res_conn: List[int] = None,
        lower_rank: List[int] = None,
    ):
        super().__init__(
            n_FMs_per_layer,
            kernel_dims_per_layer,
            pad_mode_per_layer,
            dropout,
            apply_bn,
            mp_params,
            res_conn,
            lower_rank,
        )
        self.subsample_factor = self._get_list(subsample_factor, default=[3, 3, 3])


class FCLayersConfig(PathWayConfig):
    __slots__ = [
        "n_FMs_per_layer",
        "kernel_dims_per_layer",
        "pad_mode_per_layer",
        "dropout",
        "apply_bn",
        "mp_params",
        "res_conn",
        "lower_rank",
        "rank_of_lower_rank",
        "softmax_temperature"
    ]

    def __init__(
        self,
        n_FMs_per_layer: List[int] = None,
        kernel_dims_per_layer: List[List[int]] = None,
        pad_mode_per_layer: List[str] = None,
        softmax_temperature: float = None,
        dropout: List[float] = None,
        apply_bn: bool = None,
        mp_params: List = None,
        res_conn: List[int] = None,
        lower_rank: List[int] = None,
    ):
        n_FMs_per_layer = self._get_list_of_int(n_FMs_per_layer, default=[])
        n_layers_fc = len(n_FMs_per_layer) + 1
        kernel_dims_per_layer = self._get_list_of_list_int(
            kernel_dims_per_layer, default=[[1, 1, 1] for _ in range(n_layers_fc)]
        )
        pad_mode_per_layer = self._get_list_of_str(pad_mode_per_layer, default=["VALID"] * n_layers_fc)
        super().__init__(
            n_FMs_per_layer,
            kernel_dims_per_layer,
            pad_mode_per_layer,
            dropout,
            apply_bn,
            mp_params,
            res_conn,
            lower_rank,
        )
        self.mp_params = self._get_list(mp_params, default=[[] for _ in range(len(n_FMs_per_layer) + 1)])
        self.apply_bn = self._get_bool(apply_bn, default=True)
        self.softmax_temperature = self._get_float(softmax_temperature, default=1.0)
        self.rank_of_lower_rank = None

    def default_dropout(self) -> List[float]:
        # n_fms_in_extra_fcs: List of integers, 1 per layer in the final classification path, except final classif layer
        n_extra_fcs = len(self.n_FMs_per_layer)
        if n_extra_fcs > 0:
            dropout_for_each_l_including_classifier = [0.0] + [0.5] * (n_extra_fcs - 1) + [0.5]
        else:
            dropout_for_each_l_including_classifier = [0.5]
        return dropout_for_each_l_including_classifier


class ModelConfig(BaseConfig):
    __slots__ = [
        "model_name",
        "n_classes",
        "n_input_channels",
        "normal_pathway_config",
        "subsampled_pathway_configs",
        "fc_layers_config",
        "activation_function",
        "conv_w_init_type",
        "n_batches_for_bn_mov_avg",
        "segment_dim_train",
        "segment_dim_val",
        "segment_dim_inference",
    ]

    def __init__(
        self,
        n_classes: int,
        n_input_channels: int,
        normal_pathway_config: PathWayConfig,
        subsampled_pathway_configs: List[SubsampledPathwayConfig],
        fc_layers_config: FCLayersConfig,
        segment_dim_train: List[int],
        model_name: str = None,
        activation_function: str = None,
        conv_w_init_type: List = None,
        n_batches_for_bn_mov_avg: int = None,
        segment_dim_val: List[int] = None,
        segment_dim_inference: List[int] = None,
    ):
        self.model_name = self._get_str(model_name, default="deepmedic")
        self.n_classes = n_classes
        self.n_input_channels = n_input_channels
        self.normal_pathway_config = normal_pathway_config
        self.subsampled_pathway_configs = subsampled_pathway_configs
        self.fc_layers_config = fc_layers_config
        self.activation_function = self._get_str(activation_function, default="prelu")
        # Initialization
        self.conv_w_init_type = self._get_list(conv_w_init_type, default=["fanIn", 2])
        # Batch Normalization
        self.n_batches_for_bn_mov_avg = self._get_int(n_batches_for_bn_mov_avg, default=60)
        self.segment_dim_train = segment_dim_train
        self.segment_dim_val = self._get_list_of_int(
            segment_dim_val,
            default=calc_rec_field_of_path_assuming_strides_1(self.normal_pathway_config.kernel_dims_per_layer),
        )
        self.segment_dim_inference = self._get_list_of_int(segment_dim_inference, self.segment_dim_train)

    def print_params(self, logger: loggers.Logger = None):
        if logger is None:
            logger = loggers.Logger()

        logger_print = logger.print3
        logger_print("=============================================================")
        logger_print("========== PARAMETERS FOR MAKING THE ARCHITECTURE ===========")
        logger_print("=============================================================")
        logger_print("CNN model's name = " + str(self.model_name))

        logger_print("~~~~~~~~~~~~~~~~~~Model parameters~~~~~~~~~~~~~~~~")
        logger_print("Number of Classes (including background) = " + str(self.n_classes))
        logger_print("~~Normal Pathway~~")
        logger_print("Number of Input Channels = " + str(self.n_input_channels))
        logger_print("Number of Layers = " + str(len(self.normal_pathway_config.n_FMs_per_layer)))
        logger_print("Number of Feature Maps per layer = " + str(self.normal_pathway_config.n_FMs_per_layer))
        logger_print("Kernel Dimensions per layer = " + str(self.normal_pathway_config.kernel_dims_per_layer))
        logger_print("Padding mode of convs per layer = " + str(self.normal_pathway_config.pad_mode_per_layer))
        logger_print(
            "Residual connections added at the output of layers (indices from 0) = "
            + str(self.normal_pathway_config.res_conn)
        )
        logger_print(
            "Layers that will be made of Lower Rank (indices from 0) = " + str(self.normal_pathway_config.lower_rank)
        )
        logger_print("Lower Rank layers will be made of rank = " + str(self.normal_pathway_config.rank_of_lower_rank))

        logger_print("~~Subsampled Pathway~~")
        use_subsampled_path = self.subsampled_pathway_configs is not None
        logger_print("Use subsampled Pathway = " + str(use_subsampled_path))
        logger_print("Number of subsampled pathways that will be built = " + str(len(self.subsampled_pathway_configs)))
        logger_print(
            "Number of Layers (per sub-pathway) = "
            + str([len(config.n_FMs_per_layer) for config in self.subsampled_pathway_configs])
        )
        logger_print(
            "Number of Feature Maps per layer (per sub-pathway) = "
            + str([config.n_FMs_per_layer for config in self.subsampled_pathway_configs])
        )
        logger_print("Kernel Dimensions per layer = " + str(self.subsampled_pathway_configs[0].kernel_dims_per_layer))
        logger_print("Padding mode of convs per layer = " + str(self.subsampled_pathway_configs[0].pad_mode_per_layer))
        logger_print(
            "Subsampling Factor per dimension (per sub-pathway) = "
            + str([config.subsample_factor for config in self.subsampled_pathway_configs])
        )
        logger_print(
            "Residual connections added at the output of layers (indices from 0) = "
            + str(self.subsampled_pathway_configs[0].res_conn)
        )
        logger_print(
            "Layers that will be made of Lower Rank (indices from 0) = "
            + str(self.subsampled_pathway_configs[0].lower_rank)
        )
        logger_print(
            "Lower Rank layers will be made of rank = " + str(self.subsampled_pathway_configs[0].rank_of_lower_rank)
        )

        logger_print("~~Fully Connected Pathway~~")
        logger_print(
            "Number of additional FC layers (Excluding the Classif. Layer) = "
            + str(len(self.fc_layers_config.n_FMs_per_layer))
        )
        logger_print(
            "Number of Feature Maps in the additional FC layers = " + str(self.fc_layers_config.n_FMs_per_layer)
        )
        logger_print("Padding mode of convs per layer = " + str(self.fc_layers_config.pad_mode_per_layer))
        logger_print(
            "Residual connections added at the output of layers (indices from 0) = "
            + str(self.fc_layers_config.res_conn)
        )
        logger_print(
            "Layers that will be made of Lower Rank (indices from 0) = " + str(self.fc_layers_config.lower_rank)
        )
        logger_print(
            "Dimensions of Kernels in final FC path before classification = "
            + str(self.fc_layers_config.kernel_dims_per_layer)
        )

        logger_print("~~Size Of Image Segments~~")
        logger_print("Size of Segments for Training = " + str(self.segment_dim_train))
        logger_print("Size of Segments for Validation = " + str(self.segment_dim_val))
        logger_print("Size of Segments for Testing = " + str(self.segment_dim_inference))

        logger_print("~~Dropout Rates~~")
        logger_print("Drop.R. for each layer in Normal Pathway = " + str(self.normal_pathway_config.dropout))
        logger_print(
            "Drop.R. for each layer in Subsampled Pathway = " + str(self.subsampled_pathway_configs[0].dropout)
        )
        logger_print(
            "Drop.R. for each layer in FC Pathway (additional FC layers + Classific.Layer at end) = "
            + str(self.fc_layers_config.dropout)
        )

        logger_print("~~Weight Initialization~~")
        logger_print("Initialization method and params for the conv kernel weights = " + str(self.conv_w_init_type))

        logger_print("~~Activation Function~~")
        logger_print("Activation function to use = " + str(self.activation_function))

        logger_print("~~Batch Normalization~~")
        logger_print(
            "Apply BN straight on pathways' inputs (eg straight on segments) = "
            + str(
                [
                    self.normal_pathway_config.apply_bn,
                    self.subsampled_pathway_configs[0].apply_bn,
                    self.fc_layers_config.apply_bn,
                ]
            )
        )
        logger_print(
            "Batch Normalization uses a rolling average for inference, over this many batches = "
            + str(self.n_batches_for_bn_mov_avg)
        )

        logger_print("========== Done with printing session's parameters ==========")
        logger_print("=============================================================")
