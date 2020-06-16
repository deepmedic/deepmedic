from typing import List

from deepmedic.config.utils import calc_rec_field_of_path_assuming_strides_1


class PathWayConfig:
    def __init__(
        self,
        n_FMs_per_layer: List[int],
        kernel_dims_per_layer: List[List[int]],
        pad_mode_per_layer: List[str] = None,
        dropout: float = None,
        apply_bn: bool = None,
        mp_params: List = None,
        res_conn: List[int] = None,
        lower_rank: List[int] = None,
    ):
        self.n_FMs_per_layer = n_FMs_per_layer
        self.kernel_dims_per_layer = kernel_dims_per_layer
        if pad_mode_per_layer is None:
            pad_mode_per_layer = ["VALID"] * len(self.n_FMs_per_layer)
        self.pad_mode_per_layer = pad_mode_per_layer
        if dropout is None:
            dropout = self.default_dropout()
        self.dropout = dropout
        self.apply_bn = apply_bn if apply_bn is not None else False
        if mp_params is None:
            mp_params = [[] for _ in range(len(self.n_FMs_per_layer))]
        self.mp_params = mp_params
        if res_conn is None:
            res_conn = []
        self.res_conn = res_conn
        if lower_rank is None:
            lower_rank = []
        self.lower_rank = lower_rank
        self.rank_of_lower_rank = 2

    def default_dropout(self):
        return []


class SubsampledPathwayConfig(PathWayConfig):
    def __init__(
        self,
        n_FMs_per_layer: List[int],
        kernel_dims_per_layer: List[List[int]],
        pad_mode_per_layer: List[str] = None,
        subsample_factors: List[List[int]] = None,
        dropout: float = None,
        apply_bn: bool = None,
        mp_params: List = None,
        res_conn: List[int] = None,
        lower_rank: List[int] = None,
        use_subsampled_path: bool = None,
    ):
        # TODO: is n_fms here list of list of in or list of int?
        if use_subsampled_path is None:
            use_subsampled_path = False
        self.use_subsampled_path = use_subsampled_path
        if not self.use_subsampled_path:
            res_conn = []
            lower_rank = []
        apply_bn = apply_bn if apply_bn is not None else False
        pad_mode_per_layer = (
            pad_mode_per_layer if pad_mode_per_layer is not None else ["VALID"] * len(n_FMs_per_layer[0])
        )
        if subsample_factors is not None:
            if all(not isinstance(e, list) for e in subsample_factors):
                subsample_factors = [subsample_factors]
        else:
            subsample_factors = [[3, 3, 3]]
        self.subsample_factors = subsample_factors
        for _ in range(len(self.subsample_factors) - len(n_FMs_per_layer)):
            n_fms_per_l_in_prev_path = n_FMs_per_layer[-1]
            n_FMs_per_layer.append([max(1, int(n_fms_in_l_i)) for n_fms_in_l_i in n_fms_per_l_in_prev_path])
        if mp_params is None:
            mp_params = (
                [[] for _ in range(len(self.n_FMs_per_layer[0]))]
                if self.use_subsampled_path
                else []
            )
        super().__init__(n_FMs_per_layer, kernel_dims_per_layer, pad_mode_per_layer, dropout, apply_bn, mp_params, res_conn, lower_rank)


class FCLayersConfig(PathWayConfig):
    def __init__(
        self,
        n_FMs_per_layer: List[List[int]] = None,
        kernel_dims_per_layer: List[List[int]] = None,
        pad_mode_per_layer: List[str] = None,
        softmax_temperature: float = None,
        dropout: float = None,
        apply_bn: bool = None,
        mp_params: List = None,
        res_conn: List[int] = None,
        lower_rank: List[int] = None,
    ):
        if n_FMs_per_layer is None:
            n_FMs_per_layer = []
        n_layers_fc = len(n_FMs_per_layer) + 1
        if kernel_dims_per_layer is None:
            kernel_dims_per_layer = [[1, 1, 1] for _ in range(n_layers_fc)]
        if pad_mode_per_layer is None:
            pad_mode_per_layer = ["VALID"] * n_layers_fc
        if mp_params is None:
            mp_params = [[] for _ in range(len(self.n_FMs_per_layer) + 1)]
        apply_bn = apply_bn if apply_bn is not None else True
        if res_conn is None:
            res_conn = []
        if lower_rank is None:
            lower_rank = []
        super().__init__(n_FMs_per_layer, kernel_dims_per_layer, pad_mode_per_layer, dropout, apply_bn, mp_params, res_conn, lower_rank)
        if softmax_temperature is None:
            softmax_temperature = 1.0
        self.softmax_temperature = softmax_temperature
        self.rank_of_lower_rank = None

    def default_dropout(self):
        # n_fms_in_extra_fcs: List of integers, 1 per layer in the final classification path, except final classif layer
        n_extra_fcs = len(self.n_FMs_per_layer)
        if n_extra_fcs > 0:
            dropout_for_each_l_including_classifier = [0.0] + [0.5] * (n_extra_fcs - 1) + [0.5]
        else:
            dropout_for_each_l_including_classifier = [0.5]
        return dropout_for_each_l_including_classifier


class ModelConfig:
    def __init__(
        self,
        n_classes: int,
        n_input_channels: int,
        normal_pathway_config: PathWayConfig,
        use_subsampled_path: bool,
        subsampled_pathway_config: SubsampledPathwayConfig,
        fc_layers_config: FCLayersConfig,
        segment_dim_train: List[int],
        model_name: str = None,
        activation_function: str = None,
        conv_w_init_type: List = None,
        n_batches_for_bn_mov_avg: int = None,
        segment_dim_val: List[int] = None,
        segment_dim_inference: List[int] = None,
    ):
        self.model_name = model_name if model_name is not None else "deepmedic"
        self.n_classes = n_classes
        self.n_input_channels = n_input_channels
        self.normal_pathway_config = normal_pathway_config
        self.use_subsampled_path = use_subsampled_path
        self.subsampled_pathway_config = subsampled_pathway_config
        self.fc_layers_config = fc_layers_config
        self.activation_function = activation_function if activation_function is not None else "prelu"
        # Initialization
        self.conv_w_init_type = conv_w_init_type if conv_w_init_type is not None else ["fanIn", 2]
        # Batch Normalization
        self.n_batches_for_bn_mov_avg = n_batches_for_bn_mov_avg if n_batches_for_bn_mov_avg is not None else 60
        self.segment_dim_train = segment_dim_train
        if segment_dim_val is None:
            segment_dim_val = calc_rec_field_of_path_assuming_strides_1(
                self.normal_pathway_config.kernel_dims_per_layer
            )
        self.segment_dim_val = segment_dim_val
        if segment_dim_inference is None:
            segment_dim_inference = self.segment_dim_train
        self.segment_dim_inference = segment_dim_inference
