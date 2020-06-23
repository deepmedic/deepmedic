from typing import List
from pathlib import Path
from collections import defaultdict
from deepmedic.config.model import ModelConfig, PathWayConfig, SubsampledPathwayConfig, FCLayersConfig


class InputModelConfig:
    def __init__(
        self,
        model_name: str = None,
        n_classes: int = None,
        n_input_channels: int = None,
        n_fm_norm: List[int] = None,
        kern_dim_norm: List[List[int]] = None,
        pad_mode_norm: List[str] = None,
        resid_conn_layers_norm: List[int] = None,
        lower_rank_layers_norm: List[int] = None,
        use_subsampled: bool = None,
        n_fm_subs: List[List[int]] = None,
        kern_dim_subs: List[List[int]] = None,
        pad_mode_subs: List[str] = None,
        subs_factor: List[List[int]] = None,
        resid_conn_layers_subs: List[int] = None,
        lower_rank_layers_subs: List[int] = None,
        n_fm_fc: List[int] = None,
        kern_dim_fc: List[List[int]] = None,
        pad_mode_fc: List[str] = None,
        resid_conn_layers_fc: List[int] = None,
        seg_dim_train: List[int] = None,
        seg_dim_val: List[int] = None,
        seg_dim_infer: List[int] = None,
        drop_norm: List[List[int]] = None,
        drop_subs: List[List[int]] = None,
        drop_fc: List[List[int]] = None,
        conv_w_init: List = None,
        activ_func: str = None,
        bn_roll_av_batches: int = None,
    ):
        self.model_name = model_name
        self.n_classes = n_classes
        self.n_input_channels = n_input_channels
        # norm
        self.n_fm_norm = n_fm_norm
        self.kern_dim_norm = kern_dim_norm
        self.pad_mode_norm = pad_mode_norm
        self.resid_conn_layers_norm = resid_conn_layers_norm
        self.lower_rank_layers_norm = lower_rank_layers_norm
        self.drop_norm = drop_norm

        # subsampled
        self.use_subsampled = use_subsampled
        self.n_fm_subs = n_fm_subs
        self.kern_dim_subs = kern_dim_subs
        self.pad_mode_subs = pad_mode_subs
        self.subs_factor = subs_factor
        self.resid_conn_layers_subs = resid_conn_layers_subs
        self.lower_rank_layers_subs = lower_rank_layers_subs
        self.drop_subs = drop_subs
        # fc
        self.n_fm_fc = n_fm_fc
        self.kern_dim_fc = kern_dim_fc
        self.pad_mode_fc = pad_mode_fc
        self.resid_conn_layers_fc = resid_conn_layers_fc
        self.drop_fc = drop_fc

        self.seg_dim_train = seg_dim_train
        self.seg_dim_val = seg_dim_val
        self.seg_dim_infer = seg_dim_infer
        self.conv_w_init = conv_w_init
        self.activ_func = activ_func
        self.bn_roll_av_batches = bn_roll_av_batches

    @classmethod
    def from_cfg_file(cls, cfg_path: Path) -> "InputModelConfig":
        cfg = defaultdict(lambda: None)
        exec(open(str(cfg_path)).read(), cfg)
        if cfg["numberFMsPerLayerSubsampled"] is not None:
            n_fms_per_l_subs = cfg["numberFMsPerLayerSubsampled"]
        else:
            n_fms_per_l_subs = cfg["numberFMsPerLayerNormal"]
        if all(not isinstance(e, list) for e in n_fms_per_l_subs):
            n_fms_per_l_subs = [n_fms_per_l_subs]
        subs_factors = cfg["subsampleFactor"]
        if subs_factors is not None:
            if all(not isinstance(e, list) for e in subs_factors):
                subs_factors = [subs_factors]
        input_config = cls(
            model_name=cfg["modelName"],
            n_classes=cfg["numberOfOutputClasses"],
            n_input_channels=cfg["numberOfInputChannels"],
            n_fm_norm=cfg["numberFMsPerLayerNormal"],
            kern_dim_norm=cfg["kernelDimPerLayerNormal"],
            pad_mode_norm=cfg["padTypePerLayerNormal"],
            resid_conn_layers_norm=cfg["layersWithResidualConnNormal"],
            lower_rank_layers_norm=cfg["lowerRankLayersNormal"],
            use_subsampled=cfg["useSubsampledPathway"],
            n_fm_subs=n_fms_per_l_subs,
            kern_dim_subs=cfg["kernelDimPerLayerSubsampled"],
            pad_mode_subs=cfg["padTypePerLayerSubsampled"],
            subs_factor=subs_factors,
            resid_conn_layers_subs=cfg["layersWithResidualConnSubsampled"],
            lower_rank_layers_subs=cfg["lowerRankLayersSubsampled"],
            n_fm_fc=cfg["numberFMsPerLayerFC"],
            kern_dim_fc=cfg["kernelDimPerLayerFC"],
            pad_mode_fc=cfg["padTypePerLayerFC"],
            resid_conn_layers_fc=cfg["layersWithResidualConnFC"],
            seg_dim_train=cfg["segmentsDimTrain"],
            seg_dim_val=cfg["segmentsDimVal"],
            seg_dim_infer=cfg["segmentsDimInference"],
            drop_norm=cfg["dropoutRatesNormal"],
            drop_subs=cfg["dropoutRatesSubsampled"],
            drop_fc=cfg["dropoutRatesFc"],
            conv_w_init=cfg["convWeightsInit"],
            activ_func=cfg["activationFunction"],
            bn_roll_av_batches=cfg["rollAverageForBNOverThatManyBatches"],
        )
        return input_config

    def to_model_config(self) -> ModelConfig:
        res_conn_norm = (
            [i - 1 for i in self.resid_conn_layers_norm] if self.resid_conn_layers_norm is not None else None
        )
        normal_pathway_config = PathWayConfig(
            n_FMs_per_layer=self.n_fm_norm,
            kernel_dims_per_layer=self.kern_dim_norm,
            pad_mode_per_layer=self.pad_mode_norm,
            res_conn=res_conn_norm,
            lower_rank=self.lower_rank_layers_norm,
        )

        use_subs_paths = self.use_subsampled if self.use_subsampled is not None else False
        if not use_subs_paths:
            subsampled_pathway_configs = None
        else:
            # n_fms_per_l_subs is list of list of int
            n_fms_per_l_subs = self.n_fm_subs
            n_layers_subs = len(n_fms_per_l_subs[0])
            if self.kern_dim_subs is None and n_layers_subs == len(normal_pathway_config.n_FMs_per_layer):
                kern_dims_per_l_subs = normal_pathway_config.kernel_dims_per_layer
            else:
                kern_dims_per_l_subs = self.kern_dim_subs
            subsample_factors = self.subs_factor
            if self.subs_factor is None:
                subsample_factors = [None]
            for _ in range(len(subsample_factors) - len(n_fms_per_l_subs)):
                n_fms_per_l_in_prev_path = n_fms_per_l_subs[-1]
                n_fms_per_l_subs.append([max(1, int(n_fms_in_l_i)) for n_fms_in_l_i in n_fms_per_l_in_prev_path])
            # Residuals and lower ranks.
            res_conn_at_layers_subs = (
                [i - 1 for i in self.resid_conn_layers_subs]
                if self.resid_conn_layers_subs is not None
                else normal_pathway_config.res_conn
            )
            lower_rank_layers_subs = (
                self.lower_rank_layers_subs
                if self.lower_rank_layers_subs is not None
                else normal_pathway_config.lower_rank
            )
            subsampled_pathway_configs = []
            for subs_factor, n_fms in zip(subsample_factors, n_fms_per_l_subs):
                subsampled_pathway_config = SubsampledPathwayConfig(
                    n_FMs_per_layer=n_fms,
                    kernel_dims_per_layer=kern_dims_per_l_subs,
                    subsample_factor=subs_factor,
                    pad_mode_per_layer=self.pad_mode_subs,
                    res_conn=res_conn_at_layers_subs,
                    lower_rank=lower_rank_layers_subs,
                )
                subsampled_pathway_configs.append(subsampled_pathway_config)
        res_conn_fc = (
            [i - 1 for i in self.resid_conn_layers_fc] if self.resid_conn_layers_fc is not None else None
        )
        fc_layers_config = FCLayersConfig(
            n_FMs_per_layer=self.n_fm_fc,
            kernel_dims_per_layer=self.kern_dim_fc,
            pad_mode_per_layer=self.pad_mode_fc,
            res_conn=res_conn_fc,
        )

        return ModelConfig(
            model_name=self.model_name,
            n_classes=self.n_classes,
            n_input_channels=self.n_input_channels,
            normal_pathway_config=normal_pathway_config,
            use_subsampled_path=use_subs_paths,
            subsampled_pathway_configs=subsampled_pathway_configs,
            fc_layers_config=fc_layers_config,
            activation_function=self.activ_func,
            conv_w_init_type=self.conv_w_init,
            n_batches_for_bn_mov_avg=self.bn_roll_av_batches,
            segment_dim_train=self.seg_dim_train,
            segment_dim_val=self.seg_dim_val,
            segment_dim_inference=self.seg_dim_infer,
        )
