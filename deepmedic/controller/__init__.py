from deepmedic.config.input import InputModelConfig
from deepmedic.config.model import ModelConfig, PathWayConfig, SubsampledPathwayConfig, FCLayersConfig


def input_to_model_config(input_model_config: InputModelConfig) -> ModelConfig:
    normal_pathway_config = PathWayConfig(
        n_FMs_per_layer=input_model_config.n_fm_norm,
        kernel_dims_per_layer=input_model_config.kern_dim_norm,
        pad_mode_per_layer=input_model_config.pad_mode_norm,
        res_conn=input_model_config.resid_conn_layers_norm,
        lower_rank=input_model_config.lower_rank_layers_norm,
    )

    use_subs_paths = input_model_config.use_subsampled if input_model_config.use_subsampled is not None else False
    if not use_subs_paths:
        subsampled_pathway_config = SubsampledPathwayConfig([], [], [], [], use_subsampled_path=use_subs_paths)
    else:
        if input_model_config.n_fm_subs is not None:
            n_fms_per_l_subs = input_model_config.n_fm_subs
        else:
            n_fms_per_l_subs = normal_pathway_config.n_FMs_per_layer
        if all(not isinstance(e, list) for e in n_fms_per_l_subs):
            n_fms_per_l_subs = [n_fms_per_l_subs]
        n_layers_subs = len(n_fms_per_l_subs[0])
        if input_model_config.kern_dim_subs is None and n_layers_subs == len(
                normal_pathway_config.n_FMs_per_layer):
            kern_dims_per_l_subs = normal_pathway_config.kernel_dims_per_layer
        else:
            kern_dims_per_l_subs = input_model_config.kern_dim_subs
        # Residuals and lower ranks.
        res_conn_at_layers_subs = (
            input_model_config.resid_conn_layers_subs
            if input_model_config.resid_conn_layers_subs is not None
            else normal_pathway_config.res_conn
        )
        lower_rank_layers_subs = (
            input_model_config.lower_rank_layers_subs
            if input_model_config.lower_rank_layers_subs is not None
            else normal_pathway_config.lower_rank
        )
        subsampled_pathway_config = SubsampledPathwayConfig(
            n_FMs_per_layer=n_fms_per_l_subs,
            kernel_dims_per_layer=kern_dims_per_l_subs,
            subsample_factors=input_model_config.subs_factor,
            pad_mode_per_layer=input_model_config.pad_mode_subs,
            use_subsampled_path=use_subs_paths,
            res_conn=res_conn_at_layers_subs,
            lower_rank=lower_rank_layers_subs,
        )
    fc_layers_config = FCLayersConfig(
        n_FMs_per_layer=input_model_config.n_fm_fc,
        kernel_dims_per_layer=input_model_config.kern_dim_fc,
        pad_mode_per_layer=input_model_config.pad_mode_fc,
        res_conn=input_model_config.resid_conn_layers_fc,
    )

    return ModelConfig(
        model_name=input_model_config.model_name,
        n_classes=input_model_config.n_classes,
        n_input_channels=input_model_config.n_input_channels,
        normal_pathway_config=normal_pathway_config,
        use_subsampled_path=use_subs_paths,
        subsampled_pathway_config=subsampled_pathway_config,
        fc_layers_config=fc_layers_config,
        activation_function=input_model_config.activ_func,
        conv_w_init_type=input_model_config.conv_w_init,
        n_batches_for_bn_mov_avg=input_model_config.bn_roll_av_batches,
        segment_dim_train=input_model_config.seg_dim_train,
        segment_dim_val=input_model_config.seg_dim_val,
        segment_dim_inference=input_model_config.seg_dim_infer
    )
