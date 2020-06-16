from typing import List


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
        n_fm_subs: List[int] = None,
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
